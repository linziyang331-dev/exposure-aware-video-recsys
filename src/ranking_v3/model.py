"""Warm-start DIN with trainable Dice and candidate-conditioned interest gates."""
import math

import torch
from torch import nn
from recbole.model.layers import Dice

from src.ranking.model import recent_history
from src.ranking_v2.model import RichDIN


class TrainableDice(nn.Module):
    """Keep the upstream activation formula, registering its slope as a parameter.

    The existing MLP already applies BatchNorm before this activation. Adding a
    second normalization here would change the zero-alpha warm-start behavior.
    """
    def __init__(self, width):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(width))

    def forward(self, score):
        probability = torch.sigmoid(score)
        return self.alpha * (1 - probability) * score + probability * score


def trainable_dice(module):
    for name, child in list(module.named_children()):
        if isinstance(child, Dice):
            replacement = TrainableDice(child.alpha.numel()).to(child.alpha.device)
            with torch.no_grad():
                replacement.alpha.copy_(child.alpha)
            setattr(module, name, replacement)
        else:
            trainable_dice(child)


def masked_weights(logits, mask):
    """A distribution over valid keys, or exactly zero for an empty history."""
    weights = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=1)
    weights = weights * mask.to(logits.dtype)
    return weights / weights.sum(1, keepdim=True).clamp_min(1e-8)


class NetworkDIN(RichDIN):
    def __init__(self, num_users, num_items, config, retriever):
        super().__init__(num_users, num_items, config, retriever)
        trainable_dice(self.core.dnn_mlp_layers)
        self.gated = config['variant'] in {'gated', 'gated_copy_safe'}
        if self.gated:
            dim = config['embedding_dim']
            self.interest_query = nn.Linear(2 * dim, dim, bias=False)
            self.interest_key = nn.Linear(dim, dim, bias=False)
            self.interest_energy = nn.Linear(dim, 1, bias=False)
            self.interest_position = nn.Embedding(config['history_length'], 1)
            nn.init.zeros_(self.interest_position.weight)
            self.interest_gate = nn.Linear(4 * dim + 1, dim)
            width = config.get('branch_hidden', 64)
            self.interest_network = nn.Sequential(
                nn.Linear(8 * dim + 11, width), nn.LayerNorm(width),
                nn.PReLU(width), nn.Dropout(config['dropout']),
                nn.Linear(width, config['hidden_sizes'][-1]),
                nn.LayerNorm(config['hidden_sizes'][-1]),
                nn.PReLU(config['hidden_sizes'][-1]))
            self.interest_output = nn.Linear(config['hidden_sizes'][-1], 1)
            nn.init.zeros_(self.interest_output.weight)
            nn.init.zeros_(self.interest_output.bias)
        self.to(config['device'])

    def interest_details(self, users, items, tabs, history):
        if not self.gated:
            raise ValueError('Interest gate is absent in this control')
        short = recent_history(history, self.config['history_length'])
        # The adapter exposes one ID field per table; its field offset is zero.
        # Access the actual Embedding to avoid the upstream field axis wrapper.
        item_table = self.core.embedding_layer.token_embedding_table['item'].embedding
        user_table = self.core.embedding_layer.token_embedding_table['user'].embedding
        keys = item_table(short)
        target = item_table(items)
        user = user_table(users)
        if not self.config['user_features']:
            user = torch.zeros_like(user)
        valid = short > 0
        length = valid.sum(1)
        age = (length[:, None] - 1 - torch.arange(short.shape[1], device=short.device)).clamp_min(0)
        query = self.interest_query(torch.cat([user, target], dim=1))
        logits = self.interest_energy(torch.tanh(self.interest_key(keys) + query[:, None])).squeeze(-1)
        logits = logits + self.interest_position(age).squeeze(-1)
        if self.config['variant'] == 'gated_copy_safe':
            valid = valid & (short != items[:, None])
        long_weights = masked_weights(logits, valid)
        short_weights = masked_weights(logits, valid & (age < self.config.get('short_history', 5)))
        long_interest = (long_weights[:, :, None] * keys).sum(1)
        short_interest = (short_weights[:, :, None] * keys).sum(1)
        length_feature = torch.log1p(length.float())[:, None] / math.log1p(self.config['history_length'])
        gate = torch.sigmoid(self.interest_gate(torch.cat([user, target, long_interest, short_interest, length_feature], dim=1)))
        mixed = gate * short_interest + (1 - gate) * long_interest
        return {'user': user, 'target': target, 'long': long_interest, 'short': short_interest,
                'mixed': mixed, 'gate': gate, 'long_weights': long_weights,
                'short_weights': short_weights, 'length_feature': length_feature}

    def forward(self, users, items, tabs, history, base_scores=None):
        out = super().forward(users, items, tabs, history, base_scores)
        if self.gated:
            d = self.interest_details(users, items, tabs, history)
            seen = ((history == items[:, None]) & (history > 0)).any(1).float()[:, None]
            if not self.config.get('repeat_feature', True):
                seen = torch.zeros_like(seen)
            features = torch.cat([d['user'], d['target'], d['long'], d['short'], d['mixed'],
                                  d['user'] * d['target'], d['mixed'] * d['target'],
                                  (d['mixed'] - d['target']).abs(), self.tab_embedding(tabs),
                                  seen, d['length_feature'], out['base'][:, None]], dim=1)
            correction = self.interest_output(self.interest_network(features)).squeeze(1)
            out['rank'] = out['rank'] + correction
            out['delta'] = out['delta'] + correction
            out['interest_delta'] = correction
        return out


def build_model(num_users, num_items, config, teacher, initial_state=None):
    variant = config['variant']
    if variant not in {'v2_continued', 'dice', 'gated', 'gated_copy_safe'}:
        raise ValueError('Unknown Top20 network variant')
    cls = RichDIN if variant == 'v2_continued' else NetworkDIN
    model = cls(num_users, num_items, config, teacher)
    if initial_state is not None:
        loaded = model.load_state_dict(initial_state, strict=False)
        allowed = {name for name in model.state_dict()
                   if name.startswith('interest_') or name.endswith('.alpha')}
        if loaded.unexpected_keys or set(loaded.missing_keys) - allowed:
            raise ValueError('Initial checkpoint does not match the existing DIN backbone')
    return model
