"""Structural regularization: a fixed DIN backbone and learned interest branch."""
from src.ranking_v3.model import NetworkDIN


class ProtectedDIN(NetworkDIN):
    def __init__(self, num_users, num_items, config, retriever):
        if config['variant'] not in {'gated', 'gated_copy_safe'}:
            raise ValueError('Protected backbone requires an actual trainable interest branch')
        super().__init__(num_users, num_items, config, retriever)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith('interest_'))

    def train(self, mode=True):
        super().train(mode)
        # Frozen weights alone are insufficient: BN buffers and Dropout also
        # change the backbone predictor unless its entire core stays in eval.
        self.core.eval()
        self.tab_embedding.eval()
        self.rank_delta.eval()
        self.repeat_linear.eval()
        return self


def build_model(num_users, num_items, config, teacher, initial_state=None):
    model = ProtectedDIN(num_users, num_items, config, teacher)
    if initial_state is not None:
        loaded = model.load_state_dict(initial_state, strict=False)
        allowed = {name for name in model.state_dict()
                   if name.startswith('interest_') or name.endswith('.alpha')}
        if loaded.unexpected_keys or set(loaded.missing_keys) - allowed:
            raise ValueError('Initial checkpoint does not match the protected DIN backbone')
    return model
