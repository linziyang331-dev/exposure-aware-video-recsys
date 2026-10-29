"""Continue a frozen initial DIN with explicit network-only experiment evidence."""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.data.download import digest_file
from src.retrieval.data import tensor_batch
from src.retrieval_v2.model import TwoTowerV2
from src.retrieval_v2.train import parameter_digest
from src.ranking_v2.data import choice_batch
from src.ranking_v2.evaluate import assess
from src.ranking_v2.train import code_hashes as previous_hashes
from src.ranking_v3.model import build_model


def implementation_hashes():
    return {'previous': previous_hashes(), 'network': {
        p.name: digest_file(p) for p in Path(__file__).parent.glob('*.py')}}


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_model(path, device='cuda'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    arch = checkpoint['architecture']
    teacher = TwoTowerV2(**arch['teacher_spec']).to(device)
    config = arch['config'] | {'device': str(device)}
    model = build_model(arch['num_users'], arch['num_items'], config, teacher)
    model.load_state_dict(checkpoint['state'], strict=True)
    return model.eval(), checkpoint


def train_run(train, validation, train_candidates, validation_candidates,
              num_users, num_items, config, directory, seed, teacher, initial_state):
    if config.get('objective') != 'choice' or not config.get('residual'):
        raise ValueError('This experiment fixes the residual choice objective')
    if not np.isin(train['label'], [0, 1]).all():
        raise ValueError('Observed binary auxiliary labels required')
    if not 0 < config['competitors'] < train_candidates['ids'].shape[1]:
        raise ValueError('Invalid competitor count')
    teacher_before = parameter_digest(teacher)
    signature = {'config': config, 'seed': seed, 'train': train.metadata['signature'],
                 'validation': validation.metadata['signature'],
                 'train_candidates': train_candidates.metadata['signature'],
                 'validation_candidates': validation_candidates.metadata['signature'],
                 'teacher': teacher_before, 'initial_state': state_digest(initial_state),
                 'implementation': implementation_hashes()}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'training.json'
    path = directory / 'best.pt'
    if result_path.exists():
        previous = json.loads(result_path.read_text(encoding='utf-8'))
        if previous['fingerprint'] != fingerprint or digest_file(path) != previous['checkpoint_sha256']:
            raise ValueError('Completed network experiment changed; use a new directory')
        return previous
    if path.exists():
        raise FileExistsError('Incomplete run preserved; use a new experiment directory')

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(config['device'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model = build_model(num_users, num_items, config, teacher, initial_state)
    if parameter_digest(model.teacher) != teacher_before:
        raise ValueError('Initial DIN and declared frozen retriever disagree')
    initial = parameter_digest(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    repeat_parameters = list(model.repeat_linear.parameters())
    repeat_ids = {id(p) for p in repeat_parameters}
    optimizer = torch.optim.Adam([
        {'params': [p for p in trainable if id(p) not in repeat_ids]},
        {'params': repeat_parameters, 'lr': config['learning_rate'] * config.get('repeat_lr_multiplier', 10.)}
    ], lr=config['learning_rate'], weight_decay=config['weight_decay'])
    arch = {'num_users': num_users, 'num_items': num_items, 'config': config,
            'teacher_spec': model.teacher_spec()}
    probe = tensor_batch(validation, np.arange(min(16, len(validation))), device)

    def save(epoch):
        model.eval()
        with torch.no_grad():
            expected = model(probe['users'], probe['items'], probe['tabs'], probe['history'])['rank'].cpu()
        torch.save({'architecture': arch, 'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'epoch': epoch, 'seed': seed, 'fingerprint': fingerprint, 'reload_probe': expected}, path)

    start_time = time.perf_counter()
    baseline = assess(model, validation, validation_candidates, config, directory / 'baseline_ranks.npz')
    best_metrics = baseline; best_epoch = -1; best = baseline['ranking']['ndcg@20']
    save(-1)
    print(f"{directory.name} baseline N20={best:.7f} R20={baseline['ranking']['recall@20']:.7f}", flush=True)
    curve = [{'epoch': -1, 'validation': baseline}]
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(train_candidates['v2_ranks'] > 0)
    stale = updates = 0
    for epoch in range(config['epochs']):
        model.train(); epoch_start = time.perf_counter(); totals = {}; steps = 0
        order = rng.choice(eligible, min(config['queries_per_epoch'], len(eligible)), replace=False)
        for offset in range(0, len(order), config['query_batch_size']):
            optimizer.zero_grad(set_to_none=True)
            choice = choice_batch(train, train_candidates, order[offset:offset + config['query_batch_size']],
                                  rng, config['competitors'], config['mask_past'])
            b = tensor_batch(train, choice['indices'], device)
            q, k = choice['items'].shape
            aux = tensor_batch(train, rng.integers(0, len(train), size=config['aux_batch_size']), device)
            users = torch.cat([b['users'].repeat_interleave(k), aux['users']])
            tabs = torch.cat([b['tabs'].repeat_interleave(k), aux['tabs']])
            history = torch.cat([b['history'].repeat_interleave(k, dim=0), aux['history']])
            items = torch.cat([torch.as_tensor(choice['items'].ravel(), device=device, dtype=torch.long), aux['items']])
            with torch.no_grad():
                au = model.teacher.encode_user(aux['users'], aux['tabs'], aux['history'])
                ai = model.teacher.encode_item(aux['items'])
                auxiliary_base = (au * ai).sum(1) / model.teacher.temperature
            canonical = torch.cat([torch.as_tensor(choice['base_scores'].ravel(), device=device), auxiliary_base])
            output = model(users, items, tabs, history, base_scores=canonical)
            choice_loss = F.cross_entropy(output['rank'][:q*k].reshape(q, k), torch.zeros(q, device=device, dtype=torch.long))
            bce = F.binary_cross_entropy_with_logits(output['ctr'][q*k:], aux['labels'])
            regularization = output['delta'][:q*k].square().mean()
            loss = choice_loss + config['aux_weight'] * bce + config['delta_reg'] * regularization
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite network loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 5.)
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite network gradient')
            optimizer.step(); updates += 1; steps += 1
            for name, value in [('choice', choice_loss), ('observed_bce', bce), ('delta_square', regularization)]:
                totals[name] = totals.get(name, 0.) + value.item()
            if steps % 50 == 0:
                print(f'{directory.name} epoch={epoch} step={steps} queries={min(offset+q,len(order))}', flush=True)
        if device.type == 'cuda': torch.cuda.synchronize()
        seconds = time.perf_counter() - epoch_start
        metrics = assess(model, validation, validation_candidates, config, directory / f'epoch{epoch}_ranks.npz')
        row = {'epoch': epoch, 'losses': {k: v / steps for k, v in totals.items()},
               'validation': metrics, 'train_seconds': seconds,
               'repeat_weight': float(model.repeat_linear.weight.detach().item())}
        curve.append(row)
        score = metrics['ranking']['ndcg@20']
        print(f"{directory.name} epoch={epoch} N20={score:.7f} R20={metrics['ranking']['recall@20']:.7f} train_s={seconds:.2f}", flush=True)
        if score > best:
            best = score; best_metrics = metrics; best_epoch = epoch; stale = 0; save(epoch)
        else: stale += 1
        (directory / 'learning_curve.json').write_text(json.dumps(curve, indent=2), encoding='utf-8')
        if stale >= config['patience']: break

    final_digest = parameter_digest(model)
    teacher_unchanged = parameter_digest(model.teacher) == teacher_before and all(p.grad is None for p in model.teacher.parameters())
    if not teacher_unchanged: raise ValueError('Frozen retriever changed')
    restored, checkpoint = load_model(path, device)
    with torch.no_grad():
        actual = restored(probe['users'], probe['items'], probe['tabs'], probe['history'])['rank'].cpu()
    torch.testing.assert_close(actual, checkpoint['reload_probe'].cpu(), rtol=0, atol=0)
    result = {'status': 'TRAINED', 'config': config, 'seed': seed, 'signature': signature,
              'fingerprint': fingerprint, 'baseline_validation': baseline,
              'best_epoch': best_epoch, 'best_validation': best_metrics, 'curve': curve,
              'selected_baseline': best_epoch == -1, 'optimization_changed_parameters': initial != final_digest,
              'teacher_unchanged': teacher_unchanged, 'reload_identical': True,
              'checkpoint': str(path.resolve()), 'checkpoint_sha256': digest_file(path),
              'initial_parameter_sha256': initial, 'trained_parameter_sha256': parameter_digest(restored),
              'optimizer_updates': updates, 'elapsed_seconds': time.perf_counter() - start_time,
              'peak_torch_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None}
    result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
