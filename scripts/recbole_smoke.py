"""Actual GPU train/evaluate/reload smoke run on RecBole's bundled MovieLens."""
import argparse
import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import psutil
import torch

from src.compat.recbole import apply_compatibility, load_run_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/recbole_smoke.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/recbole_smoke')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Output must stay inside the project directory')
    output.mkdir(parents=True, exist_ok=True)
    log = output / 'run.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log, encoding='utf-8'), logging.StreamHandler(sys.stdout)], force=True)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU validation requires CUDA; CPU fallback is not accepted')
    torch.set_num_threads(4)
    apply_compatibility()
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_seed

    config = Config(config_file_list=[str(args.config)], config_dict={
        'checkpoint_dir': str(output / 'checkpoints'), 'show_progress': False,
    })
    init_seed(config['seed'], config['reproducibility'])
    assert str(config['device']).startswith('cuda'), config['device']
    start = time.perf_counter()
    dataset = create_dataset(config)
    train, valid, test = data_preparation(config, dataset)
    # Check actual boundaries per user, allowing equal timestamps in this example.
    boundary_checks = 0
    uid, timestamp = config['USER_ID_FIELD'], config['TIME_FIELD']
    for left, right in [(train.dataset, valid.dataset), (valid.dataset, test.dataset)]:
        a, b = left.inter_feat, right.inter_feat
        for user in torch.unique(b[uid]).tolist():
            before, after = a[timestamp][a[uid] == user], b[timestamp][b[uid] == user]
            if len(before) and len(after):
                assert before.max() <= after.min(), f'Time order violated for user {user}'
                boundary_checks += 1
    logging.info('Actual split sizes: %s', [len(part.dataset) for part in [train, valid, test]])
    torch.cuda.reset_peak_memory_stats()
    model_class = get_model(config['model'])
    model = model_class(config, train.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    train_start = time.perf_counter()
    best_score, valid_result = trainer.fit(train, valid, saved=True, show_progress=False)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start
    checkpoint_path = Path(trainer.saved_model_file)
    checkpoint = load_run_checkpoint(checkpoint_path, output, device=config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    model.eval()
    test_start = time.perf_counter()
    test_result = trainer.evaluate(test, load_best_model=False, show_progress=False)
    torch.cuda.synchronize()
    test_seconds = time.perf_counter() - test_start
    probe = train.dataset.inter_feat[:32].to(config['device'])
    with torch.no_grad():
        expected = model.predict(probe)
    restored = model_class(config, train.dataset).to(config['device'])
    restored.load_state_dict(checkpoint['state_dict'])
    restored.load_other_parameter(checkpoint.get('other_parameter'))
    restored.eval()
    with torch.no_grad():
        actual = restored.predict(probe)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    trainer.model = restored
    reloaded_result = trainer.evaluate(test, load_best_model=False, show_progress=False)
    assert dict(test_result) == dict(reloaded_result), 'Reload changed evaluation metrics'
    for value in list(valid_result.values()) + list(test_result.values()):
        assert math.isfinite(float(value)) and 0 <= float(value) <= 1
    losses = {str(k): float(v) for k, v in trainer.train_loss_dict.items()}
    assert losses and all(math.isfinite(value) for value in losses.values())
    sample = Path(config['data_path']) / (config['dataset'] + '.inter')
    result = {
        'status': 'PASS',
        'timestamp': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'purpose': 'Framework smoke test only; NOT KuaiRand or a project benchmark',
        'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
        'model': config['model'], 'dataset': config['dataset'], 'seed': config['seed'],
        'device': str(config['device']), 'gpu': torch.cuda.get_device_name(0),
        'users_including_padding': dataset.user_num, 'items_including_padding': dataset.item_num,
        'interactions': len(dataset),
        'split_rows': {name: len(part.dataset) for name, part in zip(['train', 'valid', 'test'], [train, valid, test])},
        'split': 'per-user time-ordered 80/10/10; equal timestamps allowed',
        'time_boundary_checks_passed': boundary_checks,
        'sample_inter_sha256': hashlib.sha256(sample.read_bytes()).hexdigest(),
        'train_loss_by_epoch': losses,
        'best_valid_score': float(best_score), 'best_valid_result': dict(valid_result),
        'test_result': dict(test_result), 'reloaded_test_result': dict(reloaded_result),
        'checkpoint_relative_path': str(checkpoint_path.resolve().relative_to(ROOT)),
        'checkpoint_reload_predictions_identical': True,
        'checkpoint_reload_metrics_identical': True,
        'train_seconds': train_seconds, 'test_seconds': test_seconds,
        'total_seconds': time.perf_counter() - start,
        'peak_torch_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
        'process_rss_MiB_at_end': psutil.Process().memory_info().rss / 2**20,
    }
    result_path = output / 'result.json'
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    trainer.tensorboard.close()
    logging.info('PASS: results saved to %s', result_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
