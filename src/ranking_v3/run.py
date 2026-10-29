"""Bounded network comparison, frozen selection, replication and Top20 evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.common import ROOT
from src.data.download import digest_file
from src.ranking.data import write_json
from src.ranking.evaluate import rank_scores
from src.ranking.metrics import exposure_metrics
from src.ranking_v2.inference import candidate_scores, exposure_logits, deterministic
from src.ranking_v2.run import setup as previous_setup
from src.ranking_v2.train import load_model as load_previous
from src.retrieval.statistics import paired_user_bootstrap
from src.retrieval.metrics import summarize_ranks
from src.retrieval_v2.provenance import freeze_evidence, verify_evidence
from src.ranking_v3.train import train_run, load_model

VARIANTS = ('v2_continued', 'dice', 'gated', 'gated_copy_safe')


def choose_structure(trials):
    baseline = trials['v2_continued']['baseline_validation']['ranking']
    control = trials['v2_continued']['best_validation']['ranking']
    qualified = [(name, trial) for name, trial in trials.items()
                 if name != 'v2_continued'
                 and trial['best_validation']['ranking']['ndcg@20'] > max(baseline['ndcg@20'], control['ndcg@20'])
                 and trial['best_validation']['ranking']['recall@20'] >= baseline['recall@20']]
    return max(qualified, key=lambda pair: pair[1]['best_validation']['ranking']['ndcg@20']) if qualified else None


def validate_rank_artifact(path, candidates, metrics):
    with np.load(path) as z:
        np.testing.assert_array_equal(z['event_ids'], candidates['event_ids'])
        np.testing.assert_array_equal(z['users'], candidates['user_ids'])
        ranks = z['ranks']; users = z['users']
    present = candidates['v2_ranks'] > 0
    if ((ranks < 0) | (ranks > candidates['ids'].shape[1])).any():
        raise ValueError('Candidate rank outside the unchanged pool')
    np.testing.assert_array_equal(ranks > 0, present)
    rebuilt = summarize_ranks(ranks, users, ks=(10, 20, 100))
    rebuilt['mrr@100'] = rebuilt.pop('mrr')
    rebuilt['user_macro']['mrr@100'] = rebuilt['user_macro'].pop('mrr')
    rebuilt['retrieval_miss_queries'] = int((~present).sum())
    rebuilt['conditional_hit_metrics'] = (summarize_ranks(ranks[present], users[present], ks=(10, 20))
                                          if present.any() else {'queries': 0})
    if rebuilt != metrics:
        raise ValueError('Saved ranks and complete ranking metrics disagree')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_previous_results(root=ROOT):
    root = Path(root)
    report_path = root / 'reports/ranking_v2_artifact_validation.json'
    report = read(report_path)
    if report['status'] != 'PASS':
        raise ValueError('Original DIN artifact verification did not pass')
    names = ['experiments/ranking_v2/evaluation.json'] + [
        'experiments/ranking_v2/artifacts/ranks/din_seed42/' + p + '.npz'
        for p in ['standard_validation', 'standard_test', 'random_validation', 'random_test']]
    for name in names:
        if digest_file(root / name) != report['artifact_hashes'].get(name):
            raise ValueError('Original comparison artifact changed: ' + name)
    return [report_path] + [root / name for name in names]


def output_path(config, override=None):
    output = (ROOT / (override or config['output_directory'])).resolve()
    if output.drive.upper() != 'D:' or output.parent != (ROOT / 'experiments').resolve() or not output.name.startswith('ranking_v3_'):
        raise ValueError('Use a separate D: project experiments/ranking_v3_* directory')
    return output


def get_config(override=None):
    base = yaml.safe_load((ROOT / 'configs/ranking_v2.yaml').read_text(encoding='utf-8'))
    config = base | yaml.safe_load((ROOT / 'configs/ranking_v3.yaml').read_text(encoding='utf-8'))
    if config['primary_seed'] != 42 or config['seeds'] != [42, 43, 44]:
        raise ValueError('This predeclared experiment fixes primary seed42 and continuation seeds42/43/44')
    output = output_path(config, override)
    config['output_directory'] = output.relative_to(ROOT).as_posix()
    config.update(select_metric='ndcg@20', objective='choice', residual=True,
                  teacher_features=True, pretrained=True, neural_rank=True,
                  attention=True, repeat_feature=True)
    return config, output


def initial_model():
    selected = read(ROOT / 'experiments/ranking_v2/selection.json')
    verify_evidence(ROOT, selected['evidence'], selected['config'])
    run = selected['primary_training']
    if digest_file(run['checkpoint']) != run['checkpoint_sha256']:
        raise ValueError('Initial DIN checkpoint changed')
    model, _ = load_previous(run['checkpoint'], 'cuda')
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model.teacher, state, run


def close(caches, candidates):
    for cache in candidates.values(): cache.close()
    for cache in caches.values(): cache.close()


def fit(override=None):
    deterministic()
    config, output = get_config(override)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'selection.json').exists():
        raise ValueError('Selection frozen; do not add trials')
    previous_result_files = verify_previous_results()
    _, _, vocabulary, counts, caches, candidates, frozen = previous_setup()
    teacher, initial, source = initial_model()
    ledger_path = output / 'tuning.json'
    ledger = read(ledger_path) if ledger_path.exists() else {
        'status': 'VALIDATION_ONLY', 'test_partitions_loaded': False,
        'variants_predeclared': list(VARIANTS), 'config': config, 'trials': {},
        'initial_din_checkpoint': source['checkpoint'],
        'initial_din_checkpoint_sha256': source['checkpoint_sha256'],
        'v2_checkpoint_sha256': frozen['checkpoint_sha256']}
    if ledger['config'] != config or ledger['variants_predeclared'] != list(VARIANTS):
        raise ValueError('Network trial budget/configuration changed')
    try:
        for variant in VARIANTS:
            cfg = config | {'variant': variant}
            directory = output / 'artifacts/trials' / variant
            run = train_run(caches['standard/train'], caches['standard/validation'],
                            candidates['standard/train'], candidates['standard/validation'],
                            len(vocabulary['user_raw_to_index']), len(counts)-1,
                            cfg, directory, 42, teacher, initial)
            ledger['trials'][variant] = {
                'config': cfg, 'training_result_path': str((directory / 'training.json').resolve()),
                'checkpoint_sha256': run['checkpoint_sha256'], 'best_epoch': run['best_epoch'],
                'best_validation': run['best_validation'], 'baseline_validation': run['baseline_validation']}
            write_json(ledger_path, ledger)
        baseline = ledger['trials']['v2_continued']['baseline_validation']['ranking']
        control = ledger['trials']['v2_continued']['best_validation']['ranking']
        choice = choose_structure(ledger['trials'])
        if choice:
            name, trial = choice
            status = 'VALIDATION_STRUCTURE_SELECTED'
            selected_run = read(trial['training_result_path'])
        else:
            name, selected_run = None, None
            status = 'NO_STRUCTURAL_GAIN_RETAIN_EXISTING_DIN'
        files = list((ROOT / 'src/ranking_v3').glob('*.py')) + [
            ROOT / 'configs/ranking_v3.yaml', ledger_path, ROOT / 'experiments/ranking_v2/selection.json',
            Path(source['checkpoint']), ROOT / 'reports/retrieval_v2_freeze.json']
        files += [Path(t['training_result_path']) for t in ledger['trials'].values()]
        files += [Path(read(t['training_result_path'])['checkpoint']) for t in ledger['trials'].values()]
        files += previous_result_files
        selection = {'status': status, 'config': config, 'variant': name,
                     'primary_training': selected_run, 'baseline_validation': baseline,
                     'continued_control_validation': control, 'selection_uses': 'standard validation Top20 only',
                     'candidate_count': 100, 'candidate_membership_changed': False,
                     'manual_history_penalty_added': False, 'test_partitions_loaded': False,
                     'evidence': freeze_evidence(ROOT, config, files)}
        write_json(output / 'selection.json', selection)
        print('SELECTION ' + status + ' ' + str(name), flush=True)
        return selection
    finally:
        close(caches, candidates)


def replicate(override=None):
    deterministic(); config, output = get_config(override)
    selected = read(output / 'selection.json')
    verify_evidence(ROOT, selected['evidence'], config)
    if selected['variant'] is None:
        raise ValueError('No network structure selected; retain existing DIN')
    path = output / 'replication.json'
    if path.exists():
        result = read(path); verify_evidence(ROOT, result['evidence'], config); return result
    _, _, vocab, counts, caches, candidates, _ = previous_setup()
    teacher, initial, _ = initial_model()
    runs = {'seed42': selected['primary_training']}
    try:
        for seed in [43, 44]:
            runs[f'seed{seed}'] = train_run(caches['standard/train'], caches['standard/validation'],
                candidates['standard/train'], candidates['standard/validation'],
                len(vocab['user_raw_to_index']), len(counts)-1,
                config | {'variant': selected['variant']}, output / 'artifacts/replication' / f'seed{seed}',
                seed, teacher, initial)
        means = {key: float(np.mean([r['best_validation']['ranking'][key] for r in runs.values()]))
                 for key in ['ndcg@20', 'recall@20']}
        replicated_gain = (means['ndcg@20'] > selected['baseline_validation']['ndcg@20']
                           and means['recall@20'] >= selected['baseline_validation']['recall@20'])
        result = {'status': 'REPLICATED_CONTINUATION_SEEDS', 'runs': runs, 'validation_mean': means,
                  'validation_gain_replicated': replicated_gain,
                  'initial_checkpoint_shared_seed42': True,
                  'seed_scope': 'continuation sampling/dropout/optimizer randomness; same pretrained seed42 backbone',
                  'evidence': freeze_evidence(ROOT, config, [output / 'selection.json'] +
                       [Path(r['checkpoint']) for r in runs.values()] +
                       [output / 'artifacts/replication' / f'seed{s}' / 'training.json' for s in [43,44]])}
        write_json(path, result); return result
    finally:
        close(caches, candidates)


def evaluate(override=None):
    deterministic(); config, output = get_config(override)
    selected = read(output / 'selection.json'); rep = read(output / 'replication.json')
    verify_evidence(ROOT, selected['evidence'], config); verify_evidence(ROOT, rep['evidence'], config)
    if not rep['validation_gain_replicated']:
        raise ValueError('Continuation-seed mean did not retain the validation Top20 gain')
    if (output / 'evaluation.json').exists():
        raise ValueError('Already evaluated; use verify or report')
    parts = ('standard/validation', 'standard/test', 'random/validation', 'random/test')
    _, _, _, _, caches, candidates, _ = previous_setup(parts)
    old = read(ROOT / 'experiments/ranking_v2/evaluation.json')
    partial = output / 'evaluation_partial.json'
    result = read(partial) if partial.exists() else {
        'status': 'EVALUATING', 'selection_sha256': digest_file(output / 'selection.json'),
        'replication_sha256': digest_file(output / 'replication.json'),
        'test_is_iterative_reassessment': True, 'results': {}, 'paired_user_bootstrap': {}}
    if result['selection_sha256'] != digest_file(output / 'selection.json') or result['replication_sha256'] != digest_file(output / 'replication.json'):
        raise ValueError('Partial evaluation belongs to another selection')
    try:
        for name, run in rep['runs'].items():
            model, checkpoint = load_model(run['checkpoint'], config['device'])
            if digest_file(run['checkpoint']) != run['checkpoint_sha256']:
                raise ValueError('Selected network weight changed')
            for part in parts:
                if part in result['results'].get(name, {}): continue
                cache, cand = caches[part], candidates[part]
                stem = part.replace('/', '_')
                scores = candidate_scores(model, cache, cand, config['candidate_batch_size'])
                artifact = output / 'artifacts/ranks' / name / (stem + '.npz')
                ranking = rank_scores(cand, cache, scores, artifact)
                if ranking['recall@100'] != old['baseline'][part]['recall@100']:
                    raise ValueError('Network changed candidate membership')
                logits = exposure_logits(model, cache, config['eval_batch_size'])
                pointwise = exposure_metrics(cache['label'], logits)
                prediction = output / 'artifacts/predictions' / name / (stem + '.npz')
                prediction.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(prediction, logits=logits, labels=np.array(cache['label']), event_ids=np.array(cache['event_id']))
                row = {'ranking': ranking, 'pointwise': pointwise, 'rank_artifact': str(artifact.resolve()),
                       'prediction_artifact': str(prediction.resolve()), 'rank_sha256': digest_file(artifact),
                       'prediction_sha256': digest_file(prediction)}
                if name == 'seed42':
                    score_path = output / 'artifacts/scores' / (stem + '.npy')
                    score_path.parent.mkdir(parents=True, exist_ok=True); np.save(score_path, scores)
                    row['scores_artifact'] = str(score_path.resolve()); row['scores_sha256'] = digest_file(score_path)
                    with np.load(artifact) as new, np.load(old['results']['din_seed42'][part]['rank_artifact']) as baseline:
                        np.testing.assert_array_equal(new['event_ids'], baseline['event_ids'])
                        result['paired_user_bootstrap'][part] = paired_user_bootstrap(new['ranks'], baseline['ranks'], new['users'],
                            repeats=config['bootstrap_repeats'], seed=config['bootstrap_seed'])
                result['results'].setdefault(name, {})[part] = row
                write_json(partial, result)
                print(f"{name} {part} N20={ranking['ndcg@20']:.7f} R20={ranking['recall@20']:.7f}", flush=True)
            del model
        result['status'] = 'EVALUATED'
        write_json(output / 'evaluation.json', result)
        return result
    finally:
        close(caches, candidates)


def verify(override=None):
    deterministic(); config, output = get_config(override)
    selected = read(output / 'selection.json'); rep = read(output / 'replication.json')
    verify_evidence(ROOT, selected['evidence'], config); verify_evidence(ROOT, rep['evidence'], config)
    ev = read(output / 'evaluation.json')
    if ev['selection_sha256'] != digest_file(output / 'selection.json') or ev['replication_sha256'] != digest_file(output / 'replication.json'):
        raise ValueError('Evaluation freeze mismatch')
    _, _, _, _, caches, candidates, _ = previous_setup(tuple(ev['results']['seed42']))
    checks = {}
    try:
        model, _ = load_model(rep['runs']['seed42']['checkpoint'], config['device'])
        for part, row in ev['results']['seed42'].items():
            scores = candidate_scores(model, caches[part], candidates[part], config['candidate_batch_size'])
            np.testing.assert_array_equal(scores, np.load(row['scores_artifact']))
            if rank_scores(candidates[part], caches[part], scores) != row['ranking']:
                raise ValueError('Checkpoint rankings differ')
            class Changed:
                def __getitem__(self, key):
                    if key in {'label', 'item_index'}:
                        return np.zeros(len(caches[part]), dtype=caches[part][key].dtype)
                    return caches[part][key]
            class Subset(dict):
                def __len__(self): return len(self['indices'])
            n = min(config['candidate_batch_size'], len(candidates[part]))
            small = Subset({k: np.array(candidates[part][k][:n]) for k in ['ids','scores','indices']})
            np.testing.assert_array_equal(candidate_scores(model, Changed(), small, config['candidate_batch_size']), scores[:n])
            checks[part] = {'rows_recomputed': len(candidates[part]), 'scores_identical': True,
                            'label_target_mutation_invariant_rows': n}
        for name, rows in ev['results'].items():
            for part, row in rows.items():
                if digest_file(row['rank_artifact']) != row['rank_sha256'] or digest_file(row['prediction_artifact']) != row['prediction_sha256']:
                    raise ValueError('Saved evaluation artifact changed')
                validate_rank_artifact(row['rank_artifact'], candidates[part], row['ranking'])
                with np.load(row['prediction_artifact']) as pred:
                    np.testing.assert_array_equal(pred['event_ids'], caches[part]['event_id'])
                    np.testing.assert_array_equal(pred['labels'], caches[part]['label'])
                    if exposure_metrics(pred['labels'], pred['logits']) != row['pointwise']:
                        raise ValueError('Saved exposure metrics differ')
        result = {'status': 'PASS', 'primary_seed_full_score_recomputation': checks,
                  'all_seed_artifact_hashes_checked': True}
        write_json(output / 'verification.json', result); return result
    finally:
        close(caches, candidates)


def report(override=None):
    config, output = get_config(override)
    selected = read(output / 'selection.json'); ledger = read(output / 'tuning.json')
    verify_evidence(ROOT, selected['evidence'], config)
    lines = ['# 固定Top100的DIN网络结构优化', '',
             '本轮最终指标仍为Top20，保持原双塔及完全相同的Top100候选，不增加手工历史降权。', '',
             '网络修改：Dice alpha可训练；新增候选/用户相关的归一化长短历史注意力、向量门控、乘积/差值交互与LayerNorm分支。新增输出零初始化，初始逐值保持旧DIN。', '',
             '全部配置从同一个原DIN seed42检查点继续训练，包含相同训练预算的旧网络对照。长短历史是位置窗口，不是实际时间。', '',
             '## 完整标准验证集', '', '| 配置 | 最佳轮次 | Recall@20 | NDCG@20 |', '| --- | ---: | ---: | ---: |']
    baseline = selected['baseline_validation']
    lines.append(f"| 原DIN检查点 | — | {baseline['recall@20']:.6f} | {baseline['ndcg@20']:.6f} |")
    for name, trial in ledger['trials'].items():
        r = trial['best_validation']['ranking']
        lines.append(f"| {name} | {trial['best_epoch']} | {r['recall@20']:.6f} | {r['ndcg@20']:.6f} |")
    lines += ['', '选模状态：' + selected['status'] + '；结构：' + str(selected['variant']) + '。']
    ev_path = output / 'evaluation.json'
    if ev_path.exists():
        ev = read(ev_path); old = read(ROOT / 'experiments/ranking_v2/evaluation.json')
        for part in ['standard/test', 'random/test']:
            lines += ['', '## ' + part, '', '| 网络 | Recall@20 | NDCG@20 | Recall@100 | AUC |', '| --- | ---: | ---: | ---: | ---: |']
            b = old['results']['din_seed42'][part]
            r = b['ranking']
            lines.append(f"| 原DIN | {r['recall@20']:.6f} | {r['ndcg@20']:.6f} | {r['recall@100']:.6f} | {b['pointwise']['auc']:.6f} |")
            for name, rows in ev['results'].items():
                r = rows[part]['ranking']
                lines.append(f"| {name} | {r['recall@20']:.6f} | {r['ndcg@20']:.6f} | {r['recall@100']:.6f} | {rows[part]['pointwise']['auc']:.6f} |")
            ci = ev['paired_user_bootstrap'][part]['ndcg@20']
            lines += ['', f"主种子减原DIN的NDCG差：{ci['difference']:.7f}；配对用户簇95%区间[{ci['ci95_low']:.7f}, {ci['ci95_high']:.7f}]。"]
        lines += ['', '种子42/43/44均从相同的旧seed42权重开始，仅复核继续训练随机性，不能宣称三个独立从头训练。测试已在旧阶段查看，本次是冻结配置后的迭代复测。',
                  'Bootstrap反映测试用户抽样不确定性，不包含选模或训练随机性；区间跨0不主张稳定提升。']
    lines += ['', '## 复现', '', '```powershell', '. .\\scripts\\enter_env.ps1',
              'python -B -m src.ranking_v3.run --phase fit --output-directory experiments/ranking_v3_reproduction',
              'python -B -m src.ranking_v3.run --phase replicate --output-directory experiments/ranking_v3_reproduction',
              'python -B -m src.ranking_v3.run --phase evaluate --output-directory experiments/ranking_v3_reproduction',
              'python -B -m src.ranking_v3.run --phase verify --output-directory experiments/ranking_v3_reproduction', '```', '',
              '实际配置、曲线、检查点和产物保存在 ' + output.relative_to(ROOT).as_posix() + '。原冻结版本保留；新环境从头复现未验证。']
    text = '\n'.join(lines) + '\n'
    (ROOT / 'reports/ranking_v3_top20_report.md').write_text(text, encoding='utf-8')
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['fit','replicate','evaluate','verify','report'])
    p.add_argument('--output-directory')
    args = p.parse_args()
    value = globals()[args.phase](args.output_directory)
    if args.phase == 'report': print(value)


if __name__ == '__main__': main()
