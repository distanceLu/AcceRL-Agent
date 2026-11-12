"""Freeze experiment inputs without starting Ray, CUDA, or a training job."""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAMES = Path('/mnt/data/lcx4/AcceRL/accerl_vllm/data/tw_games/ftwp/games')
MODEL = Path('/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build_manifest(base=GAMES, limit=400, selection_seed=42):
    splits = {}
    seen = set()
    for split in ('train', 'valid', 'test'):
        files = sorted((Path(base) / split).glob('*.z8'))
        if split == 'train':
            random.Random(selection_seed).shuffle(files)
            if len(files) < limit:
                raise ValueError('Not enough training games')
            files = files[:limit]
        if not files:
            raise ValueError(f'Empty split: {split}')
        rows = [{'path': str(p.resolve()), 'sha256': digest(p)} for p in files]
        hashes = {r['sha256'] for r in rows}
        if hashes & seen:
            raise ValueError(f'Game content overlaps another split: {split}')
        seen.update(hashes)
        splits[split] = rows
    return {'selection_seed': selection_seed, 'splits': splits}


def verify_manifest(manifest):
    for rows in manifest['splits'].values():
        for row in rows:
            if digest(row['path']) != row['sha256']:
                raise ValueError(f"Game changed: {row['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=Path, default=GAMES)
    args = parser.parse_args()
    manifest = build_manifest(args.games)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'games.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for split, rows in manifest['splits'].items():
        with (args.output / f'{split}.jsonl').open('w') as out:
            for row in rows:
                out.write(json.dumps({'prompt': row['path'], 'metadata': row}) + '\n')
    protocol = {
        'schema_version': 1, 'gpus': [0, 1, 2, 3], 'seeds': [42, 43, 44],
        'arms': ['accerl_ppo', 'slime_ppo_sync', 'slime_ppo_fully_async'],
        'model': str(MODEL), 'model_config_sha256': digest(MODEL / 'config.json'),
        'algorithm': 'ppo', 'accerl_gae': 'jit', 'accerl_advantage_normalization': 'ema_zscore',
        'task': {'max_attempts': 40, 'history_tokens': 4096, 'action_tokens': 16,
                 'win_bonus': 1.0, 'lost_penalty': 0.5, 'invalid_penalty': 0.1},
        'evaluation': {'decoding': 'greedy', 'primary': 'macro_normalized_score',
                       'secondary': 'win_rate', 'targets': None, 'max_gpu_hours': None},
        'status': 'prepared_not_gpu_validated',
        'pending_gates': ['qwen2_moe_weight_conversion_and_sync', 'ppo_actor_and_critic_targets',
                          'independent_evaluation', 'pilot_targets_and_budget', 'gpu_release'],
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    sources = {}
    for relative in ('accerl_agent', 'framework_benchmark'):
        for path in sorted((ROOT / relative).rglob('*.py')):
            sources[str(path.relative_to(ROOT))] = digest(path)
    (args.output / 'source_hashes.json').write_text(json.dumps(sources, indent=2) + '\n')
    for name, root in [('accerl', ROOT), ('slime', Path('/mnt/data/lcx4/slime'))]:
        for filename, command in [('head', ['rev-parse', 'HEAD']), ('diff', ['diff', 'HEAD'])]:
            result = subprocess.run(['git', '-C', str(root), *command], check=True, capture_output=True)
            (args.output / f'{name}.{filename}').write_bytes(result.stdout)
    print(json.dumps({'output': str(args.output), 'games': {k: len(v) for k,v in manifest['splits'].items()}}))


if __name__ == '__main__':
    main()
