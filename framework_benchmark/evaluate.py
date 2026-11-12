"""Common isolated evaluator for either framework's EXPORTED frozen HF policy.

Run using Slime's interpreter with CUDA_VISIBLE_DEVICES restricted to the allotted
evaluation GPU. Never points at a concurrently updated training server.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

from .prepare import digest, verify_manifest
from .slime_adapter import generate, close_workers


async def evaluate(args, engine, rows):
    from slime.utils.types import Sample
    async def backend(payload):
        return await asyncio.to_thread(engine.generate, **payload)
    config = SimpleNamespace(hf_checkpoint=str(args.checkpoint), seed=42, _generate=backend)
    results = []
    for row in rows:
        sample = Sample(metadata={**row, 'evaluation':True})
        sample = await generate(config, sample, {'temperature':0., 'top_p':1., 'top_k':-1})
        if sample.status == Sample.Status.ABORTED:
            raise RuntimeError('Independent evaluation was interrupted')
        result = sample.metadata['evaluation_result']
        if result['max_score'] <= 0:
            raise ValueError(f'Undefined normalized score for {result["game"]}')
        results.append(result)
    return results


async def evaluate_and_close(args, engine, rows):
    try:
        return await evaluate(args, engine, rows)
    finally:
        await close_workers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['valid','test'], default='valid')
    parser.add_argument('--game-limit', type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    verify_manifest(manifest)
    rows = manifest['splits'][args.split]
    if args.game_limit is not None:
        if args.game_limit < 1: raise ValueError('game-limit must be positive')
        rows = rows[:args.game_limit]
    weights = sorted(args.checkpoint.glob('*.safetensors'))
    if not weights: raise ValueError('Export a frozen HuggingFace safetensors checkpoint first')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    hashes = {p.name:digest(p) for p in weights}
    file_states = {p:(p.stat().st_size,p.stat().st_mtime_ns) for p in weights}
    os.environ['TW_EVENT_DIR'] = str(args.output/'events')
    os.environ['TW_CPU_WORKERS'] = '1'
    import sglang
    engine = sglang.Engine(model_path=str(args.checkpoint), tp_size=1, mem_fraction_static=.8,
                           context_length=8192, random_seed=42)
    try:
        results = asyncio.run(evaluate_and_close(args, engine, rows))
    finally:
        engine.shutdown()
    if any((p.stat().st_size,p.stat().st_mtime_ns)!=state for p,state in file_states.items()):
        raise RuntimeError('Checkpoint changed during evaluation')
    report = {'split':args.split, 'game_count':len(results), 'checkpoint_sha256':hashes,
              'manifest_sha256':digest(args.manifest), 'decoding':'greedy',
              'macro_normalized_score':sum(r['score']/r['max_score'] for r in results)/len(results),
              'win_rate':sum(r['won'] for r in results)/len(results),
              'evaluation_seconds':time.monotonic()-start, 'games':results}
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('games','checkpoint_sha256')}))


if __name__ == '__main__': main()
