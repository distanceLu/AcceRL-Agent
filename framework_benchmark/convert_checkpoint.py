"""Convert the original checkpoint on GPU 0; refuse occupied benchmark GPUs."""
import argparse
import json
from pathlib import Path
import socket
import subprocess

from .runtime import ROOT, SLIME, ENV, environment, model_args


def require_free_gpus():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    memory = {int(row.split(',')[0]): int(row.split(',')[1]) for row in output.splitlines()}
    if any(memory.get(i, 999999) > 1024 for i in range(4)):
        raise RuntimeError(f'Benchmark GPUs are occupied: {memory}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-forward', action='store_true')
    args = parser.parse_args()
    require_free_gpus()
    args.output.mkdir(parents=True, exist_ok=False)
    env = environment('0')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
    entry = ['-m', 'framework_benchmark.validate_conversion'] if args.validate_forward else [str(SLIME/'tools/convert_hf_to_torch_dist.py')]
    command = [str(ENV/'bin/python'), *entry,
               '--hf-checkpoint', '/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat',
               '--save', str(args.output.resolve()/'checkpoint'), '--bf16',
               '--seq-length', '4096', '--max-position-embeddings', '32768',
               '--tensor-model-parallel-size', '1', '--pipeline-model-parallel-size', '1',
               '--expert-model-parallel-size', '1', '--expert-tensor-parallel-size', '1',
               '--attention-backend', 'flash', *model_args()]
    (args.output/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    print(f'Conversion log: {args.output}/console.log', flush=True)
    with (args.output/'console.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (args.output/'result.json').write_text(json.dumps({'exit_code':result.returncode})+'\n')
    if result.returncode:
        print((args.output/'console.log').read_text()[-8000:])
    raise SystemExit(result.returncode)


if __name__ == '__main__': main()
