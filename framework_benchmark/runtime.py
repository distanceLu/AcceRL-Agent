"""Launch only the explicitly selected Slime checkout and its existing environment."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SLIME = Path('/mnt/data/lcx4/slime')
ENV = Path('/mnt/data/lcx4/slime_runtime/micromamba/envs/slime')


def model_args():
    output = subprocess.check_output(['bash', '-c', 'source "$1"; printf "%s\\0" "${MODEL_ARGS[@]}"',
                                      'model-args', str(ROOT/'framework_benchmark/model_qwen15_moe.sh')])
    return output.decode().rstrip('\0').split('\0')


def environment(gpus=''):
    # pip's cuDNN and the conda prefix contain different versions. Resolve all
    # CUDA wheel libraries first, so TE cannot mix their cuDNN components.
    libraries = sorted((ENV/'lib/python3.12/site-packages/nvidia').glob('*/lib'))
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': gpus,
            'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
            'PYTHONPATH': f'{ROOT}:{SLIME}:/mnt/data/lcx4/Megatron-LM',
            'PATH': f'{ENV}/bin:' + os.environ.get('PATH', ''),
            'LD_LIBRARY_PATH': ':'.join(map(str, libraries + [ENV/'lib'])),
            'LD_PRELOAD': str(ENV/'lib/libstdc++.so.6'),
            # FlashInfer appends .cache/flashinfer to this base. The shared
            # home cache may be a dangling symlink after cache cleanup.
            'FLASHINFER_WORKSPACE_BASE': os.environ.get('FLASHINFER_WORKSPACE_BASE', str(ROOT)),
            'CUDA_HOME': str(ENV), 'RAY_ADDRESS': 'local'}


if __name__ == '__main__':
    result = subprocess.run([str(ENV/'bin/python'), *sys.argv[1:]], env=environment(), cwd=ROOT)
    raise SystemExit(result.returncode)
