"""Private local Ray cluster; never joins or stops another job's cluster."""
import os
import runpy
import ray
from .runtime import SLIME

if __name__ == '__main__':
    ray.init(address='local', num_gpus=4, num_cpus=24, include_dashboard=False,
             _temp_dir=f'/tmp/slime-textworld-{os.getpid()}')
    try:
        runpy.run_path(str(SLIME / 'train.py'), run_name='__main__')
    finally:
        ray.shutdown()
