"""Private local Ray instance, restricted to the four benchmark devices."""
import os
import runpy
import ray

if __name__ == '__main__':
    ray.init(address='local', num_gpus=4, num_cpus=24, include_dashboard=False,
             _temp_dir=f'/tmp/slime-smoke-{os.getpid()}')
    try:
        runpy.run_path('/mnt/data/lcx4/slime/train.py', run_name='__main__')
    finally:
        ray.shutdown()
