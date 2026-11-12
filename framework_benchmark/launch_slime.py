"""Apply the verified Slime runtime environment, then start a private Ray job."""
import os
import sys
from .runtime import ENV, environment
from .convert_checkpoint import require_free_gpus

if __name__ == '__main__':
    require_free_gpus()
    executable = str(ENV / 'bin/python')
    os.execve(executable, [executable, '-m', 'framework_benchmark.slime_train_entry', *sys.argv[1:]],
              environment('0,1,2,3'))
