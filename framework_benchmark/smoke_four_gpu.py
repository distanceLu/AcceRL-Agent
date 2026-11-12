"""Four-GPU capacity smoke test; not a convergence/performance benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from .runtime import ROOT, ENV, environment, model_args
from .convert_checkpoint import require_free_gpus


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expert-parallel-size',type=int,choices=[1,3],default=1)
    args=p.parse_args()
    require_free_gpus()
    args.output.mkdir(parents=True,exist_ok=False)
    env=environment('0,1,2,3')
    env.update(TW_EVENT_DIR=str(args.output.resolve()/'events'), TW_CPU_WORKERS='3',
               RAY_DEDUP_LOGS='0', CUDA_DEVICE_MAX_CONNECTIONS='1')
    config=json.loads((ROOT/'framework_benchmark/config.json').read_text())
    command=[str(ENV/'bin/python'),'-m','framework_benchmark.slime_smoke_entry',
             *config['slime_common_args'],*model_args(),
             '--expert-model-parallel-size',str(args.expert_parallel_size),
             '--hf-checkpoint','/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat',
             '--ref-load',str(ROOT/'framework_benchmark/artifacts/conversion_v1/checkpoint'),
             '--prompt-data',str(ROOT/'framework_benchmark/prepared_v2/train.jsonl'),
             '--num-rollout','1','--rollout-batch-size','3','--global-batch-size','3',
             '--num-steps-per-rollout','1','--num-critic-only-steps','0',
             '--rollout-max-response-len','16','--rollout-max-context-len','4096',
             '--rollout-data-transport','object-store','--normalize-advantages',
             '--optimizer','adam','--lr-decay-style','constant',
             '--use-distributed-optimizer','--use-dynamic-batch-size','--max-tokens-per-gpu','4096',
             '--recompute-granularity','full','--recompute-method','uniform','--recompute-num-layers','1',
             '--sglang-mem-fraction-static','0.6','--sglang-max-running-requests','8',
             '--sglang-disable-cuda-graph','--sglang-enable-metrics',
             '--attention-backend','flash','--seed','42']
    (args.output/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    started=time.monotonic()
    peaks={str(i):0 for i in range(4)}
    print(f'Log: {args.output}/console.log',flush=True)
    with (args.output/'console.log').open('w') as log, (args.output/'gpu.jsonl').open('w') as gpu:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            rows=[]
            for line in output.splitlines():
                i,m,u=map(int,line.split(','))
                if i<4:
                    peaks[str(i)]=max(peaks[str(i)],m)
                    rows.append({'gpu':i,'memory_mib':m,'utilization':u})
            gpu.write(json.dumps({'elapsed':time.monotonic()-started,'gpus':rows})+'\n');gpu.flush()
            time.sleep(1)
    result={'exit_code':child.returncode,'seconds':time.monotonic()-started,'peak_memory_mib':peaks,
            'purpose':'capacity_only','forward_parity_passed':False}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    if child.returncode: print((args.output/'console.log').read_text()[-6000:])
    raise SystemExit(child.returncode)


if __name__ == '__main__': main()
