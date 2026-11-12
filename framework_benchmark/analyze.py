"""Reduce raw events using explicit run/measurement boundaries."""
import argparse
import json
from pathlib import Path
from .metrics import Counters


def summarize(directory, run):
    start, stop = run['training_start_ns'], run['training_stop_ns']
    if not run['job_start_ns'] <= start < stop <= run['job_stop_ns']:
        raise ValueError('Invalid job / training boundaries')
    counters = Counters()
    metrics = []
    for path in sorted(Path(directory).glob('*.jsonl')):
        with path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if start <= event['wall_time_ns'] <= stop:
                    counters.ingest(event)
                    if 'metrics' in event:
                        metrics.append(event)
    report = counters.report((stop-start)/1e9, (run['job_stop_ns']-run['job_start_ns'])/1e9, len(run['gpus']))
    report['training_metrics'] = sorted(metrics, key=lambda x:x['wall_time_ns'])
    report['run'] = run
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.events, json.loads(args.run.read_text()))
    with args.output.open('x') as out:
        json.dump(report,out,indent=2)
