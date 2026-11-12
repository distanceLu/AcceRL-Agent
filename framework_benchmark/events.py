"""Identical append-only event writer for both frameworks (opt-in)."""
import json
import os
from pathlib import Path
import time


def emit(event):
    directory = os.environ.get('TW_EVENT_DIR')
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path/f'events-{os.getpid()}.jsonl').open('a') as out:
        out.write(json.dumps({'wall_time_ns':time.time_ns(), **event}, allow_nan=False)+'\n')


def accerl_episode(state, sample):
    retained = []
    if sample is not None:
        for index, record in enumerate(state.step_records):
            result = record.training_result
            if result.stop_reason != 'abort' and result.output_tokens:
                retained.append({'request_id':str(index), 'tokens':len(result.output_tokens)})
        if sum(r['tokens'] for r in retained) != sum(end-start for start,end in sample.response_spans):
            raise ValueError('Benchmark token count disagrees with response spans')
    emit({'kind':'episode_end', 'episode_id':state.telemetry_id, 'reason':state.termination_reason,
          'score':state.latest_score, 'max_score':float(state.infos.get('max_score',0)),
          'won':bool(state.won), 'retained_actions':retained})
