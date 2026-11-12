"""Multi-turn SGLang generator and action-token PPO targets.

The metadata transport patch described in README is required before training.
The generator can be exercised independently of the Megatron model conversion.
"""
import asyncio
import atexit
import json
import os
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/mnt/data/lcx4/miniforge3/envs/accerl-agent/bin/python'
SYSTEM = ('You are a TextWorld action selector. Reply with exactly one command '
          'from the admissible commands. Do not add explanations, punctuation, '
          'quotes, or a leading prompt marker.')
_pool = None
_processes = []
_tokenizer = None


def _cleanup():
    for process in _processes:
        if process.returncode is None:
            try: process.terminate()
            except ProcessLookupError: pass


atexit.register(_cleanup)


async def close_workers():
    global _pool
    _cleanup()
    await asyncio.gather(*(p.wait() for p in _processes), return_exceptions=True)
    _processes.clear()
    _pool = None


async def rpc(process, message):
    process.stdin.write((json.dumps(message) + '\n').encode())
    await process.stdin.drain()
    line = await asyncio.wait_for(process.stdout.readline(), timeout=180)
    if not line:
        raise RuntimeError('TextWorld worker exited')
    result = json.loads(line)
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result


async def acquire():
    global _pool
    if _pool is None:
        workers = int(os.environ.get('TW_CPU_WORKERS', '24'))
        if workers < 1:
            raise ValueError('TW_CPU_WORKERS must be positive')
        _pool = asyncio.Queue()
        for _ in range(workers):
            _pool.put_nowait(None)
    process = await _pool.get()
    if process is None or process.returncode is not None:
        try:
            worker_env = dict(os.environ)
            # The CPU worker uses a different torch/CUDA build. Never leak the
            # Slime environment's library search paths into that interpreter.
            for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUDA_HOME'):
                worker_env.pop(key, None)
            worker_env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT))
            process = await asyncio.create_subprocess_exec(
                PYTHON, '-m', 'framework_benchmark.env_worker', cwd=str(ROOT),
                env=worker_env,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
            _processes.append(process)
        except BaseException:
            _pool.put_nowait(None)
            raise
    return process


def record(event):
    if not os.environ.get('TW_EVENT_DIR'):
        raise ValueError('TW_EVENT_DIR must be set for benchmark instrumentation')
    from .events import emit
    emit(event)


async def generate(args, sample, sampling_params):
    from slime.utils.http_utils import post
    from slime.utils.types import Sample
    from transformers import AutoTokenizer
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint)
        template = _tokenizer.chat_template or ''
        if '<|im_start|>' not in template or '<|im_end|>' not in template:
            raise ValueError('Qwen chat template required')
    tokenizer = _tokenizer
    process = await acquire()
    episode = uuid.uuid4().hex
    retained = []
    mask, rewards, logprobs = [], [], []
    evaluation = bool(sample.metadata.get('evaluation', False))
    try:
        state = await rpc(process, {'op': 'reset', 'game': sample.metadata['path'], 'seed': args.seed})
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': state['content']}],
            tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(prompt)
        prompt_len = len(tokens)
        reason = None
        for attempt in range(40):
            if len(tokens) >= 4096:
                reason = 'history_limit'
                break
            params = {**sampling_params, 'max_new_tokens': min(16, 4096-len(tokens)), 'stop': ['\n']}
            payload = {'input_ids': tokens, 'sampling_params': params, 'return_logprob': True}
            if hasattr(args, '_generate'):
                output = await args._generate(payload)
            else:
                output = await post(f'http://{args.sglang_router_ip}:{args.sglang_router_port}/generate', payload)
            meta = output['meta_info']
            if meta['finish_reason']['type'] == 'abort':
                sample.status = Sample.Status.ABORTED
                return sample
            triples = meta.get('output_token_logprobs', [])
            if not triples:
                raise ValueError('Missing action token IDs / behavior logprobs')
            ids = [int(t[1]) for t in triples]
            probs = [float(t[0]) for t in triples]
            state = await rpc(process, {'op': 'step', 'text': tokenizer.decode(ids, skip_special_tokens=True)})
            tokens.extend(ids)
            mask.extend([1] * len(ids))
            logprobs.extend(probs)
            rewards.extend([0.0] * (len(ids)-1) + [state['reward']])
            retained.append({'request_id': str(attempt), 'tokens': len(ids)})
            if state['env_step']:
                record({'kind': 'env_step', 'episode_id': episode, 'attempt': attempt, 'evaluation': evaluation})
            reason = state['reason']
            if reason in {'won', 'lost'}:
                break
            suffix = tokenizer.encode('<|im_end|>\n<|im_start|>user\n' + state['content'] +
                                      '<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False)
            tokens.extend(suffix)
            mask.extend([0]*len(suffix))
            logprobs.extend([0.0]*len(suffix))
            rewards.extend([0.0]*len(suffix))
            if reason:
                break
        reason = reason or 'step_limit'
        # AcceRL ends the training transcript at the final action, excluding trailing observation.
        while mask and mask[-1] == 0:
            mask.pop(); rewards.pop(); logprobs.pop(); tokens.pop()
        sample.tokens = tokens
        sample.response_length = len(tokens)-prompt_len
        sample.response = tokenizer.decode(tokens[prompt_len:])
        sample.loss_mask = mask
        sample.rollout_log_probs = logprobs
        sample.reward = sum(rewards)
        sample.train_metadata = {'token_rewards': rewards, 'termination_reason': reason}
        sample.metadata['evaluation_result'] = {'game': sample.metadata['path'], 'score':state['score'],
                                               'max_score':state['max_score'], 'won':state['won'], 'reason':reason}
        sample.status = Sample.Status.COMPLETED
        sample.remove_sample = not any(mask) or len(tokens) > 4096
        record({'kind': 'episode_end', 'episode_id': episode, 'reason': reason,
                'evaluation': evaluation,
                'score': state['score'], 'max_score': state['max_score'], 'won': state['won'],
                'retained_actions': [] if sample.remove_sample else retained})
        return sample
    finally:
        # A cancelled/failed RPC can leave an unread reply in stdout. A fresh
        # process is required before the next episode in that case.
        import sys
        if sys.exc_info()[0] is not None and process.returncode is None:
            process.terminate()
            await process.wait()
            _pool.put_nowait(None)
        else:
            _pool.put_nowait(process)


def action_gae(values, rewards, mask, gamma, lam, bootstrap=0.0):
    """GAE clock advances on generated action tokens, skipping observations."""
    if not (len(values) == len(rewards) == len(mask)):
        raise ValueError('Misaligned values, rewards and masks')
    indices = [i for i, keep in enumerate(mask) if keep]
    advantages = [0.0]*len(mask)
    returns = [0.0]*len(mask)
    next_value, acc = bootstrap, 0.0
    for i in reversed(indices):
        acc = rewards[i] + gamma*next_value - values[i] + gamma*lam*acc
        advantages[i] = acc
        returns[i] = acc + values[i]
        next_value = values[i]
    return advantages, returns


def compute_advantages(args, data):
    """Hook must be installed for BOTH actor and critic; requires CP=1."""
    import torch
    if args.context_parallel_size != 1 or args.kl_coef != 0:
        raise ValueError('TextWorld hook requires CP=1 and no reference reward KL')
    if 'metadata' not in data:
        raise ValueError('Slime metadata transport patch is required for per-action PPO rewards')
    from slime.utils.ppo_utils import chunked_gae
    compact_values, compact_rewards, indices = [], [], []
    for value, mask, meta in zip(data['values'], data['loss_masks'], data['metadata'], strict=True):
        if meta['termination_reason'] == 'environment_done_without_terminal_signal':
            raise ValueError('Nonterminal boundary requires an explicit bootstrap value; refusing terminal approximation')
        mask = torch.as_tensor(mask, device=value.device, dtype=torch.bool)
        reward = torch.as_tensor(meta['token_rewards'], device=value.device, dtype=torch.float32)
        if value.numel() != reward.numel() or value.numel() != mask.numel():
            raise ValueError('Misaligned PPO metadata after DP dispatch')
        index = mask.nonzero(as_tuple=True)[0]
        indices.append(index)
        compact_values.append(value.detach().float()[index])
        compact_rewards.append(reward[index])
    maximum = max((x.numel() for x in compact_values), default=0)
    advantages = [torch.zeros_like(v, dtype=torch.float32) for v in data['values']]
    returns = [torch.zeros_like(v, dtype=torch.float32) for v in data['values']]
    if maximum:
        v = torch.nn.utils.rnn.pad_sequence(compact_values, batch_first=True)
        r = torch.nn.utils.rnn.pad_sequence(compact_rewards, batch_first=True)
        a, ret = chunked_gae(r, v, args.gamma, args.lambd)
        for i, index in enumerate(indices):
            advantages[i][index] = a[i,:index.numel()]
            returns[i][index] = ret[i,:index.numel()]
    data['advantages'], data['returns'] = advantages, returns
