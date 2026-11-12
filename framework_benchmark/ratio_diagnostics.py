"""Opt-in, exact action-token PPO diagnostics; no change to loss or gradients.

CPU rows are gathered once per optimizer step, not once per microbatch.
Only TP=PP=CP=1 is supported (the benchmark's EP=3 / DP=3 layout).
"""
import math
import torch

_rows = []


def capture(args, batch, old, new, advantages):
    if (args.context_parallel_size, args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size) != (1, 1, 1):
        raise ValueError('Ratio diagnostics require TP=PP=CP=1')
    if args.advantage_estimator != 'ppo':
        raise ValueError('Ratio diagnostics describe token-level PPO only')
    with torch.no_grad():
        mask = torch.cat(batch['loss_masks']).bool().to(new.device)
        rollout = torch.cat(batch['rollout_log_probs']).to(new.device)
        assert old.shape == new.shape == advantages.shape == mask.shape == rollout.shape
        # Store local sample/response offsets to locate outliers without dumping observations.
        sample = torch.cat([torch.full((n,), i, device=new.device) for i, n in enumerate(batch['response_lengths'])])
        offset = torch.cat([torch.arange(n, device=new.device) for n in batch['response_lengths']])
        rows = torch.stack([old.double(), new.double(), advantages.double(), rollout.double(),
                            sample.double(), offset.double(), (new-old).double(),
                            (new-old).exp().double()], dim=1)[mask].detach().cpu()
        _rows.append(rows)


def summarize(rows):
    """Rows: old,new,advantage,rollout,sample,offset,log_ratio,ratio,rank,microbatch."""
    result = {'action_tokens': len(rows)}
    if not len(rows):
        return result, []
    old, new, adv, rollout = rows[:, :4].unbind(1)
    delta, ratio = rows[:, 6], rows[:, 7]
    for name, values in [('ratio', ratio), ('log_ratio', delta), ('old_logprob', old),
                         ('new_logprob', new), ('rollout_logprob', rollout), ('advantage', adv)]:
        finite = values[torch.isfinite(values)]
        result[name+'_nonfinite_count'] = len(values)-len(finite)
        if len(finite):
            # Nonfinite values are reported separately; never silently clipped.
            prefix = name + ('_finite' if len(finite) != len(values) else '')
            result[prefix+'_mean'] = finite.mean().item()
            result[prefix+'_min'] = finite.min().item()
            result[prefix+'_max'] = finite.max().item()
            for label, q in [('p50', .5), ('p90', .9), ('p99', .99), ('p999', .999)]:
                result[prefix+'_'+label] = torch.quantile(finite, q).item()
    result['ratio_gt_1_1_fraction'] = (ratio > 1.1).double().mean().item()
    result['ratio_gt_2_fraction'] = (ratio > 2).double().mean().item()
    result['ratio_gt_10_fraction'] = (ratio > 10).double().mean().item()
    result['negative_adv_ratio_gt_10_count'] = ((adv < 0) & (ratio > 10)).sum().item()
    top = []
    for idx in torch.argsort(torch.nan_to_num(delta, nan=-math.inf), descending=True)[:20].tolist():
        row = rows[idx].tolist()
        record = dict(zip(['old_logprob','new_logprob','advantage','rollout_logprob',
                           'sample_in_microbatch','response_token_offset','log_ratio','ratio','dp_rank','microbatch'], row))
        record.update(log_ratio=delta[idx].item(), ratio=ratio[idx].item())
        top.append({k: v if math.isfinite(v) else str(v) for k, v in record.items()})
    # Values all refer to the SAME max-log-ratio token, not separate extrema.
    for key in ['ratio','log_ratio','old_logprob','new_logprob','advantage','rollout_logprob']:
        value = top[0][key]
        if isinstance(value, (int, float)):
            result['at_max_ratio/'+key] = value
    return result, top


def finish_step(step):
    from megatron.core import mpu
    from .events import emit
    global _rows
    group = mpu.get_data_parallel_group(with_context_parallel=True)
    rank = torch.distributed.get_rank(group)
    local = []
    for microbatch, rows in enumerate(_rows):
        identity = torch.tensor([rank, microbatch], dtype=torch.float64).expand(len(rows), 2)
        local.append(torch.cat([rows, identity], dim=1))
    _rows = []
    local = torch.cat(local) if local else torch.empty((0, 10), dtype=torch.float64)
    gathered = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(gathered, local, group=group)
    if rank != 0:
        return {}
    metrics, top = summarize(torch.cat(gathered))
    emit({'kind':'ppo_ratio_diagnostics','episode_id':'metrics','step':step,
          'ratio_level':'token','top_tokens':top,'metrics':metrics})
    return {f'ratio_diag/{key}': value for key, value in metrics.items()}
