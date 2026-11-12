"""TextWorld metrics for completed episodes in Slime's selected rollout batch."""
import logging
import math


def summarize(samples):
    results = []
    for sample in samples:
        if getattr(getattr(sample, 'status', None), 'name', None) == 'ABORTED':
            continue
        metadata = sample.metadata or {}
        if metadata.get('evaluation', False):
            continue
        result = metadata.get('evaluation_result')
        if result is not None:
            results.append(result)
    metrics = {'Episodes': len(results)}
    if not results:
        return metrics
    metrics['WinRate'] = sum(bool(r['won']) for r in results) / len(results)
    normalized = []
    for result in results:
        score, maximum = float(result['score']), float(result['max_score'])
        if not math.isfinite(score) or not math.isfinite(maximum) or maximum < 0:
            raise ValueError('Invalid TextWorld score/max_score')
        if maximum > 0:
            normalized.append(score / maximum)
    metrics['NormalizedScoreEpisodes'] = len(normalized)
    if normalized:
        metrics['NormalizedScore'] = sum(normalized) / len(normalized)
    return metrics


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_rollout_step
    metrics = {f'textworld/train/{key}': value for key, value in summarize(samples).items()}
    metrics['rollout/step'] = compute_rollout_step(args, rollout_id)
    logging.getLogger(__name__).info('TextWorld rollout %s: %s', rollout_id, metrics)
    logging_utils.log(args, metrics, step_key='rollout/step')
    # Preserve Slime's standard rollout/performance logging.
    return False
