"""Framework-independent counters. Replay consumption is never generation."""
from dataclasses import dataclass, field
import math

COMPLETED = {'won', 'lost', 'step_limit', 'history_limit', 'environment_done_without_terminal_signal'}


@dataclass
class Counters:
    episodes: set = field(default_factory=set)
    steps: set = field(default_factory=set)
    tokens: dict = field(default_factory=dict)
    consumed_token_uses: int = 0
    optimizer_commits: int = 0
    ends: dict = field(default_factory=dict)

    def ingest(self, event):
        if event.get('evaluation', False):
            return
        episode = event['episode_id']
        kind = event['kind']
        if kind == 'env_step':
            self.steps.add((episode, event['attempt']))
        elif kind == 'episode_end':
            if event['reason'] not in COMPLETED:
                return
            if episode in self.episodes:
                return
            self.episodes.add(episode)
            self.ends[event['reason']] = self.ends.get(event['reason'], 0) + 1
            for request in event['retained_actions']:
                self.tokens[(episode, request['request_id'])] = request['tokens']
        elif kind == 'optimizer_commit':
            self.consumed_token_uses += event['response_token_uses']
            self.optimizer_commits += 1

    def report(self, training_seconds, total_seconds, gpu_count=4):
        if not (math.isfinite(training_seconds) and 0 < training_seconds <= total_seconds and gpu_count > 0):
            raise ValueError('Invalid time or resource denominator')
        return {'complete_episodes_per_second': len(self.episodes) / training_seconds,
                'environment_steps_per_second': len(self.steps) / training_seconds,
                'new_effective_action_tokens_per_second': sum(self.tokens.values()) / training_seconds,
                'training_response_token_uses_per_second': self.consumed_token_uses / training_seconds if self.optimizer_commits else None,
                'gpu_hours': total_seconds * gpu_count / 3600, 'termination_counts': self.ends}


def time_to_target(points, target, metric='macro_normalized_score'):
    """Use first measured crossing, never interpolate or backdate an evaluation."""
    for point in sorted(points, key=lambda p: p['wall_seconds']):
        if point[metric] >= target:
            return {'reached': True, 'wall_seconds': point['wall_seconds'], 'gpu_hours': point['gpu_hours']}
    return {'reached': False, 'wall_seconds': None, 'gpu_hours': None}
