import json
from pathlib import Path
import tempfile
import unittest

from framework_benchmark.prepare import build_manifest, verify_manifest
from framework_benchmark.metrics import Counters, time_to_target
from framework_benchmark.slime_adapter import action_gae


class ProtocolTests(unittest.TestCase):
    def test_accerl_manifest_does_not_change_with_training_seed(self):
        from argparse import Namespace
        from accerl_agent.agent_textworld import load_textworld_game_files
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for split in ('train', 'valid', 'test'):
                (base/split).mkdir()
                (base/split/'one.z8').write_text(split)
            manifest = build_manifest(base, limit=1)
            path = base/'games.json'
            path.write_text(json.dumps(manifest))
            a = load_textworld_game_files(Namespace(tw_game_manifest=str(path), seed=42))
            b = load_textworld_game_files(Namespace(tw_game_manifest=str(path), seed=43))
            self.assertEqual(a, b)
            self.assertEqual(a, [str(base/'train/one.z8')])

    def test_game_selection_is_frozen_and_tamper_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for split in ('train', 'valid', 'test'):
                (base/split).mkdir()
                for i in range(5):
                    (base/split/f'{i}.z8').write_text(f'{split}-{i}')
            manifest = build_manifest(base, limit=3)
            self.assertEqual(manifest, build_manifest(base, limit=3))
            verify_manifest(manifest)
            Path(manifest['splits']['train'][0]['path']).write_text('changed')
            with self.assertRaises(ValueError): verify_manifest(manifest)

    def test_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for split in ('train', 'valid', 'test'):
                (base/split).mkdir()
                (base/split/'same.z8').write_text('same content')
            with self.assertRaises(ValueError): build_manifest(base, limit=1)

    def test_replay_and_retries_do_not_inflate_generation(self):
        counters = Counters()
        episode = {'kind': 'episode_end', 'episode_id': 'a', 'reason': 'step_limit',
                   'retained_actions': [{'request_id': 'r', 'tokens': 3}]}
        counters.ingest(episode)
        counters.ingest(episode)
        counters.ingest({**episode, 'episode_id': 'eval', 'evaluation': True})
        counters.ingest({**episode, 'episode_id': 'aborted', 'reason': 'shutdown'})
        counters.ingest({'kind': 'optimizer_commit', 'episode_id': 'a', 'response_token_uses': 9})
        for _ in range(2):
            counters.ingest({'kind': 'env_step', 'episode_id': 'a', 'attempt': 0})
        report = counters.report(10, 20)
        self.assertEqual(report['new_effective_action_tokens_per_second'], .3)
        self.assertEqual(report['training_response_token_uses_per_second'], .9)
        self.assertEqual(report['complete_episodes_per_second'], .1)
        self.assertEqual(report['environment_steps_per_second'], .1)

    def test_action_gae_skips_observations_and_keeps_step_rewards(self):
        a, r = action_gae([.5, 999, .25], [1, 0, 2], [1, 0, 1], 1, 1)
        self.assertEqual(a, [2.5, 0, 1.75])
        self.assertEqual(r, [3, 0, 2])
        _, discounted = action_gae([0, 999, 0], [1, 0, 2], [1, 0, 1], .5, 1)
        self.assertEqual(discounted, [2, 0, 2])

    def test_targets_are_measured_not_interpolated(self):
        points = [{'wall_seconds': 10, 'gpu_hours': .1, 'macro_normalized_score': .1},
                  {'wall_seconds': 20, 'gpu_hours': .2, 'macro_normalized_score': .5}]
        self.assertEqual(time_to_target(points, .3)['wall_seconds'], 20)
        self.assertFalse(time_to_target(points, .9)['reached'])


if __name__ == '__main__': unittest.main()
