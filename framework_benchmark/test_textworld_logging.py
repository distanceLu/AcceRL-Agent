import unittest
from types import SimpleNamespace
from framework_benchmark.textworld_logging import summarize


def sample(score, maximum, won=False, reason='lost', **kwargs):
    return SimpleNamespace(metadata={'evaluation_result': dict(score=score, max_score=maximum,
                                                              won=won, reason=reason)}, **kwargs)


class TextWorldLoggingTests(unittest.TestCase):
    def test_episode_mean_not_ratio_of_sums(self):
        result = summarize([sample(1, 2, True), sample(2, 8, reason='history_limit'),
                            sample(0, 4, reason='step_limit')])
        self.assertEqual(result['Episodes'], 3)
        self.assertAlmostEqual(result['WinRate'], 1/3)
        self.assertAlmostEqual(result['NormalizedScore'], .25)

    def test_abort_missing_and_evaluation_are_excluded(self):
        aborted = sample(4, 4, True, status=SimpleNamespace(name='ABORTED'))
        evaluation = sample(4, 4, True)
        evaluation.metadata['evaluation'] = True
        result = summarize([aborted, evaluation, SimpleNamespace(metadata={}), sample(0, 4)])
        self.assertEqual(result['Episodes'], 1)
        self.assertEqual(result['WinRate'], 0)

    def test_empty_and_zero_maximum_have_no_fabricated_score(self):
        self.assertEqual(summarize([]), {'Episodes': 0})
        result = summarize([sample(0, 0)])
        self.assertNotIn('NormalizedScore', result)
        self.assertEqual(result['NormalizedScoreEpisodes'], 0)


if __name__ == '__main__':
    unittest.main()
