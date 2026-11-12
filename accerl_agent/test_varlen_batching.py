"""CPU tests for conserving fixed sample batches while changing pack counts."""

from __future__ import annotations

import copy
import random
import unittest

from accerl_agent.varlen_batching import pack_sequence_indices, split_packs_to_count


class VarlenBatchingTests(unittest.TestCase):
    def assert_valid_packs(self, packs, lengths, budget, max_sequences=0, expected=None):
        flattened = [index for pack in packs for index in pack]
        if expected is None:
            expected = range(len(lengths))
        self.assertEqual(sorted(flattened), sorted(expected))
        self.assertEqual(len(flattened), len(set(flattened)))
        for pack in packs:
            self.assertTrue(pack)
            self.assertLessEqual(sum(lengths[index] for index in pack), budget)
            if max_sequences:
                self.assertLessEqual(len(pack), max_sequences)

    def test_two_episodes_per_pack_yields_32_microbatches(self):
        lengths = [4096] * 64
        packs = pack_sequence_indices(lengths, 8192)
        self.assertEqual(len(packs), 32)
        self.assertTrue(all(len(pack) == 2 for pack in packs))
        self.assert_valid_packs(packs, lengths, 8192)

    def test_four_episodes_per_pack_yields_16_microbatches(self):
        lengths = [2048] * 64
        packs = pack_sequence_indices(lengths, 8192)
        self.assertEqual(len(packs), 16)
        self.assertTrue(all(len(pack) == 4 for pack in packs))
        self.assert_valid_packs(packs, lengths, 8192)

    def test_zero_cap_accepts_many_short_sequences(self):
        lengths = [32] * 64
        unlimited = pack_sequence_indices(lengths, 8192, max_sequences=0)
        capped = pack_sequence_indices(lengths, 8192, max_sequences=2)
        self.assertEqual(unlimited, [list(range(64))])
        self.assertEqual(len(capped), 32)
        self.assert_valid_packs(capped, lengths, 8192, max_sequences=2)

    def test_first_fit_decreasing_is_stable_for_equal_lengths(self):
        lengths = [3, 3, 2, 2]
        packs = pack_sequence_indices(lengths, 5)
        self.assertEqual(packs, [[0, 2], [1, 3]])
        self.assertEqual(lengths, [3, 3, 2, 2])

    def test_mixed_lengths_are_conserved_and_not_padded_to_budget(self):
        lengths = [500, 4000, 2000, 1500, 3500, 300, 1000, 4500]
        packs = pack_sequence_indices(lengths, 8192)
        self.assertEqual(packs, [[7, 4], [1, 2, 3, 0], [6, 5]])
        self.assert_valid_packs(packs, lengths, 8192)
        self.assertEqual(sum(sum(lengths[index] for index in pack) for pack in packs), sum(lengths))

    def test_three_rank_schedules_align_to_maximum_pack_count(self):
        rank_lengths = [[4096] * 64, [2048] * 64, [1024] * 64]
        local = [pack_sequence_indices(lengths, 8192) for lengths in rank_lengths]
        self.assertEqual([len(packs) for packs in local], [32, 16, 8])
        target = max(map(len, local))
        for lengths, packs in zip(rank_lengths, local):
            with self.subTest(length=lengths[0]):
                original = copy.deepcopy(packs)
                aligned = split_packs_to_count(packs, target, lengths)
                self.assertEqual(len(aligned), 32)
                self.assertEqual(packs, original)
                self.assert_valid_packs(aligned, lengths, 8192)
                owners = {index: owner for owner, pack in enumerate(packs) for index in pack}
                # Splitting only subdivides existing packs, preserving their budgets.
                self.assertTrue(all(len({owners[index] for index in pack}) == 1 for pack in aligned))

    def test_split_to_singletons_and_noop_return_independent_lists(self):
        lengths = [10, 9, 8, 7, 6]
        original = [[0, 1, 2], [3, 4]]
        for target in [2, 3, 4, 5]:
            with self.subTest(target=target):
                result = split_packs_to_count(original, target, lengths)
                self.assertEqual(len(result), target)
                self.assert_valid_packs(result, lengths, 27)
                self.assertIsNot(result, original)
                self.assertTrue(all(pack is not source for pack in result for source in original))
        self.assertEqual(original, [[0, 1, 2], [3, 4]])

    def test_split_preserves_only_the_samples_referenced_by_input_packs(self):
        result = split_packs_to_count([[1, 3]], 2, [1000, 10, 1000, 20])
        self.assertEqual(result, [[3], [1]])
        self.assert_valid_packs(result, [1000, 10, 1000, 20], 30, expected=[1, 3])

    def test_empty_inputs(self):
        self.assertEqual(pack_sequence_indices([], 8192), [])
        self.assertEqual(split_packs_to_count([], 0, []), [])
        with self.assertRaises(ValueError):
            split_packs_to_count([], 1, [])

    def test_invalid_pack_arguments(self):
        cases = [
            ([5], 4, 0),
            ([1], 0, 0),
            ([1], -1, 0),
            ([1], 1.5, 0),
            ([0], 8, 0),
            ([-1], 8, 0),
            ([1.5], 8, 0),
            ([True], 8, 0),
            ([1], 8, -1),
            ([1], 8, 1.5),
            ([1], 8, True),
        ]
        for lengths, budget, cap in cases:
            with self.subTest(lengths=lengths, budget=budget, cap=cap):
                with self.assertRaises(ValueError):
                    pack_sequence_indices(lengths, budget, cap)

    def test_invalid_split_arguments(self):
        cases = [
            ([[0, 1]], 0, [1, 1]),
            ([[0, 1]], 3, [1, 1]),
            ([[0], [1]], 1, [1, 1]),
            ([[0]], -1, [1]),
            ([[0]], 1.5, [1]),
            ([[0]], True, [1]),
            ([[]], 1, [1]),
            ([[0, 0]], 2, [1]),
            ([[0], [0]], 2, [1]),
            ([[-1]], 1, [1]),
            ([[1]], 1, [1]),
            ([[0.5]], 1, [1]),
            ([[True]], 1, [1, 1]),
            ([[0]], 1, [0]),
        ]
        for packs, target, lengths in cases:
            with self.subTest(packs=packs, target=target, lengths=lengths):
                with self.assertRaises(ValueError):
                    split_packs_to_count(packs, target, lengths)

    def test_randomized_conservation_budget_and_sequence_caps(self):
        rng = random.Random(42)
        for case in range(100):
            lengths = [rng.randint(1, 64) for _ in range(rng.randint(1, 64))]
            cap = rng.choice([0, 1, 2, 8])
            with self.subTest(case=case, cap=cap):
                packs = pack_sequence_indices(lengths, 64, cap)
                self.assert_valid_packs(packs, lengths, 64, cap)
                target = rng.randint(len(packs), len(lengths))
                aligned = split_packs_to_count(packs, target, lengths)
                self.assertEqual(len(aligned), target)
                self.assert_valid_packs(aligned, lengths, 64, cap)


if __name__ == "__main__":
    unittest.main()
