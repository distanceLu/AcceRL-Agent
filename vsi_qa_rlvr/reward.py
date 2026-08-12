# SPDX-License-Identifier: Apache-2.0
"""Score ScanNet incremental-counting responses for AcceRL GRPO."""

"""vsiqa"""
import json


class IncrementalCountingReward:
    """Parse one generated trajectory and return a scalar task reward."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.step_ids = tokenizer.encode("<STEP>", add_special_tokens=False)
        self.answer_ids = tokenizer.encode("<ANS>", add_special_tokens=False)
        self.semicolon_id = tokenizer.encode(";", add_special_tokens=False)[0]
        self.plus_id = tokenizer.encode("+", add_special_tokens=False)[0]
        self.equal_id = tokenizer.encode("=", add_special_tokens=False)[0]
        self.special_ids = set(tokenizer.all_special_ids)

    @staticmethod
    def _find_subsequences(token_ids, pattern):
        return [
            index
            for index in range(len(token_ids) - len(pattern) + 1)
            if token_ids[index : index + len(pattern)] == pattern
        ]

    def _decode_integer(self, token_ids):
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )
        if not text.isdigit():
            raise ValueError(f"invalid integer token text={text!r}")
        return int(text)

    def _parse_response(self, response_ids, expected_step_count):
        content_length = len(response_ids)
        while (
            content_length > 0
            and response_ids[content_length - 1] in self.special_ids
        ):
            content_length -= 1
        content_ids = response_ids[:content_length]
        answer_positions = self._find_subsequences(
            content_ids,
            self.answer_ids,
        )
        if len(answer_positions) != 1:
            raise ValueError("response must contain exactly one <ANS>")

        answer_start = answer_positions[0]
        body_ids = content_ids[:answer_start]
        final_start = answer_start + len(self.answer_ids)
        if final_start >= len(content_ids):
            raise ValueError("final answer is missing")

        step_positions = self._find_subsequences(body_ids, self.step_ids)
        if len(step_positions) != expected_step_count - 1:
            raise ValueError(
                "step delimiter count does not match input image count"
            )

        segment_ranges = []
        segment_start = 0
        for step_position in step_positions:
            segment_ranges.append((segment_start, step_position))
            segment_start = step_position + len(self.step_ids)
        segment_ranges.append((segment_start, len(body_ids)))

        steps = []
        for start, end in segment_ranges:
            segment = body_ids[start:end]
            semicolons = [
                index
                for index, token_id in enumerate(segment)
                if token_id == self.semicolon_id
            ]
            pluses = [
                index
                for index, token_id in enumerate(segment)
                if token_id == self.plus_id
            ]
            equals = [
                index
                for index, token_id in enumerate(segment)
                if token_id == self.equal_id
            ]
            if len(semicolons) != 1 or len(pluses) != 1 or len(equals) != 1:
                raise ValueError(
                    "each step must contain one semicolon, plus, and equal sign"
                )
            semicolon = semicolons[0]
            plus = pluses[0]
            equal = equals[0]
            if not 0 < semicolon < plus < equal < len(segment) - 1:
                raise ValueError("step field order is invalid")
            steps.append(
                (
                    self._decode_integer(segment[:semicolon]),
                    self._decode_integer(segment[semicolon + 1 : plus]),
                    self._decode_integer(segment[plus + 1 : equal]),
                    self._decode_integer(segment[equal + 1 :]),
                )
            )

        return steps, self._decode_integer(content_ids[final_start:])

    def score(self, response_ids, ground_truth_string):
        ground_truth = json.loads(ground_truth_string)
        expected_step_count = len(ground_truth["visible_counts"])
        try:
            steps, final_answer = self._parse_response(
                response_ids,
                expected_step_count,
            )
        except ValueError:
            return self.empty_score()

        visible_rewards = []
        memory_rewards = []
        association_rewards = []
        state_rewards = []
        for step_index, predicted in enumerate(steps):
            target_visible = int(ground_truth["visible_counts"][step_index])
            target_duplicate = int(
                ground_truth["duplicate_counts"][step_index]
            )
            target_new = int(ground_truth["new_counts"][step_index])
            target_current = int(
                ground_truth["cumulative_counts"][step_index]
            )
            target_previous = (
                0
                if step_index == 0
                else int(ground_truth["cumulative_counts"][step_index - 1])
            )
            (
                predicted_visible,
                predicted_previous,
                predicted_new,
                predicted_current,
            ) = predicted
            visible_rewards.append(float(predicted_visible == target_visible))
            memory_rewards.append(float(predicted_previous == target_previous))
            association_rewards.append(
                float(
                    predicted_visible == target_visible
                    and predicted_new == target_new
                    and predicted_visible - predicted_new == target_duplicate
                )
            )
            state_rewards.append(
                float(
                    predicted_previous == target_previous
                    and predicted_new == target_new
                    and predicted_current == target_current
                    and predicted_previous + predicted_new == predicted_current
                )
            )

        answer_reward = float(
            final_answer == int(ground_truth["final_answer"])
            and final_answer == steps[-1][3]
        )
        visible_reward = sum(visible_rewards) / expected_step_count
        memory_reward = sum(memory_rewards) / expected_step_count
        association_reward = sum(association_rewards) / expected_step_count
        state_reward = sum(state_rewards) / expected_step_count
        total_reward = (
            visible_reward
            + memory_reward
            + association_reward
            + state_reward
            + 1.0
            + answer_reward
        ) / 6.0
        return {
            "score": total_reward,
            "r_format": 1.0,
            "answer_exact_reward": answer_reward,
        }

    @staticmethod
    def empty_score():
        return {
            "score": 0.0,
            "r_format": 0.0,
            "answer_exact_reward": 0.0,
        }


__all__ = ["IncrementalCountingReward"]
"""vsiqa"""
