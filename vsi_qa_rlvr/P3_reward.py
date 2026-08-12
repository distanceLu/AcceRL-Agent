# SPDX-License-Identifier: Apache-2.0
"""Score the three-frame complete-capability P3 training objective."""

"""vsiqa"""
import json
import re


P3_REWARD_WEIGHTS = {
    "V_t": 1.0,
    "C_prev": 1.0,
    "D_t_N_t": 1.0,
    "C_t": 1.0,
    "format": 0.25,
    "answer": 0.5,
}


class P3Reward:
    """Apply the P3 full-trajectory reward to one generated response."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = set(tokenizer.all_special_ids)

    @staticmethod
    def _empty_scores(n_steps):
        return {
            "format": 0.0,
            "arithmetic": 0.0,
            "chain": 0.0,
            "eos": 0.0,
            "protocol": 0.0,
            "V_t": [0.0] * n_steps,
            "D_t": [0.0] * n_steps,
            "N_t": [0.0] * n_steps,
            "C_prev": [0.0] * n_steps,
            "C_t": [0.0] * n_steps,
            "step_t": [0.0] * n_steps,
            "answer": 0.0,
            "complete": 0.0,
        }

    @classmethod
    def _score_rollout(cls, output, ground_truth, eos):
        n_steps = len(ground_truth["visible_counts"])
        scores = cls._empty_scores(n_steps)
        scores["eos"] = float(eos)
        if output.count("<ANS>") != 1:
            return scores

        body, answer_text = output.split("<ANS>")
        step_texts = body.split("<STEP>")
        if len(step_texts) != n_steps or not answer_text.isdigit():
            return scores

        steps = []
        for step_text in step_texts:
            match = re.fullmatch(r"(\d+);(\d+)\+(\d+)=(\d+)", step_text)
            if match is None:
                return scores
            steps.append(tuple(int(value) for value in match.groups()))
        scores["format"] = 1.0

        ground_truth_previous = [
            0,
            *[
                int(value)
                for value in ground_truth["cumulative_counts"][:-1]
            ],
        ]
        for step_index, step in enumerate(steps):
            predicted_visible, predicted_previous, predicted_new, predicted_current = step
            predicted_duplicate = predicted_visible - predicted_new
            target_visible = int(ground_truth["visible_counts"][step_index])
            target_duplicate = int(
                ground_truth["duplicate_counts"][step_index]
            )
            target_new = int(ground_truth["new_counts"][step_index])
            target_current = int(
                ground_truth["cumulative_counts"][step_index]
            )
            scores["V_t"][step_index] = float(
                predicted_visible == target_visible
            )
            scores["D_t"][step_index] = float(
                predicted_duplicate == target_duplicate
            )
            scores["N_t"][step_index] = float(
                predicted_new == target_new
            )
            scores["C_prev"][step_index] = float(
                predicted_previous == ground_truth_previous[step_index]
            )
            scores["C_t"][step_index] = float(
                predicted_current == target_current
            )
            scores["step_t"][step_index] = float(
                scores["V_t"][step_index] == 1.0
                and scores["D_t"][step_index] == 1.0
                and scores["N_t"][step_index] == 1.0
                and scores["C_prev"][step_index] == 1.0
                and scores["C_t"][step_index] == 1.0
            )

        predicted_answer = int(answer_text)
        scores["arithmetic"] = float(
            all(
                predicted_previous + predicted_new == predicted_current
                for _, predicted_previous, predicted_new, predicted_current in steps
            )
        )
        scores["chain"] = float(
            steps[0][1] == 0
            and all(
                current[1] == previous[3]
                for previous, current in zip(steps[:-1], steps[1:], strict=True)
            )
            and predicted_answer == steps[-1][3]
        )
        scores["protocol"] = (
            scores["format"]
            * scores["arithmetic"]
            * scores["chain"]
            * scores["eos"]
        )
        scores["answer"] = float(
            predicted_answer == int(ground_truth["final_answer"])
        )
        scores["complete"] = float(
            scores["protocol"] == 1.0
            and scores["answer"] == 1.0
            and all(step_reward == 1.0 for step_reward in scores["step_t"])
        )
        return scores

    def score(self, response_ids, ground_truth_string, stop_reason=None):
        ground_truth = json.loads(ground_truth_string)
        response_ids = list(response_ids)
        eos = bool(
            stop_reason == "stop"
            or (
                response_ids
                and response_ids[-1] == self.tokenizer.eos_token_id
            )
        )
        while response_ids and response_ids[-1] in self.special_ids:
            response_ids.pop()
        output = self.tokenizer.decode(
            response_ids,
            skip_special_tokens=False,
        )
        scores = self._score_rollout(output, ground_truth, eos)
        n_steps = len(scores["V_t"])
        rewards = {
            "V_t": sum(scores["V_t"]) / n_steps,
            "C_prev": sum(scores["C_prev"]) / n_steps,
            "D_t_N_t": sum(
                scores["D_t"][step_index]
                * scores["N_t"][step_index]
                for step_index in range(1, n_steps)
            )
            / (n_steps - 1),
            "C_t": sum(
                scores["C_prev"][step_index]
                * scores["N_t"][step_index]
                * scores["C_t"][step_index]
                for step_index in range(n_steps)
            )
            / n_steps,
            "format": scores["format"],
            "answer": scores["answer"],
        }
        total_weight = sum(P3_REWARD_WEIGHTS.values())
        total_reward = sum(
            P3_REWARD_WEIGHTS[name] * rewards[name]
            for name in P3_REWARD_WEIGHTS
        ) / total_weight
        return {
            "score": total_reward,
            "r_format": scores["format"],
            "answer_exact_reward": scores["answer"],
            "complete_reward": scores["complete"],
            "visible_reward": rewards["V_t"],
            "memory_reward": rewards["C_prev"],
            "association_reward": rewards["D_t_N_t"],
            "state_reward": rewards["C_t"],
        }


__all__ = ["P3Reward", "P3_REWARD_WEIGHTS"]
"""vsiqa"""
