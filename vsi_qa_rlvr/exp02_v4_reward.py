# SPDX-License-Identifier: Apache-2.0
"""Fixed Exp02 v4 training reward for VSI-QA RLVR."""

import fcntl
import json
import os
import tempfile

from vsi_qa_rlvr import reward_exp02_v4


REWARD_CONFIG = {
    "history_window": 1280,
    "history_min_rows": 128,
    "answer_exact_weight": 0.60,
    "process_joint_weight": 0.40,
    "shortcut_penalty_weight": 0.30,
    "history_regression_weight": 0.20,
    "answer_stable_threshold": 0.50,
}


def score_vsi_qa_group(
    solution_strings,
    ground_truth,
    extra_info,
    history_path,
):
    if not solution_strings:
        return []

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "a+", encoding="utf-8") as history_handle:
        fcntl.flock(history_handle.fileno(), fcntl.LOCK_EX)
        try:
            history_handle.seek(0)
            frozen_history_lines = history_handle.read().splitlines()[
                -int(REWARD_CONFIG["history_window"]):
            ]
            details = []
            appended_history_rows = []
            with tempfile.TemporaryDirectory(
                prefix="vsi-reward-group-",
                dir=os.path.dirname(history_path),
            ) as temporary_directory:
                for candidate_index, solution_string in enumerate(
                    solution_strings
                ):
                    candidate_history_path = os.path.join(
                        temporary_directory,
                        f"candidate_{candidate_index:04d}.jsonl",
                    )
                    with open(
                        candidate_history_path,
                        "w",
                        encoding="utf-8",
                    ) as candidate_history_handle:
                        for line in frozen_history_lines:
                            candidate_history_handle.write(line + "\n")
                    candidate_details = reward_exp02_v4.score_train(
                        solution_string,
                        ground_truth,
                        extra_info,
                        history_path=candidate_history_path,
                        **REWARD_CONFIG,
                    )
                    with open(
                        candidate_history_path,
                        encoding="utf-8",
                    ) as candidate_history_handle:
                        candidate_history_lines = (
                            candidate_history_handle.read().splitlines()
                        )
                    details.append(candidate_details)
                    appended_history_rows.append(
                        json.loads(candidate_history_lines[-1])
                    )

            history_handle.seek(0, os.SEEK_END)
            for history_row in appended_history_rows:
                history_handle.write(
                    json.dumps(history_row, ensure_ascii=False) + "\n"
                )
            history_handle.flush()
        finally:
            fcntl.flock(history_handle.fileno(), fcntl.LOCK_UN)
    return details
