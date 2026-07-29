#!/usr/bin/env python3
# Exp02 v4 RLVR reward for VSI-590K chair counting.
#
# Input data:
# /data/all/luck/derived_dataset/VSI_590K_derived/chair_count_fps2_qwen_labels_verl_video_summary_xml_max*/train.parquet
#
# v4 keeps the short Exp01/V3 XML output, but process reward is only given when
# answer, chair_count, duplicate_chair_count, and consistency all match the
# deterministic labels. Format and consistency are recorded, not rewarded.

from collections import deque
import fcntl
import json
import os
import re


TRAIN_XML_RE = re.compile(
    r"^\s*<process>\s*"
    r"<target_object>\s*椅子\s*</target_object>\s*"
    r"<chair_count>\s*(?P<chair_count>-?\d+)\s*</chair_count>\s*"
    r"<duplicate_chair_count>\s*(?P<duplicate_chair_count>-?\d+)\s*</duplicate_chair_count>\s*"
    r"</process>\s*"
    r"<answer>\s*(?P<answer>-?\d+)\s*</answer>\s*$",
    re.S,
)
ANSWER_RE = re.compile(r"<answer>\s*(?P<answer>-?\d+)\s*</answer>\s*$", re.S)

_history_path = ""
_history_offset = 0
_history_window = 0
_history_rows = deque()


def parse_train_xml(solution_str):
    match = TRAIN_XML_RE.fullmatch(solution_str)
    if not match:
        return None
    return {
        "answer": int(match.group("answer")),
        "chair_count": int(match.group("chair_count")),
        "duplicate_chair_count": int(match.group("duplicate_chair_count")),
    }


def parse_answer(solution_str):
    match = ANSWER_RE.search(solution_str)
    if not match:
        return -1
    return int(match.group("answer"))


def read_tail_lines(path, max_lines):
    if not os.path.exists(path):
        return [], 0

    chunks = []
    newline_count = 0
    with open(path, "rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        position = file_size
        while position > 0 and newline_count <= max_lines:
            read_size = min(1024 * 1024, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    if not chunks:
        return [], file_size
    lines = b"".join(reversed(chunks)).splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return [line.decode("utf-8") for line in lines if line.strip()], file_size


def load_history_window(history_path, history_window):
    global _history_path, _history_offset, _history_window, _history_rows

    if not history_path or history_window <= 0:
        return []

    if history_path != _history_path or history_window != _history_window:
        lines, file_size = read_tail_lines(history_path, history_window)
        _history_path = history_path
        _history_offset = file_size
        _history_window = history_window
        _history_rows = deque(maxlen=history_window)
        for line in lines:
            _history_rows.append(json.loads(line))
        return list(_history_rows)

    if os.path.exists(history_path):
        file_size = os.path.getsize(history_path)
        if file_size < _history_offset:
            lines, file_size = read_tail_lines(history_path, history_window)
            _history_rows = deque(maxlen=history_window)
            for line in lines:
                _history_rows.append(json.loads(line))
            _history_offset = file_size
        elif file_size > _history_offset:
            with open(history_path, "r", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                handle.seek(_history_offset)
                new_lines = handle.read().splitlines()
                _history_offset = handle.tell()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            for line in new_lines:
                if line.strip():
                    _history_rows.append(json.loads(line))

    return list(_history_rows)


def history_stats(rows):
    return {
        "history_count": len(rows),
        "recent_answer_rate": (
            sum(float(row["r_answer_exact"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }


def append_history(history_path, row):
    if not history_path:
        return
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def score_train(
    solution_str,
    ground_truth,
    extra_info,
    history_path="",
    history_window=1280,
    history_min_rows=128,
    answer_exact_weight=0.60,
    process_joint_weight=0.40,
    shortcut_penalty_weight=0.30,
    history_regression_weight=0.20,
    answer_stable_threshold=0.50,
):
    rows = load_history_window(history_path, int(history_window))
    stats = history_stats(rows)

    answer = int(ground_truth)
    parsed = parse_train_xml(solution_str)
    parsed_answer = parse_answer(solution_str)
    r_answer_exact = 1.0 if parsed_answer == answer else 0.0

    r_format = 1.0 if parsed is not None else 0.0
    r_consistency = 0.0
    r_chair_count = 0.0
    r_duplicate_chair_count = 0.0
    r_process_joint = 0.0
    shortcut_flag = 0.0

    process_label_usable = bool(extra_info["process_label_usable"])
    if parsed is not None:
        nonnegative_counts = parsed["chair_count"] >= 0 and parsed["duplicate_chair_count"] >= 0 and parsed["answer"] >= 0
        answer_from_process = parsed["answer"] == parsed["chair_count"] - parsed["duplicate_chair_count"]
        r_consistency = 1.0 if nonnegative_counts and answer_from_process else 0.0

        if process_label_usable:
            gold_chair_count = int(extra_info["chair_count"])
            gold_duplicate_chair_count = int(extra_info["duplicate_chair_count"])
            r_chair_count = 1.0 if parsed["chair_count"] == gold_chair_count else 0.0
            r_duplicate_chair_count = 1.0 if parsed["duplicate_chair_count"] == gold_duplicate_chair_count else 0.0
            r_process_joint = 1.0 if (
                r_answer_exact == 1.0
                and r_consistency == 1.0
                and r_chair_count == 1.0
                and r_duplicate_chair_count == 1.0
            ) else 0.0
            if parsed["duplicate_chair_count"] == 0 and gold_duplicate_chair_count > 0:
                shortcut_flag = 1.0

    if parsed is None:
        answer_exact_reward = 0.0
    else:
        answer_exact_reward = float(answer_exact_weight) * r_answer_exact
    process_reward = float(process_joint_weight) * r_process_joint

    history_regression_flag = 1.0 if (
        stats["history_count"] >= int(history_min_rows)
        and stats["recent_answer_rate"] >= float(answer_stable_threshold)
        and r_answer_exact == 0.0
    ) else 0.0
    history_regression = float(history_regression_weight) * history_regression_flag

    shortcut_penalty = float(shortcut_penalty_weight) * shortcut_flag

    if parsed is None:
        score = 0.0
    else:
        score = answer_exact_reward + process_reward - history_regression - shortcut_penalty
        score = min(1.0, max(0.0, score))

    result = {
        "score": score,
        "answer_exact_reward": answer_exact_reward,
        "r_format": r_format,
    }
    append_history(history_path, {
        "r_answer_exact": r_answer_exact,
    })
    return result
