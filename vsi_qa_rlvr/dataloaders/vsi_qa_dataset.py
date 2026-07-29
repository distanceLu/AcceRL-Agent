# SPDX-License-Identifier: Apache-2.0
"""VSI-QA Dataset for parquet rows, messages, and selected JPEG frames.

Dataset reading and media loading live here. Qwen3-VL tokenization and tensor
construction belong in the model-specific collators.
"""

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def load_rgb_video(frame_paths):
    with Image.open(frame_paths[0]) as image:
        first_frame = np.array(image.convert("RGB"), dtype=np.uint8)
    video = np.empty(
        (len(frame_paths), first_frame.shape[0], first_frame.shape[1], 3),
        dtype=np.uint8,
    )
    video[0] = first_frame
    for index, path in enumerate(frame_paths[1:], start=1):
        with Image.open(path) as image:
            video[index] = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(video)


class VSIQADataset(Dataset):
    """Return one message sequence, decoded video, and reward record."""

    def __init__(self, data_path):
        self.data_path = data_path
        self.row_count = pq.ParquetFile(data_path).metadata.num_rows
        self.rows = None

    def __len__(self):
        return self.row_count

    def __getitem__(self, row_index):
        if self.rows is None:
            self.rows = pq.read_table(
                self.data_path,
                columns=[
                    "sample_id",
                    "prompt",
                    "videos",
                    "reward_model",
                    "extra_info",
                ],
                memory_map=True,
            )

        row_index = int(row_index)
        row = self.rows.slice(row_index, 1).to_pylist()[0]
        video_info = row["videos"][0]
        frame_paths = video_info["video"]
        system_message, user_message = row["prompt"]
        question = user_message["content"][len("<video>\n"):]
        return {
            "row_index": row_index,
            "sample_id": row["sample_id"],
            "question": question,
            "messages": [
                system_message,
                {
                    "role": user_message["role"],
                    "content": [
                        {"type": "video"},
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            "video": load_rgb_video(frame_paths),
            "frame_paths": frame_paths,
            "sample_fps": float(video_info["sample_fps"]),
            "min_pixels": int(video_info["min_pixels"]),
            "max_pixels": int(video_info["max_pixels"]),
            "reward_model": row["reward_model"],
            "extra_info": row["extra_info"],
        }


__all__ = ["VSIQADataset"]
