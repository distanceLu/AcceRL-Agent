# SPDX-License-Identifier: Apache-2.0
"""Prepare Qwen3-VL rollout and Replay media inputs for AcceRL."""

import os

from transformers import AutoProcessor


class Qwen3VLRolloutDataCollator:
    """Create matching Qwen3-VL Trainer tensors and vLLM EngineInputs."""

    def __init__(self, model_path, max_model_len=65536):
        self.model_path = model_path
        self.max_model_len = int(max_model_len)
        self.renderer = None
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

    def __call__(self, items):
        if self.renderer is None:
            from vllm import AsyncEngineArgs
            from vllm.renderers import renderer_from_config

            engine_args = AsyncEngineArgs(
                model=self.model_path,
                trust_remote_code=True,
                dtype="bfloat16",
                load_format="dummy",
                max_model_len=self.max_model_len,
                allowed_local_media_path="/",
                limit_mm_per_prompt={"video": 1},
            )
            self.renderer = renderer_from_config(
                engine_args.create_engine_config()
            )

        batch = []
        for item in items:
            prompt = self.processor.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            frame_indices = [
                int(
                    os.path.splitext(os.path.basename(path))[0][
                        len("frame_"):
                    ]
                )
                - 1
                for path in item["frame_paths"]
            ]
            total_num_frames = frame_indices[-1] + 1
            video_metadata = {
                "fps": item["sample_fps"],
                "frames_indices": frame_indices,
                "total_num_frames": total_num_frames,
                "duration": total_num_frames / item["sample_fps"],
            }
            video_size = {
                "shortest_edge": item["min_pixels"],
                "longest_edge": item["max_pixels"],
            }

            encoded = self.processor(
                text=[prompt],
                videos=[item["video"]],
                video_metadata=[video_metadata],
                do_sample_frames=False,
                size=video_size,
                return_mm_token_type_ids=True,
                return_tensors="pt",
                padding=True,
            )
            valid = encoded["attention_mask"][0].bool()
            prepared_media = {
                "prompt_token_ids": encoded["input_ids"][0][valid].contiguous(),
                "prompt_mm_token_type_ids": (
                    encoded["mm_token_type_ids"][0][valid].contiguous()
                ),
                "pixel_values_videos": (
                    encoded["pixel_values_videos"].contiguous()
                ),
                "video_grid_thw": encoded["video_grid_thw"].contiguous(),
                "pad_token_id": int(self.processor.tokenizer.pad_token_id),
            }

            media_key = f"vsi_qa:{item['sample_id']}"
            vllm_video_metadata = dict(video_metadata)
            vllm_video_metadata["do_sample_frames"] = False
            vllm_prompt = {
                "prompt_token_ids": self.processor.tokenizer.encode(
                    prompt,
                    add_special_tokens=True,
                ),
                "multi_modal_data": {
                    "video": [(item["video"], vllm_video_metadata)]
                },
                "multi_modal_uuids": {"video": [media_key]},
                "mm_processor_kwargs": {
                    "do_sample_frames": False,
                    "size": video_size,
                },
            }
            engine_inputs = self.renderer.render_cmpl(
                [vllm_prompt],
                skip_mm_cache=True,
            )
            if len(engine_inputs) != 1:
                raise RuntimeError(
                    "vLLM Renderer must return exactly one EngineInput, got "
                    f"{len(engine_inputs)}"
                )
            llm_input = engine_inputs[0]
            if llm_input.get("type") != "multimodal":
                raise RuntimeError(
                    "VSI-QA video prompt must render to a multimodal "
                    f"EngineInput, got {llm_input.get('type')!r}"
                )

            rendered_prompt_ids = list(llm_input["prompt_token_ids"])
            training_prompt_ids = prepared_media["prompt_token_ids"].tolist()
            if rendered_prompt_ids != training_prompt_ids:
                raise RuntimeError(
                    "vLLM and Trainer prompt token IDs diverged: "
                    f"vllm={len(rendered_prompt_ids)} "
                    f"trainer={len(training_prompt_ids)} "
                    f"sample_id={item['sample_id']!r}"
                )

            batch.append(
                {
                    "sample_id": item["sample_id"],
                    "reward_model": item["reward_model"],
                    "extra_info": item["extra_info"],
                    "llm_input": llm_input,
                    "prepared_media": prepared_media,
                }
            )
        return batch


__all__ = ["Qwen3VLRolloutDataCollator"]
