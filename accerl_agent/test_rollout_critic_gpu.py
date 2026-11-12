"""Opt-in integration check using a tiny local model, with no downloads.

ACCERL_RUN_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=<idle GPU> <project-python> \
    -m unittest accerl_agent.test_rollout_critic_gpu -v
"""
from __future__ import annotations

import os
import tempfile
import unittest

import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@unittest.skipUnless(os.environ.get("ACCERL_RUN_GPU_TESTS") == "1", "opt-in GPU integration test")
class RolloutCriticGPUIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_chunked_values_match_hf_and_update_with_snapshot(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from accerl_agent.agent_textworld import create_async_engine, shutdown_vllm_engine
        from accerl_agent.ppo_value import TokenValueHead
        from accerl_agent.rollout_critic import RolloutCriticClient

        with tempfile.TemporaryDirectory(prefix="accerl-critic-smoke-") as directory:
            torch.manual_seed(37)
            model = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=128, hidden_size=64, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=256, bos_token_id=1, eos_token_id=2,
                pad_token_id=0, tie_word_embeddings=False,
            )).to(dtype=torch.bfloat16).eval()
            model.save_pretrained(directory)
            engine = create_async_engine(
                model=directory, skip_tokenizer_init=True, dtype="bfloat16",
                enforce_eager=True, max_model_len=128, max_num_seqs=4,
                max_num_batched_tokens=32, enable_chunked_prefill=True,
                enable_prefix_caching=True, gpu_memory_utilization=0.05,
                distributed_executor_backend="mp",
                worker_extension_cls="accerl_agent.rollout_critic.RolloutCriticWorkerExtension",
            )
            try:
                client = RolloutCriticClient(engine)
                head = TokenValueHead(64).eval()
                with torch.no_grad():
                    head.weight.copy_(torch.linspace(-0.05, 0.05, 64).view(1, -1))
                    head.bias.fill_(0.2)
                state = {name: value.tolist() for name, value in head.state_dict().items()}
                await client.load(state, 3)
                client.resume()
                transcripts = [[3 + i % 100 for i in range(61)], [4 + i % 99 for i in range(37)]]
                result = await client.score(transcripts)
                self.assertEqual(result["value_version"], 3)
                model = model.to("cuda")
                head = head.to("cuda")
                with torch.inference_mode():
                    for tokens, actual in zip(transcripts, result["token_values"]):
                        hidden = model.model(torch.tensor([tokens], device="cuda")).last_hidden_state[0]
                        expected = head(hidden).cpu()
                        torch.testing.assert_close(torch.tensor(actual), expected, atol=0.006, rtol=0.02)

                # Same input must be recomputed, not served from prefix cache;
                # changing the head must change every returned token value.
                await client.pause()
                await engine.pause_generation(mode="abort", clear_cache=True)
                state["bias"] = [0.7]
                await client.load(state, 4)
                await engine.resume_generation()
                client.resume()
                updated = await client.score(transcripts)
                self.assertEqual(updated["value_version"], 4)
                for before, after in zip(result["token_values"], updated["token_values"]):
                    torch.testing.assert_close(torch.tensor(after), torch.tensor(before) + 0.5, atol=1e-5, rtol=1e-5)
                await client.pause()
                client.close()
            finally:
                await shutdown_vllm_engine(engine)


if __name__ == "__main__":
    unittest.main()
