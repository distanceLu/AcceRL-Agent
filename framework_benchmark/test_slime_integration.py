"""Run under the existing Slime interpreter; no CUDA contexts or model weights."""
from argparse import Namespace, ArgumentParser
import unittest
from unittest.mock import patch

import torch

from framework_benchmark.runtime import model_args
from framework_benchmark.slime_adapter import compute_advantages


class SlimeTests(unittest.TestCase):
    def test_qwen_router_selects_before_bf16_rounding(self):
        from megatron.core.transformer.moe import router
        from framework_benchmark.qwen2_router import install
        original = router.topk_routing_with_score_function
        try:
            install()
            generator = torch.Generator().manual_seed(42)
            logits = (torch.randn(64,60,generator=generator)*.03).bfloat16()
            probabilities, mapping = router.topk_routing_with_score_function(logits,4,use_pre_softmax=True)
            scores = logits.float().softmax(-1)
            values, indices = scores.topk(4,dim=-1)
            expected = torch.zeros_like(scores).scatter(1,indices,values).bfloat16()
            torch.testing.assert_close(probabilities,expected,rtol=0,atol=0)
            self.assertTrue(torch.equal(mapping,expected!=0))
        finally:
            router.topk_routing_with_score_function = original

    def test_real_textworld_worker_across_python_environments(self):
        import asyncio
        from pathlib import Path
        from framework_benchmark import slime_adapter as adapter
        game = next(Path('/mnt/data/lcx4/AcceRL/accerl_vllm/data/tw_games/ftwp/games/train').glob('*.z8'))
        async def check():
            process = await adapter.acquire()
            try:
                initial = await adapter.rpc(process, {'op':'reset','game':str(game),'seed':42})
                self.assertIn('Admissible commands:', initial['content'])
                for _ in range(40):
                    state = await adapter.rpc(process, {'op':'step','text':'INVALID COMMAND'})
                    self.assertFalse(state['env_step'])
                    self.assertEqual(state['reward'], -.1)
                self.assertEqual(state['reason'], 'step_limit')
            finally:
                process.terminate()
                await process.wait()
                adapter._processes.clear()
                adapter._pool = None
        asyncio.run(check())

    def test_model_arguments_are_recognized(self):
        from megatron.training.arguments import add_megatron_arguments
        args = add_megatron_arguments(ArgumentParser()).parse_args(model_args())
        self.assertTrue(args.moe_shared_expert_gate)
        self.assertTrue(args.moe_router_pre_softmax)
        self.assertTrue(args.add_qkv_bias)
        self.assertEqual(args.num_experts, 60)
        self.assertFalse(args.qk_layernorm)

    def test_registered_checkpoint_loader(self):
        from mbridge import AutoBridge
        from mbridge.core.parallel_states import ParallelStates
        bridge = AutoBridge.from_pretrained('/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat',
                                             parallel_states=ParallelStates())
        self.assertEqual(type(bridge).__name__, 'Qwen2MoEBridge')
        self.assertEqual(bridge.config.num_moe_experts, 60)

    def test_export_experts_shared_gate_and_qkv_bias(self):
        from slime.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
        args = Namespace(kv_channels=2, hidden_size=4, num_attention_heads=2, num_query_groups=2, q_lora_rank=None)
        prefix = 'module.module.decoder.layers.0.'
        fc1 = torch.arange(24).reshape(6, 4)
        out = dict(_convert_to_hf_core(args, 'qwen2moe', prefix+'mlp.experts.linear_fc1.weight59', fc1))
        torch.testing.assert_close(out['model.layers.0.mlp.experts.59.gate_proj.weight'], fc1[:3])
        torch.testing.assert_close(out['model.layers.0.mlp.experts.59.up_proj.weight'], fc1[3:])
        out = dict(_convert_to_hf_core(args, 'qwen2moe', prefix+'mlp.shared_experts.gate_weight', torch.ones(1,4)))
        self.assertIn('model.layers.0.mlp.shared_expert_gate.weight', out)
        out = dict(_convert_to_hf_core(args, 'qwen2moe', prefix+'self_attention.linear_qkv.bias', torch.arange(12)))
        torch.testing.assert_close(out['model.layers.0.self_attn.q_proj.bias'], torch.tensor([0,1,6,7]))

    def test_critic_and_actor_targets_use_action_rewards(self):
        args = Namespace(context_parallel_size=1, kl_coef=0, gamma=1, lambd=1)
        data = {'values': [torch.tensor([.5,99,.25])], 'loss_masks': [torch.tensor([1,0,1])],
                'metadata': [{'token_rewards': [1,0,2], 'termination_reason':'step_limit'}]}
        compute_advantages(args, data)
        torch.testing.assert_close(data['returns'][0], torch.tensor([3.,0.,2.]))

    def test_metadata_survives_actual_dp_split(self):
        from slime.ray import rollout as module
        cls = module.RolloutManager.__ray_metadata__.modified_class
        manager = object.__new__(cls)
        manager.args = Namespace(global_batch_size=2, rollout_data_transport='object-store')
        manager.train_parallel_config = {'dp_size': 2}
        data = {'tokens': [[1,2],[3,4]], 'rollout_ids':[0,1], 'response_lengths':[1,1],
                'loss_masks':[[1],[1]], 'metadata':[{'token_rewards':[1]},{'token_rewards':[2]}]}
        schedule = ([[1],[0]], [[[0]],[[0]]], [1], [2])
        with patch.object(module, 'build_dp_schedule', return_value=schedule), patch.object(module.ray, 'put', side_effect=lambda x:x), patch.object(module, 'Box', side_effect=lambda x:x):
            result = manager._split_train_data_by_dp(data)
        self.assertEqual(result[0]['metadata'], [{'token_rewards':[2]}])
        self.assertEqual(result[1]['metadata'], [{'token_rewards':[1]}])


if __name__ == '__main__': unittest.main()
