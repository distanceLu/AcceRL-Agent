"""Invoke upstream conversion with a pre-save full-model HF/Megatron parity gate."""
import importlib.util
import json
from pathlib import Path


def main():
    spec = importlib.util.spec_from_file_location('slime_conversion', '/mnt/data/lcx4/slime/tools/convert_hf_to_torch_dist.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from framework_benchmark.qwen2_router import install
    install()
    upstream_save = module.save_checkpoint

    def checked_save(iteration, models, *positional, **kwargs):
        import torch
        from megatron.training import get_args
        args = get_args()
        # Exact tensor round-trip distinguishes conversion errors from execution
        # differences. This gate covers every original checkpoint parameter.
        from safetensors import safe_open
        from slime.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
        index = json.loads(Path(args.hf_checkpoint,'model.safetensors.index.json').read_text())['weight_map']
        handles = {name:safe_open(str(Path(args.hf_checkpoint,name)),framework='pt',device='cpu') for name in set(index.values())}
        seen, mismatches = set(), []
        for name, param in models[0].named_parameters():
            while name.startswith('module.'):
                name = name[len('module.'):]
            pairs = _convert_to_hf_core(args,'qwen2moe','module.module.'+name,param.detach())
            for hf_name, tensor in pairs:
                seen.add(hf_name)
                original = handles[index[hf_name]].get_tensor(hf_name)
                if tensor.shape != original.shape or not torch.equal(tensor.cpu(),original):
                    mismatches.append(hf_name)
        weights = {'passed':not mismatches and seen==set(index),'checked':len(seen),
                   'mismatches':mismatches,'missing':sorted(set(index)-seen)}
        Path(args.save).parent.joinpath('weight_parity.json').write_text(json.dumps(weights,indent=2)+'\n')
        if not weights['passed']:
            raise ValueError('Exact weight round-trip failed')
        reference_cases = torch.load(Path(__file__).parent/'artifacts/accerl_reference.pt', weights_only=True)
        models[0].eval()
        reports = []
        with torch.inference_mode():
            for case in reference_cases:
                tokens = case['tokens'].cuda()
                positions = torch.arange(tokens.shape[1], device='cuda').unsqueeze(0)
                hf = case['logits'].cuda()
                mc = models[0](tokens, positions, None).float()[..., :hf.shape[-1]]
                if mc.shape != hf.shape:
                    raise ValueError(f'Logit shapes differ: {mc.shape} != {hf.shape}')
                hp, mp = hf.log_softmax(-1), mc.log_softmax(-1)
                kl = (hp.exp()*(hp-mp)).sum(-1)
                actual = tokens[:, 1:].unsqueeze(-1)
                difference = (hp[:, :-1].gather(-1, actual)-mp[:, :-1].gather(-1, actual)).abs()
                reports.append({'tokens': tokens.numel(), 'mean_kl':kl.mean().item(),
                                'max_kl':kl.max().item(), 'max_token_logprob_difference':difference.max().item(),
                                'token_logprob_differences':difference.flatten().tolist(),
                                'mean_logit_difference':(hf-mc).abs().mean().item()})
        passed = all(r['mean_kl'] < .02 and r['max_token_logprob_difference'] < .2 for r in reports)
        report = {'passed':passed, 'tolerances':{'mean_kl':.02, 'max_token_logprob_difference':.2},
                  'precision':'bf16', 'hf_attention':'flash_attention_2', 'reference_environment':'accerl-agent', 'cases':reports}
        Path(args.save).parent.joinpath('forward_parity.json').write_text(json.dumps(report,indent=2)+'\n')
        print('FORWARD_PARITY',json.dumps(report),flush=True)
        if not passed:
            raise ValueError('HF/Megatron forward parity gate failed; checkpoint not saved')
        torch.cuda.empty_cache()
        return upstream_save(iteration, models, *positional, **kwargs)

    module.save_checkpoint = checked_save
    try:
        module.main()
    finally:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__': main()
