"""Compute reference logits with the actual AcceRL Transformers/FlashAttention stack."""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    'You are a TextWorld action selector.\nObservation: You are in the kitchen.\nCommand: take apple',
    'Inventory: a knife.\nAdmissible commands:\n- go north\n- open fridge\nCommand: open fridge',
    'Objective: prepare a meal.\nObservation: The stove is switched off.\nCommand: turn on stove',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    path = '/mnt/data/lcx4/hf_cache/Qwen1.5-MoE-A2.7B-Chat'
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                attn_implementation='flash_attention_2').to('cuda').eval()
    cases = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            tokens = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
            logits = model(tokens,use_cache=False).logits.float().cpu()
            cases.append({'tokens':tokens.cpu(), 'logits':logits})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(cases,args.output)
    print(f'Saved {len(cases)} reference forward cases to {args.output}')


if __name__ == '__main__': main()
