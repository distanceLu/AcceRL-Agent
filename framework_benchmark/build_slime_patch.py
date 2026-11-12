"""Create a reviewable patch against the pinned upstream source; never applies it."""
import difflib
from pathlib import Path

ROOT = Path('/mnt/data/lcx4/slime')
EDITS = {
    'slime/ray/rollout.py': [
        ('                "sample_indices",\n                "rollout_ids",',
         '                "sample_indices",\n                "metadata",\n                "rollout_ids",')],
    'slime/utils/arguments.py': [
        ('        role_args.custom_advantage_function_path = None\n',
         '        # Task-specific reward placement must also define critic targets.\n')],
    'slime/backends/megatron_utils/megatron_to_hf/__init__.py': [
        ('    elif "qwen3moe" in model_name:\n',
         '    elif "qwen3moe" in model_name or "qwen2moe" in model_name or "qwen2_moe" in model_name:\n')],
}


def main():
    patch = []
    for name, edits in EDITS.items():
        old = (ROOT/name).read_text()
        new = old
        for before, after in edits:
            if new.count(before) != 1:
                raise ValueError(f'Source changed or patch already applied: {name}')
            new = new.replace(before, after)
        patch.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                         fromfile='a/'+name, tofile='b/'+name))
    Path(__file__).with_name('slime_integration.patch').write_text(''.join(patch))


if __name__ == '__main__': main()
