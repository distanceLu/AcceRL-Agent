# Repository instructions

## Directory responsibilities

- `accerl_agent/` is the original AcceRL reference implementation. Treat it as
  read-only unless the user explicitly requests changes there.
- The primary VSI-QA implementation consists of Python files directly under
  `vsi_qa_rlvr/`, including `main.py`, `trainer.py`, `synchronization.py`,
  `vllm_rollout_actor.py`, `rollout.py`, `replay.py`, `trajectory.py`, and
  `inference.py`.
- These root-level `vsi_qa_rlvr/*.py` files are the split and aligned form of
  the original implementation under `accerl_agent/`. When porting fixes or
  behavior, use `accerl_agent/` as the baseline but apply changes to the
  corresponding root-level `vsi_qa_rlvr/*.py` files.
- `vsi_qa_rlvr/scannet_incremental_counting/` is the existing Qwen3-VL
  multi-image adaptation. Use it as the task-specific Qwen reference. Do not
  modify it unless the user explicitly names that directory.

## Weight synchronization invariants

- Capture weight metadata before `fully_shard()`.
- Keep `names`, `dtype_names`, and `shapes` identical in length and ordering.
- Metadata generation and the actual weight iterator must apply the same name
  conversion and tensor-splitting rules.
- Metadata does not replace `full_tensor()` during synchronization.
- The current Qwen3-VL transfer uses checkpoint-format weights; vLLM's
  `model.load_weights()` performs model-specific parameter mapping and packing.
- Use scope `all` for initial synchronization; later synchronization may use
  scope `trainable`.
- Do not introduce a separate Qwen3-VL kernel-weight converter unless the user
  explicitly requests kernel-format transfer.

## Change discipline

- Preserve existing uncommitted changes.
- Outside paired `"""vsiqa"""` blocks, preserve the corresponding AcceRL
  class names, function and method names, signatures, member and local variable
  names, method order, control flow, calls, return values, exceptions, logs,
  and state updates exactly.
- Do not add `VSIQA` prefixes or rename a public AcceRL symbol merely to label
  the task-specific copy. Keep names such as `FSDPTrainWorker`,
  `VLLMInferenceActor`, `RolloutWorkerActor`, `RLSample`, and
  `run_weight_sync_demo` identical to the reference.
- Before editing, map the target root-level `vsi_qa_rlvr` file to its
  `accerl_agent` reference and, where relevant, its
  `scannet_incremental_counting` Qwen adaptation.
- Do not copy an entire reference file over a split module.
- Keep task-specific image or video processing separate from generic FSDP,
  synchronization, and vLLM transaction code.
- Surround each Qwen/VSI-QA-specific deviation from the AcceRL reference with
  matching `"""vsiqa"""` strings at the beginning and end of the adapted
  block. Do not add `start` or `end` words to the markers.
- Treat alignment as proven only when every remaining difference is contained
  in a paired Qwen/VSI-QA block and the unmarked public logic matches the
  corresponding AcceRL implementation.
