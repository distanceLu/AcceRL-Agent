# SPDX-License-Identifier: Apache-2.0
"""Run the ScanNet task on AcceRL's unchanged async main loop."""

import asyncio

from vsi_qa_rlvr import main as accerl_main
from vsi_qa_rlvr.scannet_incremental_counting.rollout_worker import (
    ScanNetIncrementalCountingRolloutWorkerActor,
)
from vsi_qa_rlvr.scannet_incremental_counting.trainer import (
    ScanNetIncrementalCountingFSDPTrainWorker,
)
from vsi_qa_rlvr.scannet_incremental_counting.vllm_actor import (
    ScanNetIncrementalCountingVLLMInferenceActor,
)


def main():
    args = accerl_main.parse_args()
    accerl_main.validate_args(args)
    accerl_main.FSDPTrainWorker = (
        ScanNetIncrementalCountingFSDPTrainWorker
    )
    accerl_main.VSIQAVLLMInferenceActor = (
        ScanNetIncrementalCountingVLLMInferenceActor
    )
    accerl_main.VSIQARolloutWorkerActor = (
        ScanNetIncrementalCountingRolloutWorkerActor
    )
    asyncio.run(accerl_main.run_vsi_qa(args))


if __name__ == "__main__":
    main()
