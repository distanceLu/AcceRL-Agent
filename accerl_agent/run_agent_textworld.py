"""Canonical launcher for the Ray TextWorld training pipeline."""

import os
from pathlib import Path

# FlashInfer appends .cache/flashinfer to this base and ignores XDG_CACHE_HOME.
# Set it before importing vLLM or starting Ray so child workers inherit it.
os.environ.setdefault(
    "FLASHINFER_WORKSPACE_BASE", str(Path(__file__).resolve().parents[1])
)

from accerl_agent.agent_textworld import main


if __name__ == "__main__":
    main()
