"""Compose optional research observers with rollout critic RPCs."""
from accerl_agent.rollout_critic import RolloutCriticWorkerExtension
from testForAcceRL.vllm_backend import ResearchWorker


class ResearchRolloutCriticWorker(RolloutCriticWorkerExtension, ResearchWorker):
    pass
