"""Compute Generalized Advantage Estimation (GAE) with PyTorch.

Example:
    python compute_GAE.py --rewards '[1, 1, 1]' \
        --values '[0.5, 0.6, 0.7]' --bootstrap '0.0'
"""

import argparse
import json

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the GAE advantages and TD residuals (deltas).

    ``rewards`` and ``values`` must be one-dimensional tensors with the same
    shape. For a terminal trajectory, leave ``bootstrap`` at zero. For a
    non-terminal trajectory segment, pass the value of the following state.

    Args:
        rewards: Rewards with shape ``(T,)``.
        values: State values with shape ``(T,)``.
        gamma: Discount factor.
        gae_lambda: GAE lambda coefficient.
        bootstrap: Scalar value following the last step.

    Returns:
        ``(gae, deltas)``, both tensors with shape ``(T,)`` on the same
        device and with the same dtype as the inputs.
    """
    if rewards.ndim != 1 or values.ndim != 1 or rewards.shape != values.shape:
        raise ValueError(
            f"rewards and values must be 1-D tensors with the same shape, got "
            f"{rewards.shape} and {values.shape}"
        )

    bootstrap = torch.as_tensor(bootstrap, dtype=values.dtype, device=values.device)
    next_values = torch.cat((values[1:], bootstrap.reshape(1)))

    deltas = rewards + gamma * next_values - values
    gae = torch.empty_like(deltas)
    running_gae = torch.zeros_like(deltas[0])
    for index in range(len(rewards) - 1, -1, -1):
        running_gae = deltas[index] + gamma * gae_lambda * running_gae
        gae[index] = running_gae

    return gae, deltas


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GAE and TD deltas.")
    parser.add_argument("--rewards", required=True, type=json.loads)
    parser.add_argument("--values", required=True, type=json.loads)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=json.loads, default=0.0)
    args = parser.parse_args()

    gae, deltas = compute_gae(
        torch.tensor(args.rewards, dtype=torch.float32),
        torch.tensor(args.values, dtype=torch.float32),
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        bootstrap=args.bootstrap,
    )
    print(json.dumps({"gae": gae.tolist(), "deltas": deltas.tolist()}))


if __name__ == "__main__":
    main()
