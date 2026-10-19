"""Real CPU attention/cache operations for controlled, teacher-forced audits.

This is an untrained single-head model, not a benchmark of LLM task quality.
A snapshot and its KV must be kept together to reproduce a frozen policy.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn

from test.partial_rollout import Action


@dataclass(frozen=True)
class Cache:
    keys: torch.Tensor
    values: torch.Tensor
    versions: tuple[int, ...]

    @property
    def length(self) -> int:
        return self.keys.shape[0]


class TinyAttentionLM(nn.Module):
    def __init__(self, seed: int = 0, version: int = 0):
        super().__init__()
        self.version = version
        self.width = 16
        self.vocab_size = 16
        self.max_context = 256
        # Keep test/demo construction independent of the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(self.vocab_size, self.width)
            self.position = nn.Embedding(self.max_context, self.width)
            self.query = nn.Linear(self.width, self.width, bias=False)
            self.key = nn.Linear(self.width, self.width, bias=False)
            self.value = nn.Linear(self.width, self.width, bias=False)
            self.output = nn.Linear(self.width, self.width, bias=False)
            self.head = nn.Linear(self.width, self.vocab_size, bias=False)
        self.double().eval()
        self.requires_grad_(False)

    def changed(self, version: int, magnitude: float, seed: int) -> TinyAttentionLM:
        if version < self.version or magnitude < 0 or not math.isfinite(magnitude):
            raise ValueError("invalid version or update magnitude")
        result = deepcopy(self)
        result.version = version
        rng = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for parameter in result.parameters():
                noise = torch.randn(parameter.shape, generator=rng, dtype=parameter.dtype)
                parameter.add_(noise, alpha=magnitude)
        return result

    def empty_cache(self) -> Cache:
        empty = self.embedding.weight.new_empty((0, self.width))
        return Cache(empty, empty.clone(), ())

    @torch.no_grad()
    def prefill(self, tokens: list[int]) -> tuple[torch.Tensor, Cache]:
        if not tokens or len(tokens) > self.max_context:
            raise ValueError("prefill length must be within model context")
        ids = torch.tensor(tokens, dtype=torch.long)
        x = self.embedding(ids) + self.position(torch.arange(len(tokens)))
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.T / math.sqrt(self.width)
        causal = torch.ones_like(scores, dtype=torch.bool).triu(1)
        probabilities = scores.masked_fill(causal, -torch.inf).softmax(-1)
        logits = self.head(x + self.output(probabilities @ v))
        return logits[-1], Cache(k, v, (self.version,) * len(tokens))

    @torch.no_grad()
    def step(self, token: int, cache: Cache) -> tuple[torch.Tensor, Cache]:
        if cache.length >= self.max_context:
            raise ValueError("context exhausted")
        x = self.embedding.weight[token] + self.position.weight[cache.length]
        q = self.query(x)
        keys = torch.cat((cache.keys, self.key(x).unsqueeze(0)))
        values = torch.cat((cache.values, self.value(x).unsqueeze(0)))
        probability = (q @ keys.T / math.sqrt(self.width)).softmax(-1)
        logits = self.head(x + self.output(probability @ values))
        return logits, Cache(keys, values, cache.versions + (self.version,))


def sample_prefix(model: TinyAttentionLM, length: int, seed: int) -> list[int]:
    if not 4 <= length <= model.max_context:
        raise ValueError("prefix length must be between 4 and max_context")
    tokens = [1, 2, 3, 4]
    logits, cache = model.prefill(tokens)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    while len(tokens) < length:
        token = int(torch.multinomial(logits.softmax(-1), 1, generator=rng))
        tokens.append(token)
        logits, cache = model.step(token, cache)
    return tokens


def conditional_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    reference = reference_logits.log_softmax(-1)
    candidate = candidate_logits.log_softmax(-1)
    return max(0.0, float((reference.exp() * (reference - candidate)).sum()))


@dataclass(frozen=True)
class Audit:
    risks_by_token: dict[Action, tuple[float, ...]]
    behavior_logprobs: dict[Action, tuple[float, ...]]
    continuation_tokens: tuple[int, ...]
    final_cache_versions: dict[Action, tuple[int, ...]]

    def mean_risk(self, action: Action) -> float:
        values = self.risks_by_token[action]
        return sum(values) / len(values)


@torch.no_grad()
def audit_continuations(
    old: TinyAttentionLM,
    new: TinyAttentionLM,
    prefix: list[int],
    steps: int,
    seed: int,
) -> Audit:
    """Audit A/B/C on identical prefixes and reference-sampled continuations.

    The cache contains all tokens BEFORE the pending final prefix token. This
    mirrors decoding at an update boundary: the next forward consumes that
    pending token. B therefore uses new weights for that forward and old KV
    for preceding tokens. C rebuilds the entire prefix with new weights.

    Full KLs are measured under the fresh model's conditional distribution on
    a common teacher-forced path. These are not independent on-policy rollouts,
    trajectory-distribution KLs, rewards, or a convergence evaluation.
    """
    if len(prefix) < 2 or steps < 1 or len(prefix) + steps > new.max_context:
        raise ValueError("invalid audit prefix/horizon")
    _, prefix_cache = old.prefill(prefix[:-1])
    old_logits, old_cache = old.step(prefix[-1], prefix_cache)
    mixed_logits, mixed_cache = new.step(prefix[-1], prefix_cache)
    fresh_logits, fresh_cache = new.prefill(prefix)
    logits = {Action.OLD: old_logits, Action.KEEP: mixed_logits, Action.REBUILD: fresh_logits}
    caches = {Action.OLD: old_cache, Action.KEEP: mixed_cache, Action.REBUILD: fresh_cache}
    risks = {action: [] for action in Action}
    logprobs = {action: [] for action in Action}
    continuation = []
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        reference = logits[Action.REBUILD]
        token = int(torch.multinomial(reference.softmax(-1), 1, generator=rng))
        continuation.append(token)
        for action in Action:
            risks[action].append(conditional_kl(reference, logits[action]))
            logprobs[action].append(float(logits[action].log_softmax(-1)[token]))
            model = old if action == Action.OLD else new
            logits[action], caches[action] = model.step(token, caches[action])
    return Audit(
        {action: tuple(values) for action, values in risks.items()},
        {action: tuple(values) for action, values in logprobs.items()},
        tuple(continuation),
        {action: cache.versions for action, cache in caches.items()},
    )
