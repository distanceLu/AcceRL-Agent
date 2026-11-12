"""Pure CPU packing helpers for fixed-sample, variable-microbatch training."""

from __future__ import annotations


def _require_integer(value: int, name: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _validate_lengths(lengths: list[int]) -> None:
    for index, length in enumerate(lengths):
        _require_integer(length, f"lengths[{index}]")


def pack_sequence_indices(
    lengths: list[int],
    token_budget: int,
    max_sequences: int = 0,
) -> list[list[int]]:
    """Pack every sample exactly once using stable first-fit decreasing.

    Returned indices refer to ``lengths``. Each pack contains at most
    ``token_budget`` real tokens and, when positive, ``max_sequences`` samples.
    A zero sequence cap leaves only the token constraint. Empty input returns
    no packs; a single sample larger than the budget is rejected.
    """
    _require_integer(token_budget, "token_budget")
    _require_integer(max_sequences, "max_sequences", allow_zero=True)
    _validate_lengths(lengths)
    for index, length in enumerate(lengths):
        if length > token_budget:
            raise ValueError(
                f"Sample {index} has length {length}, exceeding token_budget {token_budget}"
            )

    packs: list[list[int]] = []
    token_counts: list[int] = []
    ordered = sorted(range(len(lengths)), key=lambda index: lengths[index], reverse=True)
    for index in ordered:
        length = lengths[index]
        for pack_index, pack in enumerate(packs):
            if max_sequences and len(pack) >= max_sequences:
                continue
            if token_counts[pack_index] + length <= token_budget:
                pack.append(index)
                token_counts[pack_index] += length
                break
        else:
            packs.append([index])
            token_counts.append(length)
    return packs


def split_packs_to_count(
    packs: list[list[int]],
    target_count: int,
    lengths: list[int],
) -> list[list[int]]:
    """Increase the pack count without changing the referenced samples.

    The largest multi-sample pack is split into two token-balanced subsets
    until the target is reached. Each resulting pack is a subset of an input
    pack, so splitting preserves its token and sequence-count upper bounds.
    Inputs are not modified. Indices must be valid and unique, and all input
    packs must be nonempty. ``target_count`` cannot be below the current pack
    count or above the number of samples referenced by ``packs``.
    """
    _require_integer(target_count, "target_count", allow_zero=True)
    _validate_lengths(lengths)
    result: list[list[int]] = []
    seen: set[int] = set()
    for pack_index, pack in enumerate(packs):
        if not pack:
            raise ValueError(f"Pack {pack_index} must be nonempty")
        copied = []
        for index in pack:
            _require_integer(index, "sample index", allow_zero=True)
            if index >= len(lengths):
                raise ValueError(f"Sample index {index} is outside lengths")
            if index in seen:
                raise ValueError(f"Sample index {index} appears more than once")
            seen.add(index)
            copied.append(index)
        result.append(copied)

    if target_count < len(result):
        raise ValueError(
            f"target_count {target_count} is below current pack count {len(result)}"
        )
    if target_count > len(seen):
        raise ValueError(
            f"target_count {target_count} exceeds referenced sample count {len(seen)}"
        )

    while len(result) < target_count:
        candidates = (index for index, pack in enumerate(result) if len(pack) > 1)
        pack_index = max(
            candidates,
            key=lambda index: sum(lengths[sample] for sample in result[index]),
        )
        halves: list[list[int]] = [[], []]
        token_counts = [0, 0]
        ordered = sorted(result[pack_index], key=lambda index: lengths[index], reverse=True)
        for index in ordered:
            side = 0 if token_counts[0] <= token_counts[1] else 1
            halves[side].append(index)
            token_counts[side] += lengths[index]
        result[pack_index] = halves[0]
        result.append(halves[1])
    return result
