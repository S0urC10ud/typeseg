"""Security decoding: confidence, nearest visible whitespace, then short runs.

Unlike the benchmark's reduced routing labels, the public API retains all 35
canonical types. The policy is applied consistently across release runtimes.
"""
from __future__ import annotations

import numpy as np


def visual_whitespace(text: str, labels: np.ndarray) -> np.ndarray:
    original = np.asarray(labels)
    result = original.copy()
    n = len(text)
    if not n:
        return result
    # UTF-32 keeps one element per Python character, including lone surrogates.
    chars = np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype="<u4")
    visible = (chars != 32) & (chars != 9) & (chars != 13) & (chars != 10)
    if visible.all() or not visible.any():
        return result
    positions = np.arange(n)
    left = np.maximum.accumulate(np.where(visible, positions, -1))
    right = np.minimum.accumulate(np.where(visible, positions, n)[::-1])[::-1]
    newline = chars == 10
    lines = np.cumsum(newline) - newline  # A newline belongs to its preceding line.
    same_left = (left >= 0) & (lines[np.maximum(left, 0)] == lines)
    same_right = (right < n) & (lines[np.minimum(right, n - 1)] == lines)
    local = same_left | same_right
    left = np.where(local & ~same_left, -1, left)
    right = np.where(local & ~same_right, n, right)
    prefer_left = (left >= 0) & ((right == n) | (positions - left <= right - positions))
    sources = np.where(prefer_left, left, right)
    take = ~visible & (sources >= 0) & (sources < n)
    result[take] = original[sources[take]]
    return result


def merge_short_runs(labels: np.ndarray, minimum: int) -> np.ndarray:
    """Exact leftmost-greedy research rule, with O(n) linked-run bookkeeping.

    Rescanning every character after removing each short run was quadratic on
    highly fragmented malware. Only adjacent runs change after a merge.
    """
    result = np.asarray(labels).copy()
    if minimum <= 1 or len(result) < 3:
        return result
    starts = np.r_[0,np.flatnonzero(result[1:] != result[:-1])+1].tolist()
    ends = starts[1:] + [len(result)]
    values = result[starts].tolist()
    count = len(starts)
    previous = list(range(-1,count-1))
    following = list(range(1,count)) + [-1]
    i = 1 if count > 1 else -1
    while i >= 0:
        left,right = previous[i],following[i]
        if left < 0 or right < 0 or ends[i]-starts[i] >= minimum:
            i = right
            continue
        if values[left] == values[right]:
            ends[left] = ends[right]
            following[left] = following[right]
            if following[right] >= 0: previous[following[right]] = left
            i = left
        elif ends[left]-starts[left] >= ends[right]-starts[right]:
            ends[left] = ends[i]
            following[left] = right;previous[right] = left
            i = left
        else:
            starts[right] = starts[i]
            following[left] = right;previous[right] = left
            i = right
    i = 0
    while i >= 0:
        result[starts[i]:ends[i]] = values[i]
        i = following[i]
    return result


def decode(text: str, probabilities: np.ndarray, options, other_index: int) -> np.ndarray:
    labels = probabilities.argmax(axis=1).astype(np.int64)
    if options.confidence_gating:
        labels[probabilities.max(axis=1) < options.other_threshold] = other_index
    if options.whitespace_relabel:
        labels = visual_whitespace(text, labels)
    if options.min_run_normalize:
        labels = merge_short_runs(labels, options.min_run_chars)
    return labels
