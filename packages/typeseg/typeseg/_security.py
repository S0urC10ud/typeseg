"""Security decoding: confidence, nearest visible whitespace, then short runs.

Unlike the benchmark's reduced routing labels, the public API retains all 35
canonical types. The policy is applied consistently across release runtimes.
"""
from __future__ import annotations

import numpy as np


def visual_whitespace(text: str, labels: np.ndarray) -> np.ndarray:
    visual = set(" \t\r\n")
    result = np.asarray(labels).copy()
    n = len(text)
    same_left = np.full(n, -1, dtype=int)
    same_right = np.full(n, -1, dtype=int)
    starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n" and i + 1 < n]
    for start, end in zip(starts, starts[1:] + [n]):
        last = -1
        for i in range(start, end):
            if text[i] not in visual:
                last = i
            same_left[i] = last
        last = -1
        for i in range(end - 1, start - 1, -1):
            if text[i] not in visual:
                last = i
            same_right[i] = last
    left, right = np.full(n, -1, dtype=int), np.full(n, -1, dtype=int)
    last = -1
    for i, ch in enumerate(text):
        if ch not in visual:
            last = i
        left[i] = last
    last = -1
    for i in range(n - 1, -1, -1):
        if text[i] not in visual:
            last = i
        right[i] = last
    for i, ch in enumerate(text):
        if ch in visual:
            candidates = [v for v in (same_left[i], same_right[i]) if v >= 0]
            if not candidates:
                candidates = [v for v in (left[i], right[i]) if v >= 0]
            if candidates:
                source = min(candidates, key=lambda v: (abs(i - v), v > i))
                result[i] = labels[source]
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
