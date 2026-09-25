"""spdee.backends.fd2.profiling — cumulative phase timer.

Enabled when the environment variable SPDEE_PROFILE is set to a
non-empty, non-zero value at import time, or by calling `enable()`.
Timings accumulate per label (total seconds and entry count) in a
process-wide table; every recording call is a no-op while disabled.

Instrumented code uses either the context manager

    with timed("solve_bvp/origin_regularity"):
        ...

or the explicit pair

    t0 = perf_now()
    ...
    record("solve_bvp/sparse_solve", perf_now() - t0)

Drivers call `reset()` before a phase and `snapshot()` or
`print_report()` afterwards. The table is not thread-safe.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Optional, Tuple


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    return v not in ("", "0", "false", "False", "no", "No")


_enabled: bool = _env_truthy("SPDEE_PROFILE")
_totals: Dict[str, float] = defaultdict(float)
_counts: Dict[str, int] = defaultdict(int)


def enable() -> None:
    """Turn on profiling from this point."""
    global _enabled
    _enabled = True


def is_enabled() -> bool:
    return _enabled


def reset() -> None:
    """Clear all accumulated timings."""
    _totals.clear()
    _counts.clear()


def record(label: str, seconds: float) -> None:
    """Add an elapsed time to a labeled bucket; no-op when disabled."""
    if not _enabled:
        return
    _totals[label] += float(seconds)
    _counts[label] += 1


def perf_now() -> float:
    """Monotonic wall-clock seconds (time.perf_counter)."""
    return time.perf_counter()


@contextmanager
def timed(label: str):
    """Context manager that records the wall time of its block."""
    if not _enabled:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _totals[label] += dt
        _counts[label] += 1


def snapshot() -> Dict[str, Tuple[float, int]]:
    """Return {label: (total_seconds, count)} for every recorded bucket."""
    return {k: (_totals[k], _counts[k]) for k in _totals}


def print_report(label: Optional[str] = None, top_n: Optional[int] = None) -> None:
    """Print the recorded buckets sorted by total time, largest first."""
    if not _totals:
        print(f"[profiling] no data recorded (enabled={_enabled})")
        return
    header = "[profiling]" + (f" {label}" if label else "")
    print(header)
    items = sorted(_totals.items(), key=lambda kv: -kv[1])
    if top_n is not None:
        items = items[:top_n]
    total = sum(_totals.values())
    name_w = max(8, max(len(k) for k in _totals))
    print(f"  {'phase':<{name_w}}  {'total (s)':>10}  {'count':>6}  {'avg (s)':>10}  "
          f"{'% of total':>10}")
    for k, t in items:
        c = _counts[k]
        avg = t / c if c else 0.0
        pct = 100.0 * t / total if total > 0 else 0.0
        print(f"  {k:<{name_w}}  {t:>10.3f}  {c:>6d}  {avg:>10.4f}  {pct:>9.1f}%")
    print(f"  {'(sum)':<{name_w}}  {total:>10.3f}")
