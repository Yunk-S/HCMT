"""Auditable rules for converting measurements into clinical transitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Transition:
    time_min: float
    name: str
    value: float
    severity: float = 0.0


def merge_series(series: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Merge priority-ordered series, retaining the first source at tied times."""
    parts = [(np.asarray(t, dtype=np.float32), np.asarray(v, dtype=np.float32), rank)
             for rank, (t, v) in enumerate(series) if len(t)]
    if not parts:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    times = np.concatenate([item[0] for item in parts])
    values = np.concatenate([item[1] for item in parts])
    priority = np.concatenate([np.full(len(item[0]), item[2], dtype=np.int16) for item in parts])
    order = np.lexsort((priority, times))
    times, values = times[order], values[order]
    keep = np.r_[True, times[1:] != times[:-1]]
    return times[keep], values[keep]


def state_transitions(times: np.ndarray, values: np.ndarray,
                      abnormal: Callable[[float], bool],
                      severe: Callable[[float], bool],
                      recovered: Callable[[float], bool],
                      onset_name: str, recovery_name: str | None,
                      severity: Callable[[float], float],
                      max_confirmation_gap: float = 10.0) -> List[Transition]:
    """Detect confirmed onset/recovery transitions.

    A severe single observation is accepted immediately.  Otherwise two
    abnormal observations within ``max_confirmation_gap`` minutes are needed.
    Recoveries also need two confirming observations.  This deliberately
    reduces isolated monitor-artifact tokens.
    """
    times = np.asarray(times, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if len(times) != len(values):
        raise ValueError("times and values must have equal length")
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    result: List[Transition] = []
    active = False
    pending_abnormal: Tuple[float, float] | None = None
    pending_recovery: Tuple[float, float] | None = None
    for time, value in zip(times, values):
        if not np.isfinite(value) or not np.isfinite(time):
            continue
        t, v = float(time), float(value)
        if abnormal(v):
            pending_recovery = None
            if active:
                continue
            confirmed = severe(v)
            if pending_abnormal is not None and t - pending_abnormal[0] <= max_confirmation_gap:
                confirmed = True
            if confirmed:
                result.append(Transition(t, onset_name, v, max(0.0, float(severity(v)))))
                active = True
                pending_abnormal = None
            else:
                pending_abnormal = (t, v)
        else:
            pending_abnormal = None
            if not active or recovery_name is None or not recovered(v):
                pending_recovery = None
                continue
            if pending_recovery is not None and t - pending_recovery[0] <= max_confirmation_gap:
                result.append(Transition(t, recovery_name, v, 0.0))
                active = False
                pending_recovery = None
            else:
                pending_recovery = (t, v)
    return result


def binary_transitions(times: np.ndarray, values: np.ndarray, start_name: str,
                       stop_name: str, inactivity_gap: float = 30.0) -> List[Transition]:
    """Extract starts/stops from a binary or non-negative treatment signal."""
    order = np.argsort(times, kind="stable")
    times = np.asarray(times, dtype=np.float32)[order]
    values = np.asarray(values, dtype=np.float32)[order]
    result: List[Transition] = []
    active = False
    last_time = None
    for time, value in zip(times, values):
        if not np.isfinite(time) or not np.isfinite(value):
            continue
        t, v = float(time), float(value)
        if active and last_time is not None and t - last_time > inactivity_gap:
            result.append(Transition(float(last_time), stop_name, 0.0, 0.0))
            active = False
        now = v > 0
        if now and not active:
            result.append(Transition(t, start_name, v, 1.0))
            active = True
        elif not now and active:
            result.append(Transition(t, stop_name, v, 0.0))
            active = False
        last_time = t
    return result


def observed_vasopressor_response(start_min: float, bp_times: np.ndarray,
                                  bp_abnormality: np.ndarray,
                                  lookback_min: float = 15.0,
                                  horizon_min: float = 30.0):
    """Label observed BP evolution around a vasopressor start.

    Returns ``(status, time)`` where status is ``recovered``, ``persistent``,
    or ``unknown``.  The label is descriptive and does not claim a causal drug
    effect.  It is emitted only when hypotension was observed shortly before
    treatment and adequate post-treatment measurements exist.
    """
    times = np.asarray(bp_times, dtype=np.float32)
    abnormality = np.asarray(bp_abnormality, dtype=np.float32)
    before = np.flatnonzero((times <= start_min) & (times >= start_min - lookback_min))
    if not len(before) or abnormality[before[-1]] <= 0:
        return "unknown", None
    after = np.flatnonzero((times > start_min) & (times <= start_min + horizon_min))
    if not len(after):
        return "unknown", None
    normal_run = []
    for idx in after:
        if abnormality[idx] <= 0:
            if normal_run and times[idx] - times[normal_run[-1]] <= 10:
                return "recovered", float(times[idx])
            normal_run = [int(idx)]
        else:
            normal_run = []
    late = [int(idx) for idx in after if times[idx] >= start_min + 20]
    if late and abnormality[late[-1]] > 0:
        return "persistent", float(times[late[-1]])
    return "unknown", None

