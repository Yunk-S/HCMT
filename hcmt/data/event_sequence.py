"""Sparse perioperative event sequences.

One admission is represented by irregular ``(token, time)`` pairs.  Static
baseline attributes are supplied separately and never become prediction
targets.  Medication administrations and procedural metadata are context
tokens; clinically meaningful state transitions are outcome tokens.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


PAD_TOKEN_ID = 0
TOKEN_KINDS = {
    "padding": 0,
    "boundary": 1,
    "static_observation": 2,
    "procedure_context": 3,
    "medication_context": 4,
    "diagnosis_context": 5,
    "clinical_outcome": 6,
    "clock": 7,
}


class EventSequenceDataset(Dataset):
    """Windowed, all-admission event sequence dataset.

    Every admission is represented at least once.  Long admissions use
    overlapping windows.  Targets are the complete set of clinical outcomes
    at the next strictly later outcome time, plus the elapsed time to it.
    Context-only tokens (for example a drug administration) therefore inform
    the next physiological outcome without being treated as an outcome.
    """

    def __init__(self, path: str | Path, block_size: int = 256,
                 window_stride: int | None = None,
                 dynamic_windows: bool = False, seed: int = 42,
                 minimum_context_fraction: float = 0.75,
                 outcome_limit: int = 0):
        self.path = Path(path)
        self.meta = json.loads((self.path / "event_sequence_meta.json").read_text())
        if self.meta.get("version") not in (2, 3, 4, 5) or not self.meta.get("complete"):
            raise ValueError("Event sequence dataset is incomplete or incompatible")
        self.split = self.meta.get("split")
        if self.split not in {"all_train", "validation", "external_validation", "test"}:
            raise ValueError(f"Unsupported event-sequence split: {self.split!r}")

        self.block_size = int(block_size)
        if self.block_size < 8:
            raise ValueError("block_size must be at least 8")
        self.window_stride = int(window_stride or max(1, self.block_size // 2))
        self.dynamic_windows = bool(dynamic_windows)
        self.seed = int(seed)
        self.minimum_context_fraction = float(minimum_context_fraction)
        if not 0.5 <= self.minimum_context_fraction <= 1.0:
            raise ValueError("minimum_context_fraction must be between 0.5 and 1")
        base_outcomes = list(self.meta["outcome_vocabulary"])
        stored_outcomes = self.meta.get("stored_outcome_vocabulary", base_outcomes)
        remap = self.meta.get("outcome_class_remap", list(range(len(stored_outcomes))))
        base_remap = np.asarray(remap, dtype=np.int16)
        base_family_ids = list(self.meta.get("outcome_to_family", []))
        if outcome_limit and outcome_limit < len(base_outcomes):
            counts = self.meta.get("audit_counts", {})
            ranked = sorted(range(len(base_outcomes)), key=lambda index: (
                -int(counts.get(base_outcomes[index], 0)), index))[:int(outcome_limit)]
            selected = set(ranked)
            kept_indices = [index for index in range(len(base_outcomes)) if index in selected]
            base_to_new = np.full(len(base_outcomes), -1, dtype=np.int16)
            base_to_new[kept_indices] = np.arange(len(kept_indices), dtype=np.int16)
            valid = base_remap >= 0
            limited_remap = np.full_like(base_remap, -1)
            limited_remap[valid] = base_to_new[base_remap[valid]]
            base_remap = limited_remap
            base_outcomes = [base_outcomes[index] for index in kept_indices]
            if base_family_ids:
                old_families = [base_family_ids[index] for index in kept_indices]
                family_order = list(dict.fromkeys(old_families))
                family_reindex = {family: index for index, family in enumerate(family_order)}
                base_family_ids = [family_reindex[family] for family in old_families]
            self.meta = dict(self.meta)
            self.meta["outcome_vocabulary"] = base_outcomes
            self.meta["outcome_to_family"] = base_family_ids
            self.meta["runtime_outcome_limit"] = int(outcome_limit)
        self.num_outcomes = len(base_outcomes)
        self.outcome_remap = base_remap
        self.num_tokens = len(self.meta["token_vocabulary"])
        self.num_static = int(self.meta["num_static"])
        self.outcome_family_ids = np.asarray(
            base_family_ids, dtype=np.int16)
        self.num_event_families = (int(self.outcome_family_ids.max()) + 1
                                   if len(self.outcome_family_ids) else 0)
        self.trajectory_horizons_hours = tuple(
            float(value) for value in self.meta.get("trajectory_horizons_hours", (1, 6, 24)))

        self.ptr = np.load(self.path / "sequence_ptr.npy", mmap_mode="r")
        count = int(self.ptr[-1])
        self.token_id = np.memmap(self.path / "token_id.bin", dtype="int32", mode="r", shape=(count,))
        self.time_min = np.memmap(self.path / "time_min.bin", dtype="float32", mode="r", shape=(count,))
        self.value = np.memmap(self.path / "value.bin", dtype="float32", mode="r", shape=(count,))
        self.has_value = np.memmap(self.path / "has_value.bin", dtype="uint8", mode="r", shape=(count,))
        self.kind = np.memmap(self.path / "token_kind.bin", dtype="uint8", mode="r", shape=(count,))
        self.outcome = np.memmap(self.path / "outcome_class.bin", dtype="int16", mode="r", shape=(count,))
        self.static = np.load(self.path / "static_baseline.npy", mmap_mode="r")
        token_vocabulary = self.meta["token_vocabulary"]
        self.phase_transition_ids = {
            token_vocabulary.get("event:or_entry"): 1,
            token_vocabulary.get("event:anesthesia_start"): 2,
            token_vocabulary.get("event:surgery_start"): 3,
            token_vocabulary.get("event:surgery_end"): 4,
            token_vocabulary.get("event:or_exit"): 5,
            token_vocabulary.get("event:icu_transfer"): 6,
            token_vocabulary.get("event:icu_discharge"): 7,
        }
        self.phase_transition_ids.pop(None, None)

        admissions: List[int] = []
        starts: List[int] = []
        for ai in range(len(self.ptr) - 1):
            length = int(self.ptr[ai + 1] - self.ptr[ai])
            if length <= self.block_size:
                admissions.append(ai)
                starts.append(0)
                continue
            local = list(range(0, max(1, length - self.block_size + 1), self.window_stride))
            final = length - self.block_size
            if not local or local[-1] != final:
                local.append(final)
            admissions.extend([ai] * len(local))
            starts.extend(local)
        self.window_admission = np.asarray(admissions, dtype=np.int32)
        self.base_window_start = np.asarray(starts, dtype=np.int32)
        self.window_start = self.base_window_start.copy()
        self.window_length = np.asarray([
            min(self.block_size,
                int(self.ptr[ai + 1] - self.ptr[ai]) - int(start))
            for ai, start in zip(self.window_admission, self.window_start)
        ], dtype=np.int32)

    def set_epoch(self, epoch: int) -> None:
        """Move training crops deterministically to avoid repeating one context."""
        if not self.dynamic_windows:
            return
        rng = np.random.default_rng(self.seed + int(epoch) * 1_000_003)
        starts = self.base_window_start.copy()
        lengths = np.empty_like(starts)
        minimum = max(8, int(round(self.block_size * self.minimum_context_fraction)))
        for index, (ai, base) in enumerate(zip(self.window_admission, starts)):
            admission_length = int(self.ptr[int(ai) + 1] - self.ptr[int(ai)])
            max_start = max(0, admission_length - minimum)
            jitter = int(rng.integers(-self.window_stride // 2,
                                     self.window_stride // 2 + 1))
            starts[index] = min(max(0, int(base) + jitter), max_start)
            available = admission_length - int(starts[index])
            upper = min(self.block_size, available)
            lower = min(upper, minimum)
            lengths[index] = int(rng.integers(lower, upper + 1)) if upper > lower else upper
        self.window_start = starts
        self.window_length = lengths

    def __len__(self) -> int:
        return len(self.window_admission)

    def sequence_length(self, index: int) -> int:
        """Return a window length without materialising the sample.

        This is intentionally O(1): the training batch sampler calls it while
        arranging length-similar windows to avoid quadratic attention work on
        padding.
        """
        ai = int(self.window_admission[index])
        admission_length = int(self.ptr[ai + 1] - self.ptr[ai])
        return min(int(self.window_length[index]),
                   admission_length - int(self.window_start[index]))

    def _targets(self, times: np.ndarray, outcomes: np.ndarray,
                 local_start: int = 0, local_end: int | None = None):
        """Build local-window targets while looking beyond the crop safely."""
        n = len(times)
        local_end = n if local_end is None else int(local_end)
        local_start = int(local_start)
        local_length = local_end - local_start
        target_set = np.zeros((local_length, self.num_outcomes), dtype=np.float32)
        target_dt = np.zeros(local_length, dtype=np.float32)
        loss_mask = np.zeros(local_length, dtype=np.bool_)
        time_mask = np.zeros(local_length, dtype=np.bool_)

        groups = []
        start = 0
        while start < n:
            end = start + 1
            while end < n and times[end] == times[start]:
                end += 1
            groups.append((start, end))
            start = end

        next_time = None
        next_classes: np.ndarray | None = None
        for start, end in reversed(groups):
            pos = end - 1
            if (local_start <= pos < local_end and next_time is not None and
                    next_classes is not None):
                local_pos = pos - local_start
                target_set[local_pos, next_classes] = 1.0
                target_dt[local_pos] = max(
                    float(next_time - times[pos]) / 60.0, 1.0 / 60.0)
                loss_mask[local_pos] = True
                time_mask[local_pos] = True
            present = np.unique(outcomes[start:end])
            present = present[present >= 0]
            if len(present):
                present = self.outcome_remap[present]
                present = present[present >= 0]
            if len(present):
                next_time = float(times[start])
                next_classes = present.astype(np.int64, copy=False)

        if len(groups) >= 2:
            final_start, _ = groups[-1]
            _, previous_end = groups[-2]
            pos = previous_end - 1
            if local_start <= pos < local_end:
                local_pos = pos - local_start
                if not loss_mask[local_pos] and times[final_start] > times[pos]:
                    target_dt[local_pos] = max(
                        float(times[final_start] - times[pos]) / 60.0, 1.0 / 60.0)
                    time_mask[local_pos] = True

        horizon_count = len(self.trajectory_horizons_hours)
        trajectory_target = np.zeros(
            (local_length, horizon_count, self.num_outcomes), dtype=np.float32)
        trajectory_mask = np.zeros((local_length, horizon_count), dtype=np.bool_)
        episode_end = float(times[-1]) if n else 0.0
        outcome_positions = np.flatnonzero(outcomes >= 0)
        mapped_outcomes = self.outcome_remap[outcomes[outcome_positions]]
        keep = mapped_outcomes >= 0
        outcome_times = times[outcome_positions][keep]
        mapped_outcomes = mapped_outcomes[keep]
        for start, end in groups:
            pos = end - 1
            if not local_start <= pos < local_end:
                continue
            local_pos = pos - local_start
            current_time = float(times[start])
            left = int(np.searchsorted(outcome_times, current_time, side="right"))
            for horizon_index, horizon_hours in enumerate(self.trajectory_horizons_hours):
                cutoff = current_time + horizon_hours * 60.0
                if cutoff <= episode_end:
                    right = int(np.searchsorted(outcome_times, cutoff, side="right"))
                    if right > left:
                        classes = np.unique(mapped_outcomes[left:right])
                        trajectory_target[local_pos, horizon_index, classes] = 1.0
                    trajectory_mask[local_pos, horizon_index] = True
        return (target_set, target_dt, loss_mask, time_mask,
                trajectory_target, trajectory_mask)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ai = int(self.window_admission[index])
        local_start = int(self.window_start[index])
        lo, hi = int(self.ptr[ai]), int(self.ptr[ai + 1])

        # Targets are derived from the whole admission so the final token in a
        # window may still learn from an outcome immediately beyond the crop.
        all_times = np.asarray(self.time_min[lo:hi], dtype=np.float32)
        all_outcomes = np.asarray(self.outcome[lo:hi], dtype=np.int16)
        all_tokens = np.asarray(self.token_id[lo:hi], dtype=np.int32)
        local_end = min(local_start + int(self.window_length[index]), len(all_times))
        (target_set, target_dt, loss_mask, time_mask,
         trajectory_target, trajectory_mask) = self._targets(
            all_times, all_outcomes, local_start, local_end)

        start = lo + local_start
        end = min(start + int(self.window_length[index]), hi)
        sl = slice(start, end)
        times = np.array(self.time_min[sl], dtype=np.float32, copy=True)
        gaps = np.diff(times, prepend=times[0]).astype(np.float32, copy=False)
        phase = 0
        phase_ids = np.zeros(local_end - local_start, dtype=np.int64)
        for position, token in enumerate(all_tokens[:local_end]):
            phase = self.phase_transition_ids.get(int(token), phase)
            if position >= local_start:
                phase_ids[position - local_start] = phase
        history_family_counts = np.zeros(self.num_event_families, dtype=np.float32)
        if self.num_event_families and local_start:
            previous_outcomes = all_outcomes[:local_start]
            previous_outcomes = previous_outcomes[previous_outcomes >= 0]
            if len(previous_outcomes):
                mapped = self.outcome_remap[previous_outcomes]
                mapped = mapped[mapped >= 0]
                if len(mapped):
                    families = self.outcome_family_ids[mapped]
                    history_family_counts = np.log1p(np.bincount(
                        families, minlength=self.num_event_families)).astype(np.float32)
        observation_features = np.zeros((local_end - local_start, 3), dtype=np.float32)
        last_family_time: Dict[int, float] = {}
        left = 0
        for position in range(local_end):
            while all_times[position] - all_times[left] > 360.0:
                left += 1
            if position >= local_start:
                local_position = position - local_start
                observation_features[local_position, 0] = math.log1p(
                    max(0.0, float(all_times[position] - all_times[left]))) / math.log1p(360.0)
                observation_features[local_position, 1] = math.log1p(position - left) / math.log1p(128.0)
            outcome = int(all_outcomes[position])
            if outcome >= 0 and self.num_event_families:
                mapped = int(self.outcome_remap[outcome])
                if mapped >= 0:
                    family = int(self.outcome_family_ids[mapped])
                    previous = last_family_time.get(family)
                    if position >= local_start and previous is not None:
                        local_position = position - local_start
                        observation_features[local_position, 2] = (
                            math.log1p(max(0.0, float(all_times[position] - previous))) /
                            math.log1p(43200.0))
                    last_family_time[family] = float(all_times[position])
        return {
            "token_id": torch.from_numpy(np.array(self.token_id[sl], dtype=np.int64, copy=True)),
            "time_min": torch.from_numpy(times),
            "gap_min": torch.from_numpy(gaps),
            "value": torch.from_numpy(np.array(self.value[sl], dtype=np.float32, copy=True)),
            "has_value": torch.from_numpy(np.array(self.has_value[sl], dtype=np.float32, copy=True)),
            "token_kind": torch.from_numpy(np.array(self.kind[sl], dtype=np.int64, copy=True)),
            "phase_id": torch.from_numpy(phase_ids),
            "history_family_counts": torch.from_numpy(history_family_counts),
            "observation_features": torch.from_numpy(observation_features),
            "static": torch.from_numpy(np.array(self.static[ai], dtype=np.float32, copy=True)),
            "target_set": torch.from_numpy(target_set),
            "target_dt_hours": torch.from_numpy(target_dt),
            "loss_mask": torch.from_numpy(loss_mask),
            "time_mask": torch.from_numpy(time_mask),
            "trajectory_target": torch.from_numpy(trajectory_target),
            "trajectory_mask": torch.from_numpy(trajectory_mask),
            "admission_index": torch.tensor(ai, dtype=torch.long),
        }


def collate_event_sequences(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for key in ("token_id", "time_min", "gap_min", "value", "has_value", "token_kind", "phase_id"):
        result[key] = pad_sequence([sample[key] for sample in samples], batch_first=True)
    result["target_set"] = pad_sequence(
        [sample["target_set"] for sample in samples], batch_first=True)
    result["target_dt_hours"] = pad_sequence(
        [sample["target_dt_hours"] for sample in samples], batch_first=True)
    result["loss_mask"] = pad_sequence(
        [sample["loss_mask"] for sample in samples], batch_first=True)
    result["time_mask"] = pad_sequence(
        [sample["time_mask"] for sample in samples], batch_first=True)
    result["trajectory_target"] = pad_sequence(
        [sample["trajectory_target"] for sample in samples], batch_first=True)
    result["trajectory_mask"] = pad_sequence(
        [sample["trajectory_mask"] for sample in samples], batch_first=True)
    lengths = torch.tensor([len(sample["token_id"]) for sample in samples], dtype=torch.long)
    result["attention_mask"] = (
        torch.arange(result["token_id"].shape[1])[None, :] < lengths[:, None]
    )
    result["static"] = torch.stack([sample["static"] for sample in samples])
    result["history_family_counts"] = torch.stack(
        [sample["history_family_counts"] for sample in samples])
    result["observation_features"] = pad_sequence(
        [sample["observation_features"] for sample in samples], batch_first=True)
    result["admission_index"] = torch.stack([sample["admission_index"] for sample in samples])
    return result
