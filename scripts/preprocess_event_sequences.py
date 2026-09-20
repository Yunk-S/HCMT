#!/usr/bin/env python3
"""Build sparse perioperative ``event + time`` sequences from train_data.

The script consumes the corrected v6 sparse timeline, which is itself derived
exclusively from ``data/train_data``.  It never reads MIMIC.  Five-minute rows
are used only to detect clinically meaningful state transitions; normal,
unchanged rows are not copied into the output.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcmt.data.clinical_events import (  # noqa: E402
    ALWAYS_KEEP_OUTCOMES,
    FEATURE_NAMES,
    OUTCOME_DEFINITIONS,
    OUTCOMES,
    extract_measurement_events,
)
from hcmt.data.event_extraction import (  # noqa: E402
    Transition, binary_transitions, merge_series, observed_vasopressor_response,
)
from hcmt.data.event_sequence import TOKEN_KINDS  # noqa: E402
from hcmt.data.outcome_families import OUTCOME_TO_FAMILY, family_names  # noqa: E402
from hcmt.data.preprocess_timeline import stream  # noqa: E402


OUTCOME_INDEX = {name: i for i, name in enumerate(OUTCOMES)}

REQUIRED_SOURCE_FILES = (
    "operations.csv.gz", "vitals.csv.gz", "labs.csv.gz",
    "ward_vitals.csv.gz", "medications.csv.gz", "diagnosis.csv.gz",
)

VASOPRESSOR_DRUG_TERMS = (
    "norepinephrine", "noradrenaline", "epinephrine", "adrenaline",
    "phenylephrine", "vasopressin", "ephedrine", "metaraminol",
    "dopamine", "dobutamine",
)

@dataclass(frozen=True)
class Record:
    time_min: float
    token: str
    kind: int
    outcome: int = -1
    value: float = 0.0
    has_value: int = 0


class BinaryWriter:
    def __init__(self, path: Path, dtype: str):
        self.dtype = np.dtype(dtype)
        self.handle = path.open("wb")
        self.count = 0

    def write(self, values: Iterable):
        array = np.asarray(list(values), dtype=self.dtype)
        array.tofile(self.handle)
        self.count += len(array)

    def close(self):
        self.handle.flush()
        self.handle.close()


def _series(feature_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]], *names: str):
    return merge_series([feature_arrays[name] for name in names if name in feature_arrays])


def _abnormality_series(sbp, mbp):
    times = []
    severity = []
    if len(sbp[0]):
        times.append(sbp[0])
        severity.append(np.maximum(0.0, (90.0 - sbp[1]) / 20.0))
    if len(mbp[0]):
        times.append(mbp[0])
        severity.append(np.maximum(0.0, (65.0 - mbp[1]) / 10.0))
    if not times:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    t = np.concatenate(times).astype(np.float32)
    s = np.concatenate(severity).astype(np.float32)
    order = np.argsort(t, kind="stable")
    t, s = t[order], s[order]
    unique, starts = np.unique(t, return_index=True)
    return unique.astype(np.float32), np.maximum.reduceat(s, starts).astype(np.float32)


def _dedupe(records: List[Record]) -> List[Record]:
    unique = {}
    for record in records:
        key = (round(record.time_min, 4), record.token, record.outcome)
        previous = unique.get(key)
        if previous is None or record.value > previous.value:
            unique[key] = record
    return sorted(unique.values(), key=lambda r: (r.time_min, r.kind, r.token))


def _insert_clock_tokens(records: List[Record], interval_min: float,
                         max_per_gap: int) -> List[Record]:
    """Represent long clinically quiet intervals without restoring 5-minute rows."""
    if interval_min <= 0 or max_per_gap <= 0 or len(records) < 2:
        return records
    result: List[Record] = []
    for previous, current in zip(records, records[1:]):
        result.append(previous)
        gap = float(current.time_min - previous.time_min)
        count = min(max_per_gap, max(0, int(math.ceil(gap / interval_min)) - 1))
        for step in range(1, count + 1):
            clock_time = previous.time_min + step * interval_min
            if clock_time >= current.time_min:
                break
            result.append(Record(clock_time, "<CLOCK>", TOKEN_KINDS["clock"]))
    result.append(records[-1])
    return sorted(result, key=lambda r: (r.time_min, r.kind, r.token))


def _annotate_clock_intensity(records: List[Record], measurement_times: np.ndarray,
                              interval_min: float) -> List[Record]:
    """Attach recent raw-observation count to sparse clock tokens."""
    if not len(measurement_times):
        return records
    measurement_times = np.sort(np.asarray(measurement_times, dtype=np.float32))
    annotated = []
    for record in records:
        if record.token != "<CLOCK>":
            annotated.append(record)
            continue
        right = int(np.searchsorted(measurement_times, record.time_min, side="right"))
        left = int(np.searchsorted(
            measurement_times, record.time_min - interval_min, side="left"))
        annotated.append(Record(
            record.time_min, record.token, record.kind, record.outcome,
            float(right - left), 1))
    return annotated


def _insert_phase_summary_tokens(records: List[Record]) -> List[Record]:
    """Add explicit phase-summary context tokens at clinically meaningful boundaries."""
    phases = [(0.0, "phase_summary:preop")]
    seen = {phases[0][1]}
    transition_to_phase = {
        "or_entry": "phase_summary:induction",
        "anesthesia_start": "phase_summary:induction",
        "surgery_start": "phase_summary:surgery",
        "surgery_end": "phase_summary:emergence",
        "anesthesia_end": "phase_summary:emergence",
        "or_exit": "phase_summary:pacu",
        "icu_transfer": "phase_summary:icu",
    }
    for record in records:
        if record.outcome < 0:
            continue
        phase = transition_to_phase.get(record.token.removeprefix("event:"))
        if phase and phase not in seen:
            phases.append((record.time_min, phase))
            seen.add(phase)
    summaries = [Record(time, phase, TOKEN_KINDS["procedure_context"])
                 for time, phase in phases]
    return _dedupe(records + summaries)


def _build_token_vocabulary(old_vocab: Dict[str, int]):
    names = ["<PAD>", "<BOS>", "<EPISODE_END>", "<MASK>", "<CLOCK>",
             "phase_summary:preop", "phase_summary:induction", "phase_summary:surgery",
             "phase_summary:emergence", "phase_summary:pacu", "phase_summary:icu",
             "context:weight_kg", "context:height_cm"]
    names.extend(f"event:{name}" for name in OUTCOMES)
    names.extend(f"med_signal:{name.split(':', 1)[1]}" for name in FEATURE_NAMES if name.startswith("vitals:") and name.split(':', 1)[1] in {"epi", "nepi", "vaso", "eph", "phe", "dopai", "dobui"})
    for old_name, _ in sorted(old_vocab.items(), key=lambda item: item[1]):
        namespace = old_name.split(":", 1)[0]
        if namespace in {"phase", "department", "asa", "antype"}:
            names.append("context:" + old_name)
        elif namespace == "drug":
            names.append("med:" + old_name.split(":", 1)[1])
        elif namespace == "icd10_cm":
            names.append("diagnosis:" + old_name.split(":", 1)[1])
    return {name: index for index, name in enumerate(dict.fromkeys(names))}


def _outcome(record_name: str, time_min: float, value: float = 0.0,
             has_value: int = 0) -> Record:
    return Record(time_min, f"event:{record_name}", TOKEN_KINDS["clinical_outcome"],
                  OUTCOME_INDEX[record_name], value, has_value)


def _append_transitions(records: List[Record], transitions: Iterable[Transition], audit: Counter):
    for transition in transitions:
        records.append(_outcome(transition.name, transition.time_min,
                                transition.severity or transition.value, 1))
        audit[transition.name] += 1


def _manifest(source_root: Path):
    files = []
    for name in REQUIRED_SOURCE_FILES:
        path = source_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required train_data file: {path}")
        stat = path.stat()
        files.append({"name": name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return files


def preprocess(input_dir: Path, source_root: Path, output_dir: Path,
               max_admissions: Optional[int] = None, progress_every: int = 1000,
               min_outcome_count: int = 100, min_outcome_patients: int = 50,
               clock_interval_min: float = 360.0,
               max_clock_tokens_per_gap: int = 16):
    input_dir = input_dir.resolve()
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if "mimic" in str(source_root).lower():
        raise ValueError("MIMIC is explicitly outside this preprocessing task")
    source_manifest = _manifest(source_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_meta = json.loads((input_dir / "admission_timeline_meta.json").read_text())
    if source_meta.get("version") != 6 or not source_meta.get("complete"):
        raise ValueError("Input must be a complete v6 timeline")
    if source_meta.get("split") != "all_train":
        raise ValueError("Input must contain all train_data admissions")
    source_audit = json.loads((input_dir / source_meta["source_audit"]).read_text())
    for table in ("vitals", "labs", "ward_vitals", "medications", "diagnosis"):
        if table not in source_audit:
            raise ValueError(f"Source audit lacks {table}; refusing partial preprocessing")

    admissions = pd.read_csv(input_dir / "admissions.csv")
    operations = pd.concat(stream(source_root / "operations.csv.gz"), ignore_index=True)
    for column in ("subject_id", "hadm_id", "op_id", "orin_time", "orout_time",
                   "opstart_time", "opend_time", "anstart_time", "anend_time",
                   "icuin_time", "age", "weight", "height"):
        operations[column] = pd.to_numeric(operations[column], errors="coerce")
    admission_lookup = admissions[["subject_id", "hadm_id", "adm_idx", "start", "end", "death"]].rename(
        columns={"start": "admission_start", "end": "admission_end", "death": "admission_death"})
    episodes = operations.merge(admission_lookup, on=["subject_id", "hadm_id"], how="left", validate="many_to_one")
    if episodes["adm_idx"].isna().any() or episodes["orin_time"].isna().any() or episodes["orout_time"].isna().any():
        raise ValueError("Every operation must map to one audited admission with OR entry/exit")
    episodes["episode_start"] = np.maximum(
        episodes["admission_start"], episodes["orin_time"] - 7 * 1440.0)
    end_frame = episodes[["orout_time", "opend_time", "anend_time"]].copy()
    ignored_implausible_end_fields = 0
    for column in end_frame:
        valid_end = ((end_frame[column] >= episodes["orin_time"]) &
                     (end_frame[column] <= episodes["orin_time"] + 24 * 60.0))
        ignored_implausible_end_fields += int((end_frame[column].notna() & ~valid_end).sum())
        end_frame[column] = end_frame[column].where(valid_end)
    capped_operation_ends = int(end_frame.isna().all(axis=1).sum())
    surgery_end = end_frame.max(axis=1, skipna=True).fillna(episodes["orin_time"] + 24 * 60.0)
    episodes["episode_end"] = np.minimum(episodes["admission_end"], surgery_end + 30 * 1440.0)
    death_inside = (episodes["admission_death"].notna() &
                    (episodes["admission_death"] >= episodes["episode_start"]) &
                    (episodes["admission_death"] <= episodes["episode_end"]))
    episodes.loc[death_inside, "episode_end"] = episodes.loc[death_inside, "admission_death"]
    if (episodes["episode_end"] < episodes["episode_start"]).any():
        raise ValueError("Invalid perioperative episode interval")
    episodes = episodes.sort_values(["subject_id", "hadm_id", "orin_time", "op_id"]).reset_index(drop=True)
    if max_admissions is not None:
        episodes = episodes.iloc[:int(max_admissions)].copy()
    num_admissions = len(episodes)  # compatibility name: one sequence per operation episode
    old_vocab = json.loads((input_dir / "vocabulary.json").read_text())
    id_to_old = {int(value): key for key, value in old_vocab.items()}
    token_vocab = _build_token_vocabulary(old_vocab)
    features = list(source_meta["features"])
    feature_to_id = {name: index for index, name in enumerate(features)}
    obs_ptr = np.load(input_dir / "obs_admission_ptr.npy", mmap_mode="r")
    obs_bin = np.load(input_dir / "obs_bin.npy", mmap_mode="r")
    obs_feature = np.load(input_dir / "obs_feature.npy", mmap_mode="r")
    obs_value = np.load(input_dir / "obs_value.npy", mmap_mode="r")
    code_ptr = np.load(input_dir / "code_admission_ptr.npy", mmap_mode="r")
    code_bin = np.load(input_dir / "code_bin.npy", mmap_mode="r")
    code_id = np.load(input_dir / "code_id.npy", mmap_mode="r")
    static_baseline = np.column_stack([
        pd.to_numeric(episodes["age"], errors="coerce").fillna(0).to_numpy(np.float32) / 100.0,
        episodes["sex"].eq("M").to_numpy(np.float32),
    ]).astype(np.float32)
    np.save(output_dir / "static_baseline.npy", static_baseline)

    writers = {
        "token_id": BinaryWriter(output_dir / "token_id.bin", "int32"),
        "time_min": BinaryWriter(output_dir / "time_min.bin", "float32"),
        "value": BinaryWriter(output_dir / "value.bin", "float32"),
        "has_value": BinaryWriter(output_dir / "has_value.bin", "uint8"),
        "token_kind": BinaryWriter(output_dir / "token_kind.bin", "uint8"),
        "outcome_class": BinaryWriter(output_dir / "outcome_class.bin", "int16"),
    }
    ptr = [0]
    audit = Counter()
    written_outcomes = Counter()
    outcome_patients = {name: set() for name in OUTCOMES}
    written_context = Counter()
    response_audit = Counter()
    total_records = 0
    started = time.time()
    relevant_ids = {feature_to_id[name]: name for name in FEATURE_NAMES if name in feature_to_id}

    incomplete = {
        "version": 5, "complete": False, "source_root": str(source_root),
        "source_timeline": str(input_dir), "split": "all_train",
    }
    (output_dir / "event_sequence_meta.json").write_text(json.dumps(incomplete, indent=2))

    try:
        for episode_index, episode in episodes.iterrows():
            ai = int(episode.adm_idx)
            row = admissions.iloc[ai]
            full_duration = max(0.0, float(row.end) - float(row.start))
            death_rel = None
            if pd.notna(row.death) and float(row.start) < float(row.death) <= float(row.end):
                death_rel = float(row.death) - float(row.start)

            # One training sequence corresponds to one operation, preventing
            # operations years apart in a reused/long admission interval from
            # becoming one pseudo-perioperative trajectory.
            cp_lo, cp_hi = int(code_ptr[ai]), int(code_ptr[ai + 1])
            episode_start = float(episode.episode_start) - float(row.start)
            episode_end = float(episode.episode_end) - float(row.start)
            or_in = float(episode.orin_time) - float(row.start)
            or_out = float(episode.orout_time) - float(row.start)
            duration = episode_end - episode_start
            records: List[Record] = []

            # Age and sex remain admission-level static covariates. Weight and
            # height become timestamped context because v6 records when those
            # operation attributes first become available.
            static_known = or_in
            if pd.notna(episode.weight) and pd.notna(episode.height):
                known_at = max(episode_start, static_known)
                records.append(Record(known_at, "context:weight_kg",
                                      TOKEN_KINDS["static_observation"], value=float(episode.weight), has_value=1))
                records.append(Record(known_at, "context:height_cm",
                                      TOKEN_KINDS["static_observation"], value=float(episode.height), has_value=1))

            # Operation-specific context comes directly from operations.csv,
            # avoiding phase tokens from a different operation in the same admission.
            phase_outcomes = {
                "orin_time": "or_entry", "anstart_time": "anesthesia_start",
                "opstart_time": "surgery_start", "cpbon_time": "cpb_start",
                "cpboff_time": "cpb_stop", "opend_time": "surgery_end",
                "anend_time": "anesthesia_end", "orout_time": "or_exit",
            }
            for phase_name, outcome_name in phase_outcomes.items():
                absolute = pd.to_numeric(episode.get(phase_name, np.nan), errors="coerce")
                if pd.notna(absolute):
                    t = float(absolute) - float(row.start)
                    if episode_start <= t <= episode_end:
                        records.append(_outcome(outcome_name, t))
                        audit[outcome_name] += 1
            for field in ("department", "asa", "antype"):
                value = episode.get(field, np.nan)
                if pd.notna(value):
                    token = f"context:{field}:{value}"
                    if token in token_vocab:
                        records.append(Record(or_in, token, TOKEN_KINDS["procedure_context"]))

            op_lo, op_hi = int(obs_ptr[ai]), int(obs_ptr[ai + 1])
            local_bins = np.asarray(obs_bin[op_lo:op_hi], dtype=np.float32)
            local_features = np.asarray(obs_feature[op_lo:op_hi], dtype=np.int32)
            local_values = np.asarray(obs_value[op_lo:op_hi], dtype=np.float32)
            feature_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for feature_id in np.intersect1d(np.unique(local_features), np.fromiter(relevant_ids, dtype=np.int32)):
                use = local_features == feature_id
                feature_arrays[relevant_ids[int(feature_id)]] = (
                    local_bins[use] * 5.0,
                    # v6 stores physical source values.  Normalizer statistics
                    # are consumed by the old dense dataset at load time; they
                    # must not be inverted here.
                    local_values[use],
                )

            # Declarative v4 ontology: only state transitions, severity
            # escalations and abrupt measured changes become event tokens.
            # Repeated normal five-minute observations are never copied.
            _append_transitions(records, extract_measurement_events(feature_arrays), audit)

            sbp = _series(feature_arrays, "vitals:art_sbp", "vitals:nibp_sbp",
                          "ward_vitals:art_sbp", "ward_vitals:nibp_sbp")
            mbp = _series(feature_arrays, "vitals:art_mbp", "vitals:nibp_mbp",
                          "ward_vitals:nibp_mbp")
            bp_times, bp_abnormality = _abnormality_series(sbp, mbp)

            for feature_name, start_name, stop_name, gap in (
                ("ward_vitals:vent", "ventilation_start", "ventilation_stop", 360.0),
                ("ward_vitals:crrt", "crrt_start", "crrt_stop", 720.0),
                ("ward_vitals:ecmo", "ecmo_start", "ecmo_stop", 720.0),
                ("ward_vitals:iabp", "iabp_start", "iabp_stop", 720.0),
            ):
                if feature_name in feature_arrays:
                    _append_transitions(records, binary_transitions(
                        *feature_arrays[feature_name], start_name, stop_name,
                        inactivity_gap=gap), audit)

            if "vitals:ebl" in feature_arrays:
                ebl_times, ebl_values = feature_arrays["vitals:ebl"]
                emitted_major = emitted_severe = False
                for t, value in zip(ebl_times, ebl_values):
                    if value >= 500 and not emitted_severe:
                        records.append(_outcome("severe_bleeding_signal", float(t), float(value), 1))
                        audit["severe_bleeding_signal"] += 1
                        emitted_severe = True
                    elif value >= 300 and not emitted_major:
                        records.append(_outcome("major_bleeding_signal", float(t), float(value), 1))
                        audit["major_bleeding_signal"] += 1
                        emitted_major = True

            for feature_name, event_name in (
                ("vitals:rbc", "rbc_transfusion"), ("vitals:ffp", "ffp_transfusion"),
                ("vitals:pheresis", "platelet_transfusion"),
                ("vitals:cryo", "cryo_transfusion"),
            ):
                if feature_name not in feature_arrays:
                    continue
                last = -math.inf
                for t, value in zip(*feature_arrays[feature_name]):
                    if value > 0 and float(t) - last >= 30.0:
                        records.append(_outcome(event_name, float(t), float(value), 1))
                        audit[event_name] += 1
                        last = float(t)

            vasopressor_starts: List[float] = []
            for name in ("vitals:epi", "vitals:nepi", "vitals:vaso", "vitals:eph",
                         "vitals:phe", "vitals:dopai", "vitals:dobui"):
                if name not in feature_arrays:
                    continue
                signal = feature_arrays[name]
                for transition in binary_transitions(*signal, "vasopressor_start", "vasopressor_stop"):
                    records.append(_outcome(transition.name, transition.time_min,
                                            transition.value, 1))
                    records.append(Record(transition.time_min, "med_signal:" + name.split(":", 1)[1],
                                          TOKEN_KINDS["medication_context"], value=transition.value,
                                          has_value=1))
                    audit[transition.name] += 1
                    if transition.name == "vasopressor_start":
                        vasopressor_starts.append(transition.time_min)

            last_vaso_drug = -math.inf
            last_medication: Dict[str, float] = {}
            for bin_value, old_id in zip(code_bin[cp_lo:cp_hi], code_id[cp_lo:cp_hi]):
                t = float(bin_value) * 5.0
                if t < 0 or t > full_duration:
                    continue
                old_name = id_to_old.get(int(old_id), "")
                namespace, _, value_name = old_name.partition(":")
                if namespace in {"phase", "department", "asa", "antype"}:
                    continue  # operation-specific context was added from the raw operation row
                elif namespace == "drug":
                    if not episode_start <= t <= episode_end:
                        continue
                    token = "med:" + value_name
                    # Collapse duplicate charting of the same medication inside
                    # a 15-minute window while preserving later administrations.
                    if t - last_medication.get(token, -math.inf) >= 15.0:
                        records.append(Record(t, token, TOKEN_KINDS["medication_context"]))
                        last_medication[token] = t
                        audit["medication_administration"] += 1
                    lower = value_name.lower()
                    if any(term in lower for term in VASOPRESSOR_DRUG_TERMS) and t - last_vaso_drug >= 30.0:
                        records.append(_outcome("vasopressor_start", t))
                        audit["vasopressor_start"] += 1
                        vasopressor_starts.append(t)
                        last_vaso_drug = t
                elif namespace == "icd10_cm":
                    if t <= episode_end:
                        records.append(Record(max(t, episode_start), "diagnosis:" + value_name,
                                              TOKEN_KINDS["diagnosis_context"]))

            for vaso_time in sorted(set(vasopressor_starts)):
                if not episode_start <= vaso_time <= episode_end:
                    continue
                status, response_time = observed_vasopressor_response(
                    vaso_time, bp_times, bp_abnormality)
                response_audit[status] += 1
                if status == "recovered":
                    records.append(_outcome("bp_recovered_after_vasopressor_observed", response_time))
                    audit["bp_recovered_after_vasopressor_observed"] += 1
                elif status == "persistent":
                    records.append(_outcome("hypotension_persistent_after_vasopressor_observed", response_time))
                    audit["hypotension_persistent_after_vasopressor_observed"] += 1

            for column, event_name in (("icuin_time", "icu_transfer"),
                                       ("icuout_time", "icu_discharge")):
                absolute = pd.to_numeric(episode.get(column, np.nan), errors="coerce")
                if pd.notna(absolute):
                    t = float(absolute) - float(row.start)
                    if episode_start <= t <= episode_end:
                        records.append(_outcome(event_name, t))
                        audit[event_name] += 1
            if death_rel is not None and episode_start <= death_rel <= episode_end:
                records.append(_outcome("inhospital_death", death_rel))
                audit["inhospital_death"] += 1

            shifted = [Record(record.time_min - episode_start, record.token, record.kind,
                              record.outcome, record.value, record.has_value)
                       for record in records if episode_start <= record.time_min <= episode_end]
            records = _dedupe(
                [Record(0.0, "<BOS>", TOKEN_KINDS["boundary"]), *shifted,
                 Record(duration, "<EPISODE_END>", TOKEN_KINDS["boundary"])]
            )
            records = _insert_clock_tokens(
                records, clock_interval_min, max_clock_tokens_per_gap)
            measurement_chunks = [
                times[(times >= episode_start) & (times <= episode_end)] - episode_start
                for times, _ in feature_arrays.values()
                if np.any((times >= episode_start) & (times <= episode_end))
            ]
            measurement_times = (np.concatenate(measurement_chunks).astype(np.float32)
                                 if measurement_chunks else np.empty(0, dtype=np.float32))
            records = _annotate_clock_intensity(
                records, measurement_times, clock_interval_min)
            records = _insert_phase_summary_tokens(records)
            if not records or records[0].token != "<BOS>":
                raise RuntimeError(f"Admission {ai} lost its BOS token")
            times = np.asarray([record.time_min for record in records], dtype=np.float32)
            if np.any(times[1:] < times[:-1]):
                raise RuntimeError(f"Admission {ai} is not chronologically sorted")
            episode_outcomes = set()
            for record in records:
                if record.outcome >= 0:
                    outcome_name = OUTCOMES[record.outcome]
                    written_outcomes[outcome_name] += 1
                    episode_outcomes.add(outcome_name)
                elif record.token.startswith(("med:", "med_signal:")):
                    written_context["medication_context"] += 1
                elif record.token.startswith("diagnosis:"):
                    written_context["diagnosis_context"] += 1
                elif record.token.startswith("context:"):
                    written_context["procedure_or_static_context"] += 1
                elif record.token == "<CLOCK>":
                    written_context["clock_no_event"] += 1
            patient_id = str(episode.subject_id)
            for outcome_name in episode_outcomes:
                outcome_patients[outcome_name].add(patient_id)

            writers["token_id"].write(token_vocab[record.token] for record in records)
            writers["time_min"].write(record.time_min for record in records)
            writers["value"].write(record.value for record in records)
            writers["has_value"].write(record.has_value for record in records)
            writers["token_kind"].write(record.kind for record in records)
            writers["outcome_class"].write(record.outcome for record in records)
            total_records += len(records)
            ptr.append(total_records)

            if progress_every and (episode_index + 1) % progress_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(json.dumps({
                    "operation_episodes": episode_index + 1, "total_operation_episodes": num_admissions,
                    "tokens": total_records, "episodes_per_second": (episode_index + 1) / elapsed,
                    "progress_percent": round(100 * (episode_index + 1) / num_admissions, 2),
                }), flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    counts = {writer.count for writer in writers.values()}
    if counts != {total_records}:
        raise RuntimeError(f"Binary array length mismatch: {counts} versus {total_records}")
    np.save(output_dir / "sequence_ptr.npy", np.asarray(ptr, dtype=np.int64))
    episodes["episode_length_min"] = episodes["episode_end"] - episodes["episode_start"]
    episodes.to_csv(output_dir / "admissions.csv", index=False)
    (output_dir / "token_vocabulary.json").write_text(
        json.dumps(token_vocab, ensure_ascii=False, indent=2))

    observed = [name for name in OUTCOMES if written_outcomes[name] > 0]
    patient_counts = {name: len(outcome_patients[name]) for name in OUTCOMES}
    selected = {
        name for name in observed
        if written_outcomes[name] >= min_outcome_count
        and patient_counts[name] >= min_outcome_patients
    }
    active_outcomes = [name for name in OUTCOMES if name in selected]
    active_index = {name: index for index, name in enumerate(active_outcomes)}
    outcome_class_remap = [active_index.get(name, -1) for name in OUTCOMES]
    active_families = family_names(active_outcomes)
    unavailable_outcomes = {
        name: "No qualifying source observations in the complete train_data dataset; not a learnable output."
        for name in OUTCOMES if written_outcomes[name] == 0
    }
    meta = {
        "version": 5,
        "complete": True,
        "created_unix": time.time(),
        "source_root": str(source_root),
        "source_timeline": str(input_dir),
        "source_timeline_version": 6,
        "source_manifest": source_manifest,
        "source_tables": list(REQUIRED_SOURCE_FILES),
        "mimic_used": False,
        "split": "all_train",
        "num_admissions": num_admissions,
        "training_unit": "operation_episode",
        "num_operation_episodes": num_admissions,
        "num_unique_admissions": int(episodes[["subject_id", "hadm_id"]].drop_duplicates().shape[0]),
        "num_tokens": total_records,
        "num_static": 2,
        "num_source_dynamic_features": len(features),
        "source_dynamic_features": features,
        "static_features": [
            {"name": "age_at_operation", "encoding": "age_years/100", "availability": "admission baseline", "predict": False},
            {"name": "male", "encoding": "binary", "availability": "admission baseline", "predict": False},
        ],
        "time_varying_context": [
            "weight and height at first operation-known time", "procedure phase", "department",
            "ASA", "anaesthesia type", "medication administration", "diagnosis recording",
        ],
        "prediction_target": (
            "multi-task sparse event learning: next outcome set and elapsed time, future "
            "1/6/24-hour event trajectory, and masked event-token reconstruction"
        ),
        "trajectory_horizons_hours": [1, 6, 24],
        "perioperative_window": {
            "preoperative_minutes": 10080,
            "postoperative_minutes": 43200,
            "maximum_operation_span_minutes": 1440,
            "ignored_implausible_operation_end_fields": ignored_implausible_end_fields,
            "operation_ends_capped_at_maximum_span": capped_operation_ends,
            "start_anchor": "this operation's recorded OR entry",
            "end_anchor": "this operation's OR/anesthesia/operation end",
            "earlier_terminal_events": ["discharge", "in-hospital death"],
        },
        "token_vocabulary": token_vocab,
        "outcome_vocabulary": active_outcomes,
        "outcome_family_vocabulary": active_families,
        "outcome_to_family": [active_families.index(OUTCOME_TO_FAMILY[name])
                              for name in active_outcomes],
        "stored_outcome_vocabulary": OUTCOMES,
        "outcome_class_remap": outcome_class_remap,
        "unavailable_outcomes": unavailable_outcomes,
        "token_kinds": TOKEN_KINDS,
        "outcome_definitions": OUTCOME_DEFINITIONS,
        "outcome_selection": {
            "candidate_count": len(OUTCOMES),
            "observed_candidate_count": len(observed),
            "selected_count": len(active_outcomes),
            "minimum_occurrences": min_outcome_count,
            "minimum_independent_patients": min_outcome_patients,
            "maximum_selected_outcomes": None,
            "selection_rule": "all events meeting both occurrence and independent-patient thresholds",
            "formerly_mandatory_outcomes_meeting_rule": [
                name for name in ALWAYS_KEEP_OUTCOMES if name in selected
            ],
            "patient_counts": {
                name: patient_counts[name] for name in observed
            },
            "excluded_observed": {
                name: {
                    "occurrences": written_outcomes[name],
                    "independent_patients": patient_counts[name],
                    "below_occurrence_threshold": written_outcomes[name] < min_outcome_count,
                    "below_patient_threshold": patient_counts[name] < min_outcome_patients,
                }
                for name in observed if name not in selected
            },
        },
        "normal_measurement_policy": (
            "Unchanged/normal five-minute measurements are omitted. Sparse clock/no-event "
            "tokens explicitly represent elapsed quiet time; episode-end right censoring "
            "represents periods with no later outcome."
        ),
        "clock_token_policy": {
            "interval_minutes": clock_interval_min,
            "maximum_tokens_per_inter_record_gap": max_clock_tokens_per_gap,
        },
        "phase_summary_tokens": [
            "phase_summary:preop", "phase_summary:induction", "phase_summary:surgery",
            "phase_summary:emergence", "phase_summary:pacu", "phase_summary:icu",
        ],
        "medication_response_semantics": "Observed post-medication trajectory; association only, no counterfactual causal claim.",
        "audit_counts": dict(written_outcomes),
        "context_counts": dict(written_context),
        "extraction_candidate_counts_before_deduplication": dict(audit),
        "vasopressor_response_audit": dict(response_audit),
        "source_audit": source_audit,
        "max_admissions": max_admissions,
    }
    (output_dir / "event_sequence_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps({
        "complete": True, "operation_episodes": num_admissions, "tokens": total_records,
        "outcomes": dict(written_outcomes), "vasopressor_response": dict(response_audit),
        "elapsed_seconds": round(time.time() - started, 1),
    }, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/admission_timeline_v6")
    parser.add_argument("--source_root", default="data/train_data")
    parser.add_argument("--output", default="data/perioperative_event_sequences_v5_full")
    parser.add_argument("--max_admissions", type=int)
    parser.add_argument("--progress_every", type=int, default=1000)
    parser.add_argument("--min_outcome_count", type=int, default=100)
    parser.add_argument("--min_outcome_patients", type=int, default=50)
    parser.add_argument("--clock_interval_min", type=float, default=360.0)
    parser.add_argument("--max_clock_tokens_per_gap", type=int, default=16)
    args = parser.parse_args()
    preprocess(Path(args.input), Path(args.source_root), Path(args.output),
               args.max_admissions, args.progress_every,
               args.min_outcome_count, args.min_outcome_patients,
               args.clock_interval_min, args.max_clock_tokens_per_gap)


if __name__ == "__main__":
    main()
