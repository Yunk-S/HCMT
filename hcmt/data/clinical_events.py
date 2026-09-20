"""Declarative clinical event ontology for sparse perioperative timelines.

The source timeline contains irregular observations.  This module converts
those observations into state changes, severity escalations and abrupt-change
events.  It deliberately does not emit a token for every five-minute row.
Thresholds are screening/event definitions for representation learning, not
stand-alone diagnoses or treatment recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .event_extraction import Transition, merge_series


Series = Tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class LevelSpec:
    sources: Tuple[str, ...]
    direction: str
    levels: Tuple[Tuple[float, str], ...]
    recovery: str | None = None
    confirmation_gap_min: float = 0.0
    immediate_level: int = 2
    description: str = ""


def _low(sources, levels, recovery=None, gap=0.0, immediate=2, description=""):
    return LevelSpec(tuple(sources), "low", tuple(levels), recovery, gap, immediate, description)


def _high(sources, levels, recovery=None, gap=0.0, immediate=2, description=""):
    return LevelSpec(tuple(sources), "high", tuple(levels), recovery, gap, immediate, description)


# Levels are ordered mild -> severe.  Vital-sign rules require confirmation for
# their first level but accept a severe reading immediately.  Laboratory rules
# are emitted from a measured result and therefore do not invent intermediate
# five-minute values.
LEVEL_SPECS: Tuple[LevelSpec, ...] = (
    _low(("vitals:art_sbp", "vitals:nibp_sbp", "ward_vitals:art_sbp", "ward_vitals:nibp_sbp"),
         ((90, "systolic_hypotension"), (80, "severe_systolic_hypotension"),
          (70, "profound_systolic_hypotension")), "systolic_bp_normalized", 120),
    _low(("vitals:art_mbp", "vitals:nibp_mbp", "ward_vitals:nibp_mbp"),
         ((65, "map_hypotension"), (55, "severe_map_hypotension"),
          (45, "profound_map_hypotension")), "map_normalized", 120),
    _high(("vitals:art_sbp", "vitals:nibp_sbp", "ward_vitals:art_sbp", "ward_vitals:nibp_sbp"),
          ((140, "systolic_hypertension"), (160, "severe_systolic_hypertension"),
           (180, "hypertensive_crisis_signal")), "systolic_bp_normalized", 120),
    _low(("vitals:art_dbp", "vitals:nibp_dbp", "ward_vitals:nibp_dbp"),
         ((50, "diastolic_hypotension"), (40, "severe_diastolic_hypotension")),
         "diastolic_bp_normalized", 120),
    _high(("vitals:art_dbp", "vitals:nibp_dbp", "ward_vitals:nibp_dbp"),
          ((90, "diastolic_hypertension"), (100, "severe_diastolic_hypertension")),
          "diastolic_bp_normalized", 120),
    _low(("vitals:hr", "ward_vitals:hr"),
         ((60, "bradycardia"), (50, "severe_bradycardia")),
         "heart_rate_normalized", 120),
    _high(("vitals:hr", "ward_vitals:hr"),
          ((100, "tachycardia"), (110, "marked_tachycardia"), (120, "severe_tachycardia")),
          "heart_rate_normalized", 120),
    _low(("vitals:spo2", "ward_vitals:spo2"),
         ((94, "oxygen_desaturation"), (90, "hypoxemia"), (85, "severe_hypoxemia")),
         "oxygenation_normalized", 120),
    _low(("vitals:rr", "ward_vitals:rr"),
         ((10, "bradypnea"), (8, "severe_bradypnea")),
         "respiratory_rate_normalized", 120),
    _high(("vitals:rr", "ward_vitals:rr"),
          ((24, "tachypnea"), (30, "respiratory_distress"), (40, "severe_respiratory_distress")),
          "respiratory_rate_normalized", 120),
    _low(("vitals:etco2",), ((30, "low_etco2"), (25, "severe_low_etco2")),
         "etco2_normalized", 30),
    _high(("vitals:etco2",), ((45, "high_etco2"), (55, "severe_high_etco2")),
          "etco2_normalized", 30),
    _low(("vitals:minvol",), ((3, "low_minute_ventilation"), (2, "severe_low_minute_ventilation")),
         "minute_ventilation_normalized", 30),
    _high(("vitals:pip",), ((25, "elevated_peak_airway_pressure"), (30, "high_peak_airway_pressure")),
          "airway_pressure_normalized", 30),
    _high(("vitals:pplat",), ((25, "elevated_plateau_pressure"), (30, "high_plateau_pressure")),
          "plateau_pressure_normalized", 30),
    _high(("vitals:peep",), ((8, "high_peep_support"), (10, "very_high_peep_support")),
          "peep_support_reduced", 30),
    _high(("vitals:fio2", "ward_vitals:fio2"),
          ((50, "high_oxygen_requirement"), (80, "very_high_oxygen_requirement")),
          "oxygen_requirement_reduced", 120),
    _low(("vitals:bt", "ward_vitals:bt"),
         ((36, "hypothermia"), (35, "severe_hypothermia")), "temperature_normalized", 120),
    _high(("vitals:bt", "ward_vitals:bt"),
          ((37.5, "elevated_temperature"), (38, "fever")), "temperature_normalized", 120),
    _low(("vitals:ci",), ((2.2, "low_cardiac_index"), (1.8, "severely_low_cardiac_index")),
         "cardiac_index_normalized", 30),
    _high(("vitals:cvp",), ((12, "elevated_cvp"), (16, "markedly_elevated_cvp")),
          "cvp_normalized", 30),
    _high(("vitals:pap_mbp",), ((25, "elevated_pulmonary_artery_pressure"),
                                (35, "severe_pulmonary_artery_pressure")),
          "pulmonary_artery_pressure_normalized", 30),
    _low(("labs:glucose",), ((70, "hypoglycemia"), (54, "severe_hypoglycemia")),
         "glucose_normalized"),
    _high(("labs:glucose",), ((180, "hyperglycemia"), (250, "severe_hyperglycemia")),
          "glucose_normalized"),
    _high(("labs:lactate",), ((2, "hyperlactatemia"), (4, "severe_hyperlactatemia")),
          "lactate_normalized"),
    _low(("labs:sodium",), ((135, "hyponatremia"), (130, "moderate_hyponatremia"),
                            (125, "severe_hyponatremia")), "sodium_normalized"),
    _high(("labs:sodium",), ((145, "hypernatremia"), (150, "severe_hypernatremia")),
          "sodium_normalized"),
    _low(("labs:potassium",), ((3.5, "hypokalemia"), (3.0, "severe_hypokalemia")),
         "potassium_normalized"),
    _high(("labs:potassium",), ((5.0, "hyperkalemia"), (5.3, "severe_hyperkalemia")),
          "potassium_normalized"),
    _low(("labs:calcium",), ((8.0, "hypocalcemia"), (7.5, "severe_hypocalcemia")),
         "calcium_normalized"),
    _low(("labs:ica",), ((1.0, "low_ionized_calcium"), (0.9, "severely_low_ionized_calcium")),
         "ionized_calcium_normalized"),
    _low(("labs:phosphorus",), ((2.5, "hypophosphatemia"), (2.0, "severe_hypophosphatemia")),
         "phosphorus_normalized"),
    _high(("labs:phosphorus",), ((4.5, "hyperphosphatemia"), (5.0, "severe_hyperphosphatemia")),
          "phosphorus_normalized"),
    _low(("labs:ph",), ((7.35, "acidemia"), (7.25, "severe_acidemia")), "ph_normalized"),
    _high(("labs:ph",), ((7.45, "alkalemia"), (7.52, "severe_alkalemia")), "ph_normalized"),
    _low(("labs:hco3",), ((22, "low_bicarbonate"), (18, "severely_low_bicarbonate")),
         "bicarbonate_normalized"),
    _high(("labs:hco3",), ((30, "high_bicarbonate"), (34, "severely_high_bicarbonate")),
          "bicarbonate_normalized"),
    _low(("labs:be",), ((-5, "base_deficit"), (-8, "severe_base_deficit")),
         "base_excess_normalized"),
    _high(("labs:be",), ((5, "base_excess"), (8, "severe_base_excess")),
          "base_excess_normalized"),
    _low(("labs:pao2",), ((60, "low_arterial_oxygen"), (45, "severely_low_arterial_oxygen")),
         "arterial_oxygen_normalized"),
    _low(("labs:sao2",), ((90, "low_arterial_oxygen_saturation"),
                           (80, "severely_low_arterial_oxygen_saturation")),
         "arterial_oxygen_saturation_normalized"),
    _low(("labs:paco2",), ((32, "hypocapnia"), (28, "severe_hypocapnia")),
         "paco2_normalized"),
    _high(("labs:paco2",), ((50, "hypercapnia"), (60, "severe_hypercapnia")),
          "paco2_normalized"),
    _high(("labs:bun",), ((30, "azotemia"), (50, "severe_azotemia")), "bun_normalized"),
    _low(("labs:hb",), ((10, "anemia"), (8, "severe_anemia")), "hemoglobin_normalized"),
    _low(("labs:hct",), ((30, "low_hematocrit"), (24, "severely_low_hematocrit")),
         "hematocrit_normalized"),
    _low(("labs:platelet",), ((150, "thrombocytopenia"), (100, "moderate_thrombocytopenia"),
                              (50, "severe_thrombocytopenia")), "platelet_normalized"),
    _high(("labs:ptinr",), ((1.5, "elevated_inr"), (2.5, "severely_elevated_inr")),
          "inr_normalized"),
    _high(("labs:aptt",), ((45, "prolonged_aptt"), (70, "severely_prolonged_aptt")),
          "aptt_normalized"),
    _low(("labs:fibrinogen",), ((200, "low_fibrinogen"), (150, "severely_low_fibrinogen")),
         "fibrinogen_normalized"),
    _low(("labs:albumin",), ((3.0, "hypoalbuminemia"), (2.5, "severe_hypoalbuminemia")),
         "albumin_normalized"),
    _high(("labs:total_bilirubin",), ((2, "hyperbilirubinemia"), (4, "severe_hyperbilirubinemia")),
          "bilirubin_normalized"),
    _high(("labs:ast",), ((100, "ast_elevation"), (180, "severe_ast_elevation")),
          "ast_normalized"),
    _high(("labs:alt",), ((100, "alt_elevation"), (200, "severe_alt_elevation")),
          "alt_normalized"),
    _low(("labs:wbc",), ((4, "leukopenia"), (3, "severe_leukopenia")), "wbc_normalized"),
    _high(("labs:wbc",), ((12, "leukocytosis"), (18, "severe_leukocytosis")),
          "wbc_normalized"),
    _high(("labs:crp",), ((5, "crp_elevation"), (10, "severe_crp_elevation")),
          "crp_normalized"),
    _high(("labs:troponin_i",), ((0.04, "troponin_i_elevation"), (1, "marked_troponin_i_elevation")),
          "troponin_i_normalized"),
    _high(("labs:ckmb",), ((5, "ckmb_elevation"), (20, "marked_ckmb_elevation")),
          "ckmb_normalized"),
    _high(("labs:ck",), ((500, "ck_elevation"), (1000, "marked_ck_elevation")),
          "ck_normalized"),
    _low(("ward_vitals:gcs_m",), ((5, "gcs_motor_decline"), (3, "severe_gcs_motor_decline")),
         "gcs_motor_normalized", 240),
    _low(("ward_vitals:gcs_e",), ((3, "gcs_eye_decline"), (2, "severe_gcs_eye_decline")),
         "gcs_eye_normalized", 240),
)


ABRUPT_SPECS = (
    (("vitals:art_sbp", "vitals:nibp_sbp", "ward_vitals:art_sbp", "ward_vitals:nibp_sbp"),
     20.0, 40.0, "acute_sbp_drop", "major_acute_sbp_drop", "drop"),
    (("vitals:art_sbp", "vitals:nibp_sbp", "ward_vitals:art_sbp", "ward_vitals:nibp_sbp"),
     20.0, 40.0, "acute_sbp_rise", "major_acute_sbp_rise", "rise"),
    (("vitals:art_mbp", "vitals:nibp_mbp", "ward_vitals:nibp_mbp"),
     15.0, 25.0, "acute_map_drop", "major_acute_map_drop", "drop"),
    (("vitals:hr", "ward_vitals:hr"), 20.0, 35.0,
     "acute_heart_rate_drop", "major_acute_heart_rate_drop", "drop"),
    (("vitals:hr", "ward_vitals:hr"), 20.0, 35.0,
     "acute_heart_rate_rise", "major_acute_heart_rate_rise", "rise"),
    (("labs:hb",), 1.5, 2.5, "acute_hemoglobin_drop", "major_acute_hemoglobin_drop", "drop"),
)


SPECIAL_OUTCOMES = (
    "aki_stage_1_signal", "aki_stage_2_signal", "aki_stage_3_signal", "aki_recovery_signal",
    "major_bleeding_signal", "severe_bleeding_signal",
    "rbc_transfusion", "ffp_transfusion", "platelet_transfusion", "cryo_transfusion",
    "vasopressor_start", "vasopressor_stop", "ventilation_start", "ventilation_stop",
    "crrt_start", "crrt_stop", "ecmo_start", "ecmo_stop", "iabp_start", "iabp_stop",
    "bp_recovered_after_vasopressor_observed",
    "hypotension_persistent_after_vasopressor_observed",
    "or_entry", "anesthesia_start", "surgery_start", "cpb_start", "cpb_stop",
    "surgery_end", "anesthesia_end", "or_exit", "icu_transfer", "icu_discharge",
    "inhospital_death",
)


def _ordered_unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))


OUTCOMES = _ordered_unique(
    [event for spec in LEVEL_SPECS for _, event in spec.levels]
    + [spec.recovery for spec in LEVEL_SPECS if spec.recovery]
    + [item for spec in ABRUPT_SPECS for item in (spec[3], spec[4])]
    + list(SPECIAL_OUTCOMES)
)

# Rare but clinically important endpoints are retained even when frequency
# pruning is needed to keep the learnable head in the requested 50--100 class
# range.  The remaining slots are selected by observed training frequency.
ALWAYS_KEEP_OUTCOMES = (
    "profound_systolic_hypotension", "profound_map_hypotension",
    "hypertensive_crisis_signal", "severe_bradycardia", "severe_tachycardia",
    "severe_hypoxemia", "severe_bradypnea", "severe_respiratory_distress",
    "severe_acidemia", "severe_hypoglycemia", "severe_hyperglycemia",
    "severe_hyperlactatemia",
    "severe_hyponatremia", "severe_hypernatremia", "severe_hypokalemia",
    "severe_hyperkalemia", "severe_anemia", "severe_thrombocytopenia",
    "severely_elevated_inr", "severely_prolonged_aptt",
    "severely_low_fibrinogen", "marked_troponin_i_elevation",
    "aki_stage_1_signal", "aki_stage_2_signal", "aki_stage_3_signal",
    "major_bleeding_signal", "severe_bleeding_signal",
    "rbc_transfusion", "ffp_transfusion", "platelet_transfusion", "cryo_transfusion",
    "vasopressor_start", "vasopressor_stop", "ventilation_start", "ventilation_stop",
    "crrt_start", "ecmo_start", "iabp_start",
    "bp_recovered_after_vasopressor_observed",
    "hypotension_persistent_after_vasopressor_observed",
    "or_entry", "anesthesia_start", "surgery_start", "cpb_start", "cpb_stop",
    "surgery_end", "anesthesia_end", "or_exit", "icu_transfer", "icu_discharge",
    "inhospital_death",
)


FEATURE_NAMES = tuple(sorted(set(
    source for spec in LEVEL_SPECS for source in spec.sources
) | set(
    source for spec in ABRUPT_SPECS for source in spec[0]
) | {
    "labs:creatinine", "vitals:ebl", "vitals:rbc", "vitals:ffp",
    "vitals:pheresis", "vitals:cryo", "ward_vitals:vent", "ward_vitals:crrt",
    "ward_vitals:ecmo", "ward_vitals:iabp", "vitals:epi", "vitals:nepi",
    "vitals:vaso", "vitals:eph", "vitals:phe", "vitals:dopai", "vitals:dobui",
}))


OUTCOME_DEFINITIONS = {
    event: (spec.description or
            f"Sparse measured-state transition ({spec.direction}) at threshold {threshold:g}; "
            "normal unchanged rows are omitted.")
    for spec in LEVEL_SPECS for threshold, event in spec.levels
}
for spec in LEVEL_SPECS:
    if spec.recovery:
        OUTCOME_DEFINITIONS.setdefault(
            spec.recovery, "Measured return from a previously emitted abnormal state.")
for sources, moderate, major, moderate_name, major_name, direction in ABRUPT_SPECS:
    OUTCOME_DEFINITIONS[moderate_name] = (
        f"Observed {direction} of at least {moderate:g} from the recent measured baseline.")
    OUTCOME_DEFINITIONS[major_name] = (
        f"Observed {direction} of at least {major:g} from the recent measured baseline.")
OUTCOME_DEFINITIONS.update({
    "aki_stage_1_signal": "Creatinine rise >=0.3 mg/dL or >=1.5x prior episode minimum; KDIGO-inspired signal.",
    "aki_stage_2_signal": "Creatinine >=2x prior episode minimum; KDIGO-inspired signal.",
    "aki_stage_3_signal": "Creatinine >=3x prior episode minimum or >=4 mg/dL; KDIGO-inspired signal.",
    "aki_recovery_signal": "Creatinine returns below stage-1 signal relative to prior episode minimum.",
    "major_bleeding_signal": "Recorded estimated blood loss reaches at least 300 mL.",
    "severe_bleeding_signal": "Recorded estimated blood loss reaches at least 500 mL.",
    "rbc_transfusion": "Positive intraoperative red-cell transfusion record.",
    "ffp_transfusion": "Positive fresh-frozen-plasma transfusion record.",
    "platelet_transfusion": "Positive platelet/pheresis transfusion record.",
    "cryo_transfusion": "Positive cryoprecipitate transfusion record.",
    "vasopressor_start": "Vasopressor monitor signal or vasoactive administration starts after inactivity.",
    "vasopressor_stop": "Vasopressor monitor signal stops or becomes inactive.",
    "ventilation_start": "Mechanical ventilation indicator becomes active.",
    "ventilation_stop": "Mechanical ventilation indicator becomes inactive.",
    "crrt_start": "CRRT indicator becomes active.", "crrt_stop": "CRRT indicator becomes inactive.",
    "ecmo_start": "ECMO indicator becomes active.", "ecmo_stop": "ECMO indicator becomes inactive.",
    "iabp_start": "IABP indicator becomes active.", "iabp_stop": "IABP indicator becomes inactive.",
    "bp_recovered_after_vasopressor_observed": "Observed BP recovery after vasopressor; association only.",
    "hypotension_persistent_after_vasopressor_observed": "Observed persistent hypotension after vasopressor; association only.",
    "or_entry": "Recorded operating-room entry.", "or_exit": "Recorded operating-room exit.",
    "anesthesia_start": "Recorded anaesthesia start.", "anesthesia_end": "Recorded anaesthesia end.",
    "surgery_start": "Recorded operation start.", "surgery_end": "Recorded operation end.",
    "cpb_start": "Recorded cardiopulmonary-bypass start.", "cpb_stop": "Recorded cardiopulmonary-bypass stop.",
    "icu_transfer": "Recorded ICU admission in the perioperative episode.",
    "icu_discharge": "Recorded ICU discharge in the perioperative episode.",
    "inhospital_death": "Recorded death within the current hospital admission.",
})


def _level(value: float, spec: LevelSpec) -> int:
    level = 0
    for index, (threshold, _) in enumerate(spec.levels, start=1):
        if ((spec.direction == "low" and value <= threshold) or
                (spec.direction == "high" and value >= threshold)):
            level = index
    return level


def level_transitions(times: np.ndarray, values: np.ndarray, spec: LevelSpec) -> List[Transition]:
    order = np.argsort(times, kind="stable")
    times = np.asarray(times, dtype=np.float32)[order]
    values = np.asarray(values, dtype=np.float32)[order]
    result: List[Transition] = []
    active_level = 0
    pending_level = 0
    pending_time = -np.inf
    recovery_time = -np.inf
    for raw_time, raw_value in zip(times, values):
        if not np.isfinite(raw_time) or not np.isfinite(raw_value):
            continue
        t, value = float(raw_time), float(raw_value)
        level = _level(value, spec)
        if level > active_level:
            immediate = level >= spec.immediate_level or spec.confirmation_gap_min <= 0
            confirmed = (pending_level >= level and
                         t - pending_time <= spec.confirmation_gap_min)
            if immediate or confirmed:
                result.append(Transition(t, spec.levels[level - 1][1], value))
                active_level = level
                pending_level = 0
            else:
                pending_level, pending_time = level, t
            recovery_time = -np.inf
        elif level == 0:
            pending_level = 0
            if active_level and spec.recovery:
                if spec.confirmation_gap_min <= 0 or t - recovery_time <= spec.confirmation_gap_min:
                    result.append(Transition(t, spec.recovery, value))
                    active_level = 0
                    recovery_time = -np.inf
                else:
                    recovery_time = t
        else:
            pending_level = 0
            recovery_time = -np.inf
    return result


def abrupt_transitions(times: np.ndarray, values: np.ndarray, moderate: float, major: float,
                       moderate_name: str, major_name: str, direction: str,
                       lookback_min: float = 360.0) -> List[Transition]:
    order = np.argsort(times, kind="stable")
    times = np.asarray(times, dtype=np.float32)[order]
    values = np.asarray(values, dtype=np.float32)[order]
    result: List[Transition] = []
    active = False
    for index, (raw_time, raw_value) in enumerate(zip(times, values)):
        t, value = float(raw_time), float(raw_value)
        if not np.isfinite(t) or not np.isfinite(value):
            continue
        prior = (times[:index] >= t - lookback_min) & (times[:index] < t)
        if not prior.any():
            continue
        baseline = float(np.median(values[:index][prior][-12:]))
        delta = baseline - value if direction == "drop" else value - baseline
        if delta >= major and not active:
            result.append(Transition(t, major_name, value))
            active = True
        elif delta >= moderate and not active:
            result.append(Transition(t, moderate_name, value))
            active = True
        elif delta < moderate * 0.5:
            active = False
    return result


def aki_transitions(times: np.ndarray, values: np.ndarray) -> List[Transition]:
    order = np.argsort(times, kind="stable")
    times = np.asarray(times, dtype=np.float32)[order]
    values = np.asarray(values, dtype=np.float32)[order]
    result: List[Transition] = []
    baseline = np.inf
    active_level = 0
    for raw_time, raw_value in zip(times, values):
        t, value = float(raw_time), float(raw_value)
        if not np.isfinite(t) or not np.isfinite(value) or value <= 0:
            continue
        if np.isfinite(baseline):
            ratio = value / max(baseline, 0.1)
            level = (3 if ratio >= 3 or value >= 4 else
                     2 if ratio >= 2 else
                     1 if ratio >= 1.5 or value - baseline >= 0.3 else 0)
            if level > active_level:
                result.append(Transition(t, f"aki_stage_{level}_signal", value))
                active_level = level
            elif level == 0 and active_level:
                result.append(Transition(t, "aki_recovery_signal", value))
                active_level = 0
        baseline = min(baseline, value)
    return result


def extract_measurement_events(feature_arrays: Dict[str, Series]) -> List[Transition]:
    events: List[Transition] = []
    for spec in LEVEL_SPECS:
        series = merge_series([feature_arrays[name] for name in spec.sources if name in feature_arrays])
        if len(series[0]):
            events.extend(level_transitions(*series, spec))
    for sources, moderate, major, moderate_name, major_name, direction in ABRUPT_SPECS:
        series = merge_series([feature_arrays[name] for name in sources if name in feature_arrays])
        if len(series[0]):
            events.extend(abrupt_transitions(
                *series, moderate, major, moderate_name, major_name, direction))
    if "labs:creatinine" in feature_arrays:
        events.extend(aki_transitions(*feature_arrays["labs:creatinine"]))
    return events
