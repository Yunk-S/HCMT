"""Clinically narrow semantic families for outcome-level evaluation."""
from __future__ import annotations

from typing import Sequence

import torch

from .clinical_events import ABRUPT_SPECS, LEVEL_SPECS, OUTCOMES, SPECIAL_OUTCOMES


FAMILY_MEMBERS = {
    "systolic_bp": (
        "systolic_hypotension", "systolic_hypertension",
        "severe_systolic_hypertension", "systolic_bp_normalized",
        "acute_sbp_drop", "major_acute_sbp_drop", "acute_sbp_rise",
        "major_acute_sbp_rise"),
    "map": (
        "map_hypotension", "severe_map_hypotension", "profound_map_hypotension",
        "map_normalized", "acute_map_drop", "major_acute_map_drop"),
    "diastolic_bp": (
        "diastolic_hypotension", "diastolic_hypertension",
        "severe_diastolic_hypertension", "diastolic_bp_normalized"),
    "heart_rate": (
        "bradycardia", "severe_bradycardia", "tachycardia",
        "marked_tachycardia", "severe_tachycardia", "heart_rate_normalized",
        "acute_heart_rate_drop", "acute_heart_rate_rise"),
    "oxygen_saturation": (
        "oxygen_desaturation", "severe_hypoxemia", "oxygenation_normalized"),
    "respiratory_rate": (
        "bradypnea", "severe_bradypnea", "respiratory_rate_normalized"),
    "etco2": ("low_etco2", "severe_low_etco2", "etco2_normalized"),
    "minute_ventilation": (
        "severe_low_minute_ventilation", "minute_ventilation_normalized"),
    "oxygen_requirement": (
        "very_high_oxygen_requirement", "oxygen_requirement_reduced"),
    "temperature": ("hypothermia", "severe_hypothermia", "fever",
                    "temperature_normalized"),
    "glucose": ("hyperglycemia", "severe_hyperglycemia", "glucose_normalized"),
    "lactate": ("severe_hyperlactatemia",),
    "sodium": ("hyponatremia", "sodium_normalized"),
    "potassium": ("hypokalemia", "severe_hypokalemia", "severe_hyperkalemia",
                  "potassium_normalized"),
    "calcium": ("calcium_normalized",),
    "phosphorus": ("phosphorus_normalized",),
    "acid_base": ("severe_acidemia", "alkalemia", "ph_normalized"),
    "hemoglobin": ("anemia", "severe_anemia"),
    "hematocrit": ("low_hematocrit", "hematocrit_normalized"),
    "platelet": ("severe_thrombocytopenia",),
    "inr": ("severely_elevated_inr",),
    "aptt": ("severely_prolonged_aptt",),
    "fibrinogen": ("severely_low_fibrinogen",),
    "albumin": ("hypoalbuminemia", "albumin_normalized"),
    "white_blood_cell": ("leukocytosis", "wbc_normalized"),
    "myocardial_injury": ("marked_troponin_i_elevation",),
    "acute_kidney_injury": (
        "aki_stage_1_signal", "aki_stage_2_signal", "aki_stage_3_signal"),
    "bleeding": ("major_bleeding_signal", "severe_bleeding_signal"),
    "transfusion": (
        "rbc_transfusion", "ffp_transfusion", "platelet_transfusion",
        "cryo_transfusion"),
    "vasopressor_response": (
        "vasopressor_start", "vasopressor_stop",
        "bp_recovered_after_vasopressor_observed",
        "hypotension_persistent_after_vasopressor_observed"),
    "mechanical_ventilation": ("ventilation_start", "ventilation_stop"),
    "crrt": ("crrt_start",),
    "ecmo": ("ecmo_start",),
    "iabp": ("iabp_start",),
    "operating_room": ("or_entry", "or_exit"),
    "anesthesia_phase": ("anesthesia_start", "anesthesia_end"),
    "surgery_phase": ("surgery_start", "surgery_end"),
    "cardiopulmonary_bypass": ("cpb_start", "cpb_stop"),
    "icu_phase": ("icu_transfer", "icu_discharge"),
    "death": ("inhospital_death",),
}


OUTCOME_TO_FAMILY = {
    outcome: family
    for family, outcomes in FAMILY_MEMBERS.items()
    for outcome in outcomes
}


_SOURCE_FAMILY_ALIASES = {
    "art_sbp": "systolic_bp", "nibp_sbp": "systolic_bp",
    "art_mbp": "map", "nibp_mbp": "map",
    "art_dbp": "diastolic_bp", "nibp_dbp": "diastolic_bp",
    "hr": "heart_rate", "spo2": "oxygen_saturation", "rr": "respiratory_rate",
    "bt": "temperature", "fio2": "oxygen_requirement", "minvol": "minute_ventilation",
    "pip": "airway_pressure", "pplat": "plateau_pressure", "peep": "peep_support",
    "ci": "cardiac_index", "cvp": "cvp", "pap_mbp": "pulmonary_artery_pressure",
    "hb": "hemoglobin", "hct": "hematocrit", "ptinr": "inr",
    "ph": "acid_base", "be": "base_excess", "ica": "ionized_calcium",
    "pao2": "arterial_oxygen", "sao2": "arterial_oxygen_saturation",
    "paco2": "paco2", "total_bilirubin": "bilirubin",
    "gcs_m": "gcs_motor", "gcs_e": "gcs_eye",
}


def _source_family(source: str) -> str:
    feature = source.split(":", 1)[-1]
    return _SOURCE_FAMILY_ALIASES.get(feature, feature)


# Expand the hand-audited mappings to every ontology member.  This keeps the
# family head valid when preprocessing selects the full eligible vocabulary.
for _spec in LEVEL_SPECS:
    _family = _source_family(_spec.sources[0])
    for _, _event in _spec.levels:
        OUTCOME_TO_FAMILY.setdefault(_event, _family)
    if _spec.recovery:
        OUTCOME_TO_FAMILY.setdefault(_spec.recovery, _family)
for _sources, _, _, _moderate, _major, _ in ABRUPT_SPECS:
    _family = _source_family(_sources[0])
    OUTCOME_TO_FAMILY.setdefault(_moderate, _family)
    OUTCOME_TO_FAMILY.setdefault(_major, _family)

_SPECIAL_FAMILIES = {
    "aki_stage_1_signal": "acute_kidney_injury",
    "aki_stage_2_signal": "acute_kidney_injury",
    "aki_stage_3_signal": "acute_kidney_injury",
    "aki_recovery_signal": "acute_kidney_injury",
    "major_bleeding_signal": "bleeding", "severe_bleeding_signal": "bleeding",
    "rbc_transfusion": "transfusion", "ffp_transfusion": "transfusion",
    "platelet_transfusion": "transfusion", "cryo_transfusion": "transfusion",
    "vasopressor_start": "vasopressor_response", "vasopressor_stop": "vasopressor_response",
    "bp_recovered_after_vasopressor_observed": "vasopressor_response",
    "hypotension_persistent_after_vasopressor_observed": "vasopressor_response",
    "ventilation_start": "mechanical_ventilation", "ventilation_stop": "mechanical_ventilation",
    "crrt_start": "crrt", "crrt_stop": "crrt",
    "ecmo_start": "ecmo", "ecmo_stop": "ecmo",
    "iabp_start": "iabp", "iabp_stop": "iabp",
    "or_entry": "operating_room", "or_exit": "operating_room",
    "anesthesia_start": "anesthesia_phase", "anesthesia_end": "anesthesia_phase",
    "surgery_start": "surgery_phase", "surgery_end": "surgery_phase",
    "cpb_start": "cardiopulmonary_bypass", "cpb_stop": "cardiopulmonary_bypass",
    "icu_transfer": "icu_phase", "icu_discharge": "icu_phase",
    "inhospital_death": "death",
}
for _event in SPECIAL_OUTCOMES:
    if _event not in _SPECIAL_FAMILIES:
        raise RuntimeError(f"Special outcome lacks a semantic family: {_event}")
    OUTCOME_TO_FAMILY.setdefault(_event, _SPECIAL_FAMILIES[_event])

_UNMAPPED = [name for name in OUTCOMES if name not in OUTCOME_TO_FAMILY]
if _UNMAPPED:
    raise RuntimeError(f"Outcomes without semantic family: {_UNMAPPED}")


def family_names(outcome_names: Sequence[str]) -> list[str]:
    """Return the stable, first-seen family vocabulary for output classes."""
    return list(dict.fromkeys(OUTCOME_TO_FAMILY[name] for name in outcome_names))


def family_ids(outcome_names: Sequence[str], device=None) -> torch.Tensor:
    """Return one integer semantic-family ID for every output class."""
    missing = [name for name in outcome_names if name not in OUTCOME_TO_FAMILY]
    if missing:
        raise ValueError(f"Outcomes without semantic family: {missing}")
    ordered = family_names(outcome_names)
    index = {name: i for i, name in enumerate(ordered)}
    return torch.tensor(
        [index[OUTCOME_TO_FAMILY[name]] for name in outcome_names],
        dtype=torch.long, device=device)


def family_hit_counts(targets: torch.Tensor, top_indices: torch.Tensor,
                      ids: torch.Tensor) -> tuple[int, int]:
    """Return any-family and all-true-families hit counts for a top-k matrix."""
    num_families = int(ids.max()) + 1
    membership = torch.nn.functional.one_hot(
        ids, num_classes=num_families).to(device=targets.device, dtype=torch.float32)
    true_families = (targets.float() @ membership).bool()
    predicted = torch.zeros(
        (len(targets), num_families), dtype=torch.bool, device=targets.device)
    predicted.scatter_(1, ids[top_indices], True)
    any_hit = (true_families & predicted).any(1)
    all_hit = (~true_families | predicted).all(1)
    return int(any_hit.sum()), int(all_hit.sum())
