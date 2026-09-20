"""Sparse perioperative event identity and event-time prediction."""

__version__ = "4.0.0"
__author__ = "Clinical AI Research Team"

from .data.event_sequence import EventSequenceDataset, collate_event_sequences
from .models.event_hcmt import EventHCMT, event_time_loss, masked_event_loss

__all__ = [
    "EventHCMT",
    "event_time_loss",
    "masked_event_loss",
    "EventSequenceDataset",
    "collate_event_sequences",
]
