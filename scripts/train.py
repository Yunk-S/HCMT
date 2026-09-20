#!/usr/bin/env python3
"""Canonical trainer for the sparse perioperative event-time HCMT model."""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hcmt.data.event_sequence import EventSequenceDataset, collate_event_sequences  # noqa: E402
from hcmt.data.sampling import LengthSortedBatchSampler  # noqa: E402
from hcmt.models.event_hcmt import (  # noqa: E402
    EventHCMT, event_time_loss, masked_event_loss, masked_value_loss,
)
from hcmt.data.outcome_families import family_hit_counts, family_ids  # noqa: E402


LOG = logging.getLogger("hcmt.event_training")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HCMT to predict the next perioperative event and its time")
    parser.add_argument("--data_dir", default="data/perioperative_event_sequences_v5_full")
    parser.add_argument("--output_dir", default="outputs/event_hcmt_v5_full")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Maximum budget; independent validation early stopping may finish earlier")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--window_stride", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sampler_type", choices=("bucket", "random"), default="bucket")
    parser.add_argument("--num_buckets", type=int, default=64,
                        help="Random-pool multiplier used by the length-bucket sampler")
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--ffn_dim", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lr_plateau_patience", type=int, default=8)
    parser.add_argument("--lr_plateau_factor", type=float, default=0.3)
    parser.add_argument("--time_loss_weight", type=float, default=1.0)
    parser.add_argument("--trajectory_loss_weight", type=float, default=0.5)
    parser.add_argument("--masked_event_loss_weight", type=float, default=0.2)
    parser.add_argument("--masked_value_loss_weight", type=float, default=0.1)
    parser.add_argument("--masked_event_probability", type=float, default=0.15)
    parser.add_argument("--masked_event_every", type=int, default=4,
                        help="Run the bidirectional masked-event task every N batches")
    parser.add_argument("--initial_event_interval_hours", type=float, default=24.0)
    parser.add_argument("--decoupled_time_head", action="store_true",
                        help="Predict event identity and total event rate with separate heads")
    parser.add_argument("--enhanced_time_encoding", action="store_true",
                        help="Add monotonic log-time features to Fourier continuous time encoding")
    parser.add_argument("--lognormal_time_head", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use a stable censored log-normal waiting-time head")
    parser.add_argument("--family_loss_weight", type=float, default=0.25)
    parser.add_argument("--same_time_block_causal", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--relative_time_attention", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--event_conditioned_time_head", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dynamic_windows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observation_intensity", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--minimum_context_fraction", type=float, default=0.75)
    parser.add_argument("--vocabulary_size", type=int, choices=(0, 50, 100, 150), default=0,
                        help="0 uses every eligible event; other values support vocabulary ablation")
    parser.add_argument("--validation_fraction", type=float, default=0.1,
                        help="Independent patient-level validation fraction")
    parser.add_argument("--validation_max_windows", type=int, default=0,
                        help="Optional deterministic validation window cap; zero uses all")
    parser.add_argument("--max_wait_hours", type=float, default=24.0 * 38)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=25)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--keep_epoch_checkpoint_interval", type=int, default=25,
                        help="Keep a named epoch checkpoint every N epochs; latest/best are always kept")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint path or 'auto' for output_dir/checkpoint_latest.pt")
    parser.add_argument("--reset_early_stopping", action="store_true",
                        help="When resuming, reset the stale-validation counter while keeping model/optimizer/scheduler state")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Optional smoke-test limit; zero means full epochs")
    return parser.parse_args()


class WindowSubset(Dataset):
    """Subset that preserves O(1) sequence lengths for bucket sampling."""

    def __init__(self, dataset: EventSequenceDataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]

    def sequence_length(self, index):
        return self.dataset.sequence_length(int(self.indices[index]))


def setup_logging(output_dir: Path, append: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(output_dir / "train.log", mode="a" if append else "w")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimizer_for(model: torch.nn.Module, lr: float, weight_decay: float, device: torch.device):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "embedding" in name or "norm" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {}
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), **kwargs)


class WarmupPlateauScheduler:
    """Linear warmup followed by validation-driven late-stage LR reductions."""

    def __init__(self, optimizer, warmup_steps: int, min_lr: float,
                 patience: int, factor: float):
        self.optimizer = optimizer
        self.warmup_steps = max(1, int(warmup_steps))
        self.min_lr = float(min_lr)
        self.patience = max(1, int(patience))
        self.factor = float(factor)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * 1e-4
        self.step_count = 0
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            ratio = max(1e-4, self.step_count / self.warmup_steps)
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * ratio

    def step_validation(self, metric: float):
        if metric < self.best - 1e-8:
            self.best, self.bad_epochs = float(metric), 0
            return False
        self.bad_epochs += 1
        if self.step_count < self.warmup_steps or self.bad_epochs < self.patience:
            return False
        changed = False
        for group in self.optimizer.param_groups:
            reduced = max(self.min_lr, group["lr"] * self.factor)
            changed |= reduced < group["lr"]
            group["lr"] = reduced
        self.bad_epochs = 0
        return changed

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()
                if key != "optimizer"}

    def load_state_dict(self, state):
        for key, value in state.items():
            if key != "optimizer":
                setattr(self, key, value)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def aggregate(metrics, result, batch_tokens: int):
    n = int(result["num_targets"].item())
    time_n = int(result["num_time_targets"].item())
    time_mae_n = int(result["num_time_mae_targets"].item())
    nonstable_n = int(result["num_nonstable_targets"].item())
    metrics["targets"] += n
    metrics["time_targets"] += time_n
    metrics["time_mae_targets"] += time_mae_n
    metrics["nonstable_targets"] += nonstable_n
    metrics["tokens"] += batch_tokens
    if n:
        for key in ("event_loss", "family_loss", "event_accuracy", "trajectory_loss"):
            metrics[key] += float(result[key].detach()) * n
        metrics["stable_prediction_fraction"] += float(result["stable_prediction_fraction"]) * n
    if time_n:
        metrics["time_loss"] += float(result["time_loss"].detach()) * time_n
    if time_mae_n:
        metrics["time_mae_hours"] += float(result["time_mae_hours"].detach()) * time_mae_n
    if nonstable_n:
        metrics["nonstable_accuracy"] += float(result["nonstable_accuracy"]) * nonstable_n


def summarize(metrics):
    event_denom = max(1, metrics["targets"])
    time_denom = max(1, metrics["time_targets"])
    time_mae_denom = max(1, metrics["time_mae_targets"])
    result = {key: metrics[key] / event_denom for key in
              ("event_loss", "family_loss", "trajectory_loss", "event_accuracy",
               "stable_prediction_fraction")}
    result["time_loss"] = metrics["time_loss"] / time_denom
    result["time_mae_hours"] = metrics["time_mae_hours"] / time_mae_denom
    result["masked_loss"] = metrics.get("masked_loss", 0.0) / max(1, metrics.get("masked_batches", 0))
    result["masked_value_loss"] = metrics.get("masked_value_loss", 0.0) / max(
        1, metrics.get("masked_value_batches", 0))
    result["loss"] = (result["event_loss"] +
                      metrics.get("family_weight", 0.0) * result["family_loss"] +
                      metrics["time_weight"] * result["time_loss"] +
                      metrics.get("trajectory_weight", 0.0) * result["trajectory_loss"] +
                      metrics.get("masked_weight", 0.0) * result["masked_loss"] +
                      metrics.get("masked_value_weight", 0.0) * result["masked_value_loss"])
    result["nonstable_accuracy"] = metrics["nonstable_accuracy"] / max(1, metrics["nonstable_targets"])
    return result


@torch.no_grad()
def monitor(model, loader, device, amp_enabled, amp_dtype, time_weight, outcome_names,
            max_wait_hours, stable_index, trajectory_weight, class_weights,
            family_weight):
    model.eval()
    totals = {key: 0.0 for key in
              ("event_loss", "family_loss", "time_loss", "trajectory_loss", "event_accuracy", "time_mae_hours",
               "nonstable_accuracy", "stable_prediction_fraction")}
    totals.update(targets=0, time_targets=0, time_mae_targets=0,
                  nonstable_targets=0, tokens=0,
                  time_weight=time_weight, family_weight=family_weight,
                  trajectory_weight=trajectory_weight,
                  masked_weight=0.0, masked_loss=0.0, masked_batches=0)
    totals.update(masked_value_weight=0.0, masked_value_loss=0.0,
                  masked_value_batches=0)
    class_positive = torch.zeros(len(outcome_names), dtype=torch.long)
    class_hit = torch.zeros(len(outcome_names), dtype=torch.long)
    class_hit5 = torch.zeros(len(outcome_names), dtype=torch.long)
    class_hit10 = torch.zeros(len(outcome_names), dtype=torch.long)
    hit5 = hit10 = hit_all = 0
    family_hit5 = family_hit10 = family_hit_all = 0
    outcome_family_ids = family_ids(outcome_names, device)
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(batch)
            result = event_time_loss(
                output.logits, batch["target_set"], batch["target_dt_hours"],
                batch["loss_mask"], batch["time_mask"], time_weight,
                max_wait_hours, stable_index, output.log_total_rate,
                output.time_mu, output.time_log_sigma,
                output.trajectory_logits, batch["trajectory_target"],
                batch["trajectory_mask"], trajectory_weight, class_weights,
                output.family_logits, outcome_family_ids, family_weight,
                output.family_time_mu, output.family_time_log_sigma,
            )
        valid = batch["loss_mask"].bool() & (batch["target_set"].sum(-1) > 0)
        targets = batch["target_set"][valid].bool()
        valid_logits = output.logits[valid]
        top = valid_logits.argmax(-1)
        top10 = valid_logits.topk(min(10, valid_logits.shape[-1]), dim=-1).indices
        top5 = top10[:, :min(5, top10.shape[1])]
        predicted5 = torch.zeros_like(targets).scatter_(1, top5, True)
        predicted10 = torch.zeros_like(targets).scatter_(1, top10, True)
        hit5 += int((targets & predicted5).any(1).sum())
        hit10 += int((targets & predicted10).any(1).sum())
        hit_all += int((~targets | predicted10).all(1).sum())
        batch_family_hit5, _ = family_hit_counts(targets, top5, outcome_family_ids)
        batch_family_hit10, batch_family_hit_all = family_hit_counts(
            targets, top10, outcome_family_ids)
        family_hit5 += batch_family_hit5
        family_hit10 += batch_family_hit10
        family_hit_all += batch_family_hit_all
        class_positive += targets.sum(0).long().cpu()
        class_hit5 += (targets & predicted5).sum(0).long().cpu()
        class_hit10 += (targets & predicted10).sum(0).long().cpu()
        for class_index in range(len(outcome_names)):
            class_hit[class_index] += int(((top == class_index) & (targets[:, class_index] > 0)).sum().cpu())
        aggregate(totals, result, int(batch["attention_mask"].sum()))
    summary = summarize(totals)
    target_count = max(1, totals["targets"])
    summary["next_event_hit_at_5"] = hit5 / target_count
    summary["next_event_hit_at_10"] = hit10 / target_count
    summary["all_true_events_hit_at_10"] = hit_all / target_count
    summary["same_family_hit_at_5"] = family_hit5 / target_count
    summary["same_family_hit_at_10"] = family_hit10 / target_count
    summary["all_true_families_hit_at_10"] = family_hit_all / target_count
    summary["outcome_hit_at_1"] = {
        name: (float(class_hit[index] / class_positive[index]) if class_positive[index] else None)
        for index, name in enumerate(outcome_names)
    }
    summary["outcome_hit_at_5"] = {
        name: (float(class_hit5[index] / class_positive[index])
               if class_positive[index] else None)
        for index, name in enumerate(outcome_names)
    }
    summary["outcome_hit_at_10"] = {
        name: (float(class_hit10[index] / class_positive[index])
               if class_positive[index] else None)
        for index, name in enumerate(outcome_names)
    }
    return summary


def make_masked_batch(batch: Dict[str, torch.Tensor], mask_token_id: int,
                      vocabulary_size: int, probability: float):
    """BERT-style 80/10/10 corruption without exposing masked numeric values."""
    masked = {key: value for key, value in batch.items()}
    eligible = batch["attention_mask"].bool() & (batch["token_kind"] > 1)
    selected = eligible & (torch.rand_like(batch["value"]) < float(probability))
    if not selected.any() and eligible.any():
        selected.view(-1)[eligible.view(-1).nonzero()[0]] = True
    labels = torch.full_like(batch["token_id"], -100)
    labels[selected] = batch["token_id"][selected]
    corrupted = batch["token_id"].clone()
    draw = torch.rand_like(batch["value"])
    corrupted[selected & (draw < 0.8)] = int(mask_token_id)
    random_positions = selected & (draw >= 0.8) & (draw < 0.9)
    corrupted[random_positions] = torch.randint(
        1, vocabulary_size, (int(random_positions.sum()),), device=corrupted.device)
    masked["token_id"] = corrupted
    masked["value"] = batch["value"].masked_fill(selected, 0.0)
    masked["has_value"] = batch["has_value"].masked_fill(selected, 0.0)
    value_mask = selected & batch["has_value"].bool()
    return masked, labels, batch["value"], value_mask


def checkpoint_payload(model, optimizer, scheduler, scaler, args, dataset, epoch,
                       global_step, best_monitor, stale_epochs, batch_in_epoch=0):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    return {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_monitor": best_monitor,
        "stale_epochs": stale_epochs,
        "batch_in_epoch": int(batch_in_epoch),
        "args": vars(args),
        "dataset_meta": dataset.meta,
    }


def atomic_save(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    resume_path = None
    if args.resume:
        resume_path = output_dir / "checkpoint_latest.pt" if args.resume == "auto" else Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    setup_logging(output_dir, append=resume_path is not None)
    history_path = output_dir / "validation_history.jsonl"
    if resume_path is None:
        history_path.write_text("")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        if args.require_cuda:
            raise RuntimeError("CUDA was required but is unavailable in this process")
        LOG.warning("CUDA unavailable; falling back to CPU for verification")
        requested = torch.device("cpu")
    device = requested

    dataset = EventSequenceDataset(
        args.data_dir, args.block_size, args.window_stride,
        dynamic_windows=args.dynamic_windows, seed=args.seed,
        minimum_context_fraction=args.minimum_context_fraction,
        outcome_limit=args.vocabulary_size)
    validation_dataset = EventSequenceDataset(
        args.data_dir, args.block_size, args.window_stride,
        dynamic_windows=False, seed=args.seed,
        outcome_limit=args.vocabulary_size)
    if dataset.split != "all_train":
        raise ValueError(
            f"Training requires split='all_train'; refusing {dataset.split!r} to prevent validation leakage")
    if "stable_interval" in dataset.meta["outcome_vocabulary"]:
        raise ValueError(
            "Canonical event training rejects stable_interval as a prediction target; "
            "rebuild with scripts/preprocess_event_sequences.py (v3).")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    admissions = pd.read_csv(Path(args.data_dir) / "admissions.csv", usecols=["subject_id"])
    if len(admissions) != len(dataset.ptr) - 1:
        raise ValueError("admissions.csv does not align with event sequences")
    patient_ids = admissions["subject_id"].astype(str).to_numpy()
    unique_patients = np.unique(patient_ids)
    if len(unique_patients) < 2:
        raise ValueError("Patient-level validation requires at least two independent patients")
    split_rng = np.random.default_rng(args.seed)
    shuffled_patients = split_rng.permutation(unique_patients)
    validation_patient_count = max(1, int(round(
        len(shuffled_patients) * args.validation_fraction)))
    validation_patients = set(shuffled_patients[:validation_patient_count].tolist())
    validation_episode = np.asarray(
        [patient in validation_patients for patient in patient_ids], dtype=bool)
    train_indices = np.flatnonzero(~validation_episode[dataset.window_admission])
    validation_indices = np.flatnonzero(
        validation_episode[validation_dataset.window_admission])
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Patient split produced an empty training or validation window set")
    if args.validation_max_windows and len(validation_indices) > args.validation_max_windows:
        cap_rng = np.random.default_rng(args.seed + 991)
        validation_indices = np.sort(cap_rng.choice(
            validation_indices, args.validation_max_windows, replace=False))
    split_path = output_dir / "patient_validation_split.npz"
    np.savez_compressed(
        split_path,
        train_patients=np.asarray(sorted(set(unique_patients) - validation_patients)),
        validation_patients=np.asarray(sorted(validation_patients)),
        train_window_indices=train_indices,
        validation_window_indices=validation_indices,
    )
    train_dataset = WindowSubset(dataset, train_indices)
    validation_subset = WindowSubset(validation_dataset, validation_indices)

    generator = torch.Generator().manual_seed(args.seed)
    train_sampler = None
    loader_kwargs = dict(
        dataset=train_dataset, num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0 and not args.dynamic_windows,
        prefetch_factor=4 if args.num_workers else None,
        collate_fn=collate_event_sequences,
    )
    if args.sampler_type == "bucket":
        train_sampler = LengthSortedBatchSampler(
            train_dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed,
            drop_last=False, num_buckets=args.num_buckets,
        )
        loader = DataLoader(batch_sampler=train_sampler, **loader_kwargs)
    else:
        loader = DataLoader(
            batch_size=args.batch_size, shuffle=True, generator=generator,
            drop_last=False, **loader_kwargs,
        )
    monitor_loader = DataLoader(
        validation_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda", collate_fn=collate_event_sequences,
    )

    outcome_family_ids = torch.as_tensor(
        dataset.meta.get("outcome_to_family", family_ids(
            dataset.meta["outcome_vocabulary"]).tolist()), dtype=torch.long)
    model = EventHCMT(
        dataset.num_tokens, dataset.num_outcomes, dataset.num_static,
        args.hidden_dim, args.num_layers, args.num_heads, args.ffn_dim, args.dropout,
        args.initial_event_interval_hours, args.decoupled_time_head,
        args.enhanced_time_encoding, len(dataset.trajectory_horizons_hours),
        args.lognormal_time_head,
        outcome_family_ids=outcome_family_ids,
        same_time_block_causal=args.same_time_block_causal,
        relative_time_attention=args.relative_time_attention,
        event_conditioned_time_head=args.event_conditioned_time_head,
        phase_memory=args.phase_memory,
        observation_intensity=args.observation_intensity,
        value_reconstruction=args.masked_value_loss_weight > 0,
    ).to(device)
    counts = torch.tensor([
        max(1, int(dataset.meta.get("audit_counts", {}).get(name, 1)))
        for name in dataset.meta["outcome_vocabulary"]
    ], dtype=torch.float32, device=device)
    class_weights = torch.sqrt(counts.sum() / (len(counts) * counts)).clamp(0.5, 5.0)
    class_weights = class_weights / class_weights.mean()
    mask_token_id = dataset.meta["token_vocabulary"].get("<MASK>")
    if (args.masked_event_loss_weight > 0 or args.masked_value_loss_weight > 0) and mask_token_id is None:
        raise ValueError("Masked-event training requires a v4 dataset with a <MASK> token")
    optimizer = optimizer_for(model, args.lr, args.weight_decay, device)
    scheduler = WarmupPlateauScheduler(
        optimizer, args.warmup_steps, args.min_lr,
        args.lr_plateau_patience, args.lr_plateau_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    amp_enabled = device.type == "cuda" and args.precision != "fp32"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    start_epoch, global_step, best_monitor, stale_epochs = 0, 0, float("inf"), 0
    resume_batch_in_epoch = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_monitor = float(checkpoint.get("best_monitor", best_monitor))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        resume_batch_in_epoch = int(checkpoint.get("batch_in_epoch", 0))
        if args.reset_early_stopping:
            stale_epochs = 0
            LOG.info("Reset stale-validation counter for resumed training")
        LOG.info("Resumed %s at epoch=%d batch=%d optimizer_step=%d",
                 resume_path, start_epoch, resume_batch_in_epoch, global_step)

    if args.compile:
        model = torch.compile(model)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOG.info("Event HCMT | device=%s parameters=%.2fM admissions=%d train_windows=%d validation_windows=%d tokens=%d",
             device, parameter_count / 1e6, dataset.meta["num_admissions"], len(train_dataset),
             len(validation_subset),
             dataset.meta["num_tokens"])
    LOG.info("Independent patient split: train_patients=%d validation_patients=%d validation_fraction=%.3f",
             len(unique_patients) - len(validation_patients), len(validation_patients),
             args.validation_fraction)
    LOG.info("Outcomes: %s", ", ".join(dataset.meta["outcome_vocabulary"]))
    stable_index = (dataset.meta["outcome_vocabulary"].index("stable_interval")
                    if "stable_interval" in dataset.meta["outcome_vocabulary"] else -1)
    LOG.info("Batch sampler=%s batch_size=%d pool_multiplier=%s",
             args.sampler_type, args.batch_size,
             args.num_buckets if args.sampler_type == "bucket" else "n/a")
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    stop = False
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        if train_sampler is not None:
            start_batch = resume_batch_in_epoch if epoch == start_epoch else 0
            train_sampler.set_epoch_batch(epoch, start_batch)
        elif resume_batch_in_epoch and epoch == start_epoch:
            raise ValueError("Exact mid-epoch resume requires --sampler_type bucket")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = {key: 0.0 for key in
                   ("event_loss", "family_loss", "time_loss", "trajectory_loss", "event_accuracy", "time_mae_hours",
                    "nonstable_accuracy", "stable_prediction_fraction")}
        running.update(targets=0, time_targets=0, time_mae_targets=0,
                       nonstable_targets=0, tokens=0,
                       time_weight=args.time_loss_weight,
                       family_weight=args.family_loss_weight,
                       trajectory_weight=args.trajectory_loss_weight,
                       masked_weight=args.masked_event_loss_weight,
                       masked_loss=0.0, masked_batches=0,
                       masked_value_weight=args.masked_value_loss_weight,
                       masked_value_loss=0.0, masked_value_batches=0)
        interval_started = time.time()
        max_steps_reached = False
        batches_completed = resume_batch_in_epoch

        for batch_index, batch in enumerate(loader, start=1):
            batches_completed = resume_batch_in_epoch + batch_index
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(batch)
                result = event_time_loss(
                    output.logits, batch["target_set"], batch["target_dt_hours"],
                    batch["loss_mask"], batch["time_mask"], args.time_loss_weight,
                    args.max_wait_hours, stable_index, output.log_total_rate,
                    output.time_mu, output.time_log_sigma,
                    output.trajectory_logits, batch["trajectory_target"],
                    batch["trajectory_mask"], args.trajectory_loss_weight,
                    class_weights,
                    output.family_logits, outcome_family_ids.to(device),
                    args.family_loss_weight,
                    output.family_time_mu, output.family_time_log_sigma,
                )
                scaled_main_loss = result["loss"] / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(scaled_main_loss).backward()
            else:
                scaled_main_loss.backward()

            mlm_loss = output.logits.new_zeros(())
            if ((args.masked_event_loss_weight > 0 or args.masked_value_loss_weight > 0) and
                    batch_index % max(1, args.masked_event_every) == 0):
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    masked_batch, masked_labels, value_labels, value_mask = make_masked_batch(
                        batch, mask_token_id, dataset.num_tokens,
                        args.masked_event_probability)
                    masked_output = model(
                        masked_batch, causal=False, return_token_logits=True,
                        masked_only=True)
                    mlm_loss = masked_event_loss(masked_output.token_logits, masked_labels)
                    value_loss = masked_value_loss(
                        masked_output.value_prediction, value_labels, value_mask)
                    scaled_mlm_loss = ((args.masked_event_loss_weight * mlm_loss +
                                        args.masked_value_loss_weight * value_loss) /
                                       args.gradient_accumulation_steps)
                if scaler.is_enabled():
                    scaler.scale(scaled_mlm_loss).backward()
                else:
                    scaled_mlm_loss.backward()
                running["masked_loss"] += float(mlm_loss.detach())
                running["masked_batches"] += 1
                running["masked_value_loss"] += float(value_loss.detach())
                running["masked_value_batches"] += 1
            aggregate(running, result, int(batch["attention_mask"].sum()))

            should_step = (batch_index % args.gradient_accumulation_steps == 0 or
                           batch_index == len(loader))
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient at optimizer step {global_step}")
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.log_interval == 0:
                    summary = summarize(running)
                    elapsed = max(time.time() - interval_started, 1e-6)
                    gpu_gb = (torch.cuda.max_memory_allocated(device) / 2**30
                              if device.type == "cuda" else 0.0)
                    LOG.info(
                        "epoch=%d batch=%d/%d progress=%.1f%% step=%d "
                        "loss=%.4f event_loss=%.4f family_loss=%.4f time_loss=%.4f trajectory_loss=%.4f "
                        "masked_loss=%.4f masked_value_loss=%.4f hit@1=%.3f "
                        "nonstable_hit@1=%.3f predicted_stable=%.3f time_MAE_h=%.3f "
                        "lr=%.3e tokens/s=%.0f gpu_peak_GB=%.2f",
                        epoch + 1, batch_index, len(loader), 100 * batch_index / len(loader),
                        global_step, summary["loss"], summary["event_loss"],
                        summary["family_loss"], summary["time_loss"], summary["trajectory_loss"],
                        summary["masked_loss"], summary["masked_value_loss"],
                        summary["event_accuracy"],
                        summary["nonstable_accuracy"], summary["stable_prediction_fraction"],
                        summary["time_mae_hours"], optimizer.param_groups[0]["lr"],
                        running["tokens"] / elapsed, gpu_gb,
                    )
                    for key in running:
                        if key not in {"time_weight", "family_weight", "trajectory_weight",
                                       "masked_weight", "masked_value_weight"}:
                            running[key] = 0
                    interval_started = time.time()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)

                if global_step % args.save_interval == 0:
                    payload = checkpoint_payload(model, optimizer, scheduler, scaler, args,
                                                 dataset, epoch, global_step,
                                                 best_monitor, stale_epochs,
                                                 resume_batch_in_epoch + batch_index)
                    atomic_save(payload, output_dir / "checkpoint_latest.pt")

                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    max_steps_reached = True
                    break

        fit_metrics = monitor(model, monitor_loader, device, amp_enabled,
                              amp_dtype, args.time_loss_weight,
                              dataset.meta["outcome_vocabulary"],
                              args.max_wait_hours, stable_index,
                              args.trajectory_loss_weight, class_weights,
                              args.family_loss_weight)
        LOG.info(
            "epoch=%d VALIDATION loss=%.4f event_loss=%.4f family_loss=%.4f time_loss=%.4f "
            "trajectory_loss=%.4f "
            "hit@1=%.3f hit@5=%.3f hit@10=%.3f family_hit@5=%.3f family_hit@10=%.3f "
            "nonstable_hit@1=%.3f predicted_stable=%.3f "
            "time_MAE_h=%.3f",
            epoch + 1, fit_metrics["loss"], fit_metrics["event_loss"],
            fit_metrics["family_loss"], fit_metrics["time_loss"], fit_metrics["trajectory_loss"],
            fit_metrics["event_accuracy"],
            fit_metrics["next_event_hit_at_5"], fit_metrics["next_event_hit_at_10"],
            fit_metrics["same_family_hit_at_5"], fit_metrics["same_family_hit_at_10"],
            fit_metrics["nonstable_accuracy"], fit_metrics["stable_prediction_fraction"],
            fit_metrics["time_mae_hours"],
        )
        LOG.info("VALIDATION_OUTCOME_HIT_AT_1 %s",
                 json.dumps(fit_metrics["outcome_hit_at_1"], ensure_ascii=False, sort_keys=True))
        with history_path.open("a") as history_file:
            history_file.write(json.dumps({
                "epoch": epoch + 1, "optimizer_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{key: value for key, value in fit_metrics.items()
                   if not isinstance(value, dict)},
            }, ensure_ascii=False) + "\n")
        if scheduler.step_validation(fit_metrics["loss"]):
            LOG.info("Validation plateau: reduced learning rate to %.3e",
                     optimizer.param_groups[0]["lr"])
        improved = fit_metrics["loss"] < best_monitor - args.early_stopping_min_delta
        if improved:
            best_monitor = fit_metrics["loss"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint_epoch = epoch if max_steps_reached else epoch + 1
        checkpoint_batch = batches_completed if max_steps_reached else 0
        payload = checkpoint_payload(model, optimizer, scheduler, scaler, args,
                                     dataset, checkpoint_epoch, global_step,
                                     best_monitor, stale_epochs, checkpoint_batch)
        atomic_save(payload, output_dir / "checkpoint_latest.pt")
        keep_epoch = (args.keep_epoch_checkpoint_interval > 0 and
                      (epoch + 1) % args.keep_epoch_checkpoint_interval == 0)
        if not max_steps_reached and (keep_epoch or epoch + 1 == args.epochs):
            atomic_save(payload, output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt")
        if improved and not max_steps_reached:
            atomic_save(payload, output_dir / "best_model.pt")
        if stale_epochs >= args.early_stopping_patience:
            LOG.info("Early stopping after %d stale independent-validation epochs", stale_epochs)
            stop = True
        if stop:
            break
        resume_batch_in_epoch = 0

    LOG.info("Training finished at optimizer_step=%d best_validation_loss=%.4f",
             global_step, best_monitor)


if __name__ == "__main__":
    main()
