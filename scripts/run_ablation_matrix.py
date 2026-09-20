#!/usr/bin/env python3
"""Run patient-validation model-size and event-vocabulary ablations."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODEL_SIZES = {
    "small": (256, 6, 8, 1024),
    "current": (384, 10, 12, 1536),
    "large": (512, 12, 16, 2048),
}


def command(data_dir: Path, output_dir: Path, name: str,
            size: tuple[int, int, int, int], vocabulary_size: int,
            epochs: int) -> list[str]:
    hidden, layers, heads, ffn = size
    return [
        sys.executable, "scripts/train.py",
        "--data_dir", str(data_dir), "--output_dir", str(output_dir / name),
        "--epochs", str(epochs), "--early_stopping_patience", "25",
        "--hidden_dim", str(hidden), "--num_layers", str(layers),
        "--num_heads", str(heads), "--ffn_dim", str(ffn),
        "--vocabulary_size", str(vocabulary_size),
        "--lognormal_time_head", "--same_time_block_causal",
        "--relative_time_attention", "--event_conditioned_time_head",
        "--dynamic_windows", "--enhanced_time_encoding", "--phase_memory",
        "--observation_intensity", "--batch_size", "128",
        "--gradient_accumulation_steps", "2", "--num_workers", "4",
        "--num_buckets", "64", "--log_interval", "100",
        "--save_interval", "1000", "--precision", "bf16", "--require_cuda",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path,
                        default=Path("data/perioperative_event_sequences_v5_full"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/ablations_v5"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    experiments = []
    for name, size in MODEL_SIZES.items():
        experiments.append((f"model_{name}_vocab_full", size, 0))
    for vocabulary_size in (50, 100, 150):
        experiments.append((f"model_current_vocab_{vocabulary_size}",
                            MODEL_SIZES["current"], vocabulary_size))
    manifest = []
    for name, size, vocabulary_size in experiments:
        cmd = command(args.data_dir, args.output_dir, name, size,
                      vocabulary_size, args.epochs)
        experiment_dir = args.output_dir / name
        log_path = experiment_dir / "train.log"
        completed = log_path.exists() and any(
            "Training finished" in line for line in log_path.read_text().splitlines())
        if completed:
            # Keep the manifest reproducible, but do not spend GPU time on a
            # configuration that already reached its requested budget.
            cmd = cmd + ["--max_steps", "0"]
        elif (experiment_dir / "checkpoint_latest.pt").exists():
            cmd = cmd + ["--resume", "auto", "--reset_early_stopping"]
        manifest.append({"name": name, "model_size": size,
                         "vocabulary_size": vocabulary_size or "full",
                         "command": cmd, "completed_before_run": completed})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ablation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    for experiment in manifest:
        if experiment.get("completed_before_run"):
            print(f"Skipping completed {experiment['name']}", flush=True)
            continue
        print(f"Starting {experiment['name']}", flush=True)
        subprocess.run(experiment["command"], check=True)


if __name__ == "__main__":
    main()
