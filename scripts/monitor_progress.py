#!/usr/bin/env python3
"""One-shot status reader for event preprocessing or training logs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TRAIN_PATTERN = re.compile(
    r"epoch=(?P<epoch>\d+) batch=(?P<batch>\d+)/(?P<batches>\d+) progress=(?P<progress>[\d.]+)% "
    r"step=(?P<step>\d+) loss=(?P<loss>[\d.eE+-]+) event_loss=(?P<event>[\d.eE+-]+) "
    r"time_loss=(?P<time>[\d.eE+-]+) trajectory_loss=(?P<trajectory>[\d.eE+-]+) "
    r"masked_loss=(?P<masked>[\d.eE+-]+) hit@1=(?P<hit>[\d.eE+-]+) "
    r"nonstable_hit@1=(?P<clinical_hit>[\d.eE+-]+) predicted_stable=(?P<predicted_stable>[\d.eE+-]+) "
    r"time_MAE_h=(?P<mae>[\d.eE+-]+).*tokens/s=(?P<speed>[\d.eE+-]+) "
    r"gpu_peak_GB=(?P<gpu>[\d.eE+-]+)"
)
FINISHED_PATTERN = re.compile(
    r"Training finished at optimizer_step=(?P<step>\d+) "
    r"best_training_fit_loss=(?P<best_loss>[\d.eE+-]+)"
)


def tail(path: Path, lines: int = 200):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(errors="replace").splitlines()[-lines:]


def training_status(path: Path):
    rows = tail(path)
    for line in reversed(rows):
        match = FINISHED_PATTERN.search(line)
        if match:
            evaluation_path = path.parent / "train_fit_evaluation.json"
            result = {"status": "complete", **match.groupdict()}
            if evaluation_path.is_file():
                result["evaluation"] = json.loads(evaluation_path.read_text())
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
    for line in reversed(rows):
        match = TRAIN_PATTERN.search(line)
        if match:
            result = match.groupdict()
            print(json.dumps({"status": "training", **result}, ensure_ascii=False, indent=2))
            return
    monitor = next((line for line in reversed(rows) if "TRAIN_FIT_MONITOR" in line), None)
    if monitor:
        print(json.dumps({"status": "epoch_monitor", "line": monitor}, ensure_ascii=False, indent=2))
        return
    print(json.dumps({"status": "no_progress_line", "last_lines": rows[-10:]}, ensure_ascii=False, indent=2))


def preprocessing_status(path: Path):
    rows = tail(path)
    for line in reversed(rows):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "progress_percent" in value or value.get("complete"):
            value["status"] = "complete" if value.get("complete") else "preprocessing"
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return
    print(json.dumps({"status": "no_progress_line", "last_lines": rows[-10:]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--mode", choices=("auto", "train", "preprocess"), default="auto")
    args = parser.parse_args()
    path = Path(args.log)
    mode = args.mode
    if mode == "auto":
        mode = "train" if path.name == "train.log" else "preprocess"
    (training_status if mode == "train" else preprocessing_status)(path)


if __name__ == "__main__":
    main()
