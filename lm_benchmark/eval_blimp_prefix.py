#!/usr/bin/env python3
"""
BLiMP prefix accuracy evaluation.
Instead of comparing full sentence probabilities, this compares
probabilities at the exact point where the minimal pair diverges.

1-prefix: probability of just the first diverging token
2-prefix: probability of first two diverging tokens
"""

import os
import sys
import json
import csv
import torch
import tiktoken
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/landformer/LandFORMER/lm_benchmark"))

import config
import models


def get_args():
    parser = argparse.ArgumentParser(description="BLiMP prefix evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--blimp_dir", type=str, default=os.path.expanduser("~/landformer/blimp/data"))
    parser.add_argument("--output_csv_1prefix", type=str, default=None)
    parser.add_argument("--output_csv_2prefix", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def load_model(checkpoint_dir, device):
    checkpoint_path = checkpoint_dir
    checkpoint_filename = "ckpt.pt"
    if os.path.isfile(checkpoint_path):
        checkpoint_path, checkpoint_filename = os.path.split(checkpoint_path)
    with open(os.path.join(checkpoint_path, "summary.json")) as f:
        summary = json.load(f)
    args_dict = summary["args"]
    class Args:
        pass
    model_args = Args()
    for k, v in args_dict.items():
        setattr(model_args, k, v)
    model_args.device = torch.device(device)
    model_args.dtype = torch.float16
    if not hasattr(model_args, 'use_cache'):
        model_args.use_cache = False
    if not hasattr(model_args, 'lm_cache'):
        model_args.lm_cache = 'none'
    model = models.make_model_from_args(model_args).to(model_args.device)
    ckpt = torch.load(os.path.join(checkpoint_path, checkpoint_filename), map_location="cpu")
    model.load_state_dict(
        {k: v for k, v in ckpt["model"].items() if "attn.bias" not in k and "wpe" not in k},
        strict=False
    )
    model.eval()
    print(f"  Loaded model from {checkpoint_path}")
    print(f"  Landmark selection: {getattr(model_args, 'landmark_selection', 'positional')}")
    return model, model_args


def find_prefix_length(good_ids, bad_ids):
    """Find the length of the common prefix between two token sequences."""
    prefix_len = 0
    for i in range(min(len(good_ids), len(bad_ids))):
        if good_ids[i] == bad_ids[i]:
            prefix_len += 1
        else:
            break
    return prefix_len


@torch.no_grad()
def get_logprobs_at_positions(model, token_ids, positions, device, dtype):
    """
    Get log probabilities at specific positions.
    Returns list of log P(token_ids[pos] | token_ids[:pos]) for each pos in positions.
    """
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    dummy_targets = torch.zeros_like(x)
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        outputs = model(x, targets=dummy_targets, get_logits=True, use_cache=False)
    logits = outputs["logits"]
    log_probs = torch.log_softmax(logits, dim=-1)
    result = []
    for pos in positions:
        if pos < len(token_ids) and pos > 0:
            result.append(log_probs[0, pos - 1, token_ids[pos]].item())
        else:
            result.append(float('-inf'))
    return result


def evaluate_blimp_prefix(model, model_args, dataset_path, tokenizer, device):
    """
    Evaluate one BLiMP dataset using prefix accuracy.
    Returns (acc_1prefix, correct_1, acc_2prefix, correct_2, total).
    """
    correct_1 = 0
    correct_2 = 0
    total = 0

    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line)
            good_ids = tokenizer.encode(item["sentence_good"])
            bad_ids = tokenizer.encode(item["sentence_bad"])

            if len(good_ids) < 2 or len(bad_ids) < 2:
                continue

            prefix_len = find_prefix_length(good_ids, bad_ids)

            # Need at least one diverging token
            if prefix_len >= len(good_ids) or prefix_len >= len(bad_ids):
                continue

            # 1-prefix: compare P(first diverging token | prefix)
            pos_1 = prefix_len
            good_lp_1 = get_logprobs_at_positions(model, good_ids, [pos_1], device, model_args.dtype)
            bad_lp_1 = get_logprobs_at_positions(model, bad_ids, [pos_1], device, model_args.dtype)

            if good_lp_1[0] > bad_lp_1[0]:
                correct_1 += 1

            # 2-prefix: compare sum of P(first two diverging tokens | prefix)
            pos_2 = prefix_len + 1
            if pos_2 < len(good_ids) and pos_2 < len(bad_ids):
                good_lp_2 = get_logprobs_at_positions(model, good_ids, [pos_1, pos_2], device, model_args.dtype)
                bad_lp_2 = get_logprobs_at_positions(model, bad_ids, [pos_1, pos_2], device, model_args.dtype)
                if sum(good_lp_2) > sum(bad_lp_2):
                    correct_2 += 1
                else:
                    pass  # incorrect
            else:
                # Only one diverging token, use 1-prefix result for 2-prefix too
                if good_lp_1[0] > bad_lp_1[0]:
                    correct_2 += 1

            total += 1

    acc_1 = correct_1 / total if total > 0 else 0.0
    acc_2 = correct_2 / total if total > 0 else 0.0
    return acc_1, correct_1, acc_2, correct_2, total


def main():
    args = get_args()
    tokenizer = tiktoken.get_encoding("gpt2")
    blimp_files = sorted(Path(args.blimp_dir).glob("*.jsonl"))
    if len(blimp_files) == 0:
        print(f"ERROR: No .jsonl files found in {args.blimp_dir}")
        sys.exit(1)
    print(f"Found {len(blimp_files)} BLiMP datasets")
    model, model_args = load_model(args.checkpoint, args.device)

    results_1 = []
    results_2 = []

    print(f"\n{'Dataset':<55} {'1-Pfx':>7} {'2-Pfx':>7} {'Total':>6}")
    print("-" * 80)

    for dataset_path in blimp_files:
        dataset_name = dataset_path.stem
        acc_1, corr_1, acc_2, corr_2, total = evaluate_blimp_prefix(
            model, model_args, dataset_path, tokenizer, args.device
        )
        results_1.append({"dataset": dataset_name, "accuracy": acc_1, "correct": corr_1, "total": total})
        results_2.append({"dataset": dataset_name, "accuracy": acc_2, "correct": corr_2, "total": total})
        print(f"{dataset_name:<55} {acc_1:>6.1%} {acc_2:>6.1%} {total:>6}")

    total_corr_1 = sum(r["correct"] for r in results_1)
    total_corr_2 = sum(r["correct"] for r in results_2)
    total_pairs = sum(r["total"] for r in results_1)
    overall_1 = total_corr_1 / total_pairs if total_pairs > 0 else 0.0
    overall_2 = total_corr_2 / total_pairs if total_pairs > 0 else 0.0

    print("-" * 80)
    print(f"{'OVERALL':<55} {overall_1:>6.1%} {overall_2:>6.1%} {total_pairs:>6}")

    for csv_path, results, overall, total_corr in [
        (args.output_csv_1prefix, results_1, overall_1, total_corr_1),
        (args.output_csv_2prefix, results_2, overall_2, total_corr_2)
    ]:
        if csv_path:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["dataset", "accuracy", "correct", "total"])
                writer.writeheader()
                writer.writerows(results)
                writer.writerow({"dataset": "OVERALL", "accuracy": overall, "correct": total_corr, "total": total_pairs})
            print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
