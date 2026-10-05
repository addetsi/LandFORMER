#!/usr/bin/env python3
"""
BLiMP (Benchmark of Linguistic Minimal Pairs) evaluation for LandFORMER models.
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
    parser = argparse.ArgumentParser(description="BLiMP evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--blimp_dir", type=str, default=os.path.expanduser("~/landformer/blimp/data"))
    parser.add_argument("--output_csv", type=str, default=None)
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
    dtype_str = args_dict.get("dtype", "torch.bfloat16")
    if dtype_str == "torch.bfloat16" or dtype_str is None:
        model_args.dtype = torch.float16
    elif dtype_str == "torch.float16":
        model_args.dtype = torch.float16
    else:
        model_args.dtype = torch.float32
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
    print(f"  Iteration: {ckpt.get('itr', 'unknown')}")
    return model, model_args


@torch.no_grad()
def compute_log_prob(model, token_ids, device, dtype):
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    # Pass dummy targets to force full logit computation (not just last position)
    dummy_targets = torch.zeros_like(x)
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        outputs = model(x, targets=dummy_targets, get_logits=True, use_cache=False)
    logits = outputs["logits"]
    log_probs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for i in range(len(token_ids) - 1):
        total += log_probs[0, i, token_ids[i + 1]].item()
    return total


def evaluate_blimp_dataset(model, model_args, dataset_path, tokenizer, device):
    correct = 0
    total = 0
    with open(dataset_path, "r") as f:
        for line in f:
            item = json.loads(line)
            good_ids = tokenizer.encode(item["sentence_good"])
            bad_ids = tokenizer.encode(item["sentence_bad"])
            if len(good_ids) < 2 or len(bad_ids) < 2:
                continue
            good_logprob = compute_log_prob(model, good_ids, device, model_args.dtype)
            bad_logprob = compute_log_prob(model, bad_ids, device, model_args.dtype)
            if good_logprob > bad_logprob:
                correct += 1
            total += 1
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def main():
    args = get_args()
    tokenizer = tiktoken.get_encoding("gpt2")
    blimp_files = sorted(Path(args.blimp_dir).glob("*.jsonl"))
    if len(blimp_files) == 0:
        print(f"ERROR: No .jsonl files found in {args.blimp_dir}")
        sys.exit(1)
    print(f"Found {len(blimp_files)} BLiMP datasets")
    print(f"Loading model from: {args.checkpoint}")
    model, model_args = load_model(args.checkpoint, args.device)
    results = []
    print(f"\n{'Dataset':<55} {'Accuracy':>8} {'Correct':>8} {'Total':>6}")
    print("-" * 80)
    for dataset_path in blimp_files:
        dataset_name = dataset_path.stem
        accuracy, correct, total = evaluate_blimp_dataset(
            model, model_args, dataset_path, tokenizer, args.device
        )
        results.append({"dataset": dataset_name, "accuracy": accuracy, "correct": correct, "total": total})
        print(f"{dataset_name:<55} {accuracy:>7.1%} {correct:>8}/{total:<6}")
    total_correct = sum(r["correct"] for r in results)
    total_pairs = sum(r["total"] for r in results)
    overall_accuracy = total_correct / total_pairs if total_pairs > 0 else 0.0
    print("-" * 80)
    print(f"{'OVERALL':<55} {overall_accuracy:>7.1%} {total_correct:>8}/{total_pairs:<6}")
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "accuracy", "correct", "total"])
            writer.writeheader()
            writer.writerows(results)
            writer.writerow({"dataset": "OVERALL", "accuracy": overall_accuracy, "correct": total_correct, "total": total_pairs})
        print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
