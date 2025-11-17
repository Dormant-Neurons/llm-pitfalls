# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import argparse
from typing import Any, Dict, List
import os
import sys

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from sklearn.metrics import f1_score
from collections import defaultdict

# ───────────────────────── configuration ──────────────────────────
MODEL_ID = "Salesforce/codet5p-220m"  # default; overridden by --fast
CTX_LIMIT = 512
SPLIT_SEED = 42
# Default sweep when --ratio is *omitted*
DEFAULT_LEAK_RATIOS: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

DATASETS: Dict[str, Dict[str, Any]] = {
    "Devign": {
        "hf_id": "google/code_x_glue_cc_defect_detection",
        "code_key": "func",
        "label_key": "target",
        "positive": True,
    },
    "DiverseVul": {
        "hf_id": "bstee615/diversevul",
        "code_key": "func",
        "label_key": "target",
        "positive": 1,
    },
    "PrimeVul": {
        "hf_id": "colin/PrimeVul",
        "code_key": "func",
        "label_key": "target",
        "positive": 1,
    },
}

# Storage for plotted results when running a full sweep (all datasets)
RESULTS = defaultdict(list)  # tag -> list[(leak_pct:int, f1:float)]

# ───────────────────────── helpers ──────────────────────────

def _as_text_list(values):
    """Coerce a sequence to list[str] for the tokenizer."""
    out = []
    for v in values:
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out


def standardise_columns(ds: Dataset, code_key: str, label_key: str, positive_val: Any) -> Dataset:
    """Rename columns to `func`, `target` and make labels 0/1 ints."""

    def convert(ex):
        ex["func"] = ex.pop(code_key)
        lbl = ex.pop(label_key)
        # Map to {0,1}; assume ints/bools for provided configs
        ex["target"] = int(lbl == positive_val) if isinstance(lbl, (int, bool)) else 0
        return ex

    keep_cols = [code_key, label_key]
    drop_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(drop_cols)
    ds = ds.map(convert)
    return ds


def add_toklen_and_split(ds_all: Dataset, tokenizer) -> DatasetDict:
    """Add token count and produce 60/20/20 random split."""

    def add_tok(ex):
        # Use truncation to avoid long-seq warning during counting
        ex["n_tok"] = len(tokenizer(ex["func"], truncation=True, max_length=CTX_LIMIT).input_ids)
        return ex

    ds_all = ds_all.map(add_tok, num_proc=1)
    ds_all = ds_all.shuffle(seed=SPLIT_SEED)

    n = len(ds_all)
    train_end, val_end = int(0.6 * n), int(0.8 * n)
    return DatasetDict({
        "train": ds_all.select(range(train_end)),
        "validation": ds_all.select(range(train_end, val_end)),
        "test": ds_all.select(range(val_end, n)),
    })


def tokenize_func(batch, tokenizer):
    codes = _as_text_list(batch["func"])  # ensure list[str]
    labels = batch["target"]
    enc = tokenizer(codes, truncation=True, max_length=CTX_LIMIT)
    enc["labels"] = [tokenizer(str(int(l))).input_ids for l in labels]
    return enc


@torch.no_grad()
def eval_f1(ds: Dataset, tokenizer, model):
    model.eval()
    B = 32
    y_true, y_pred = [], []
    for i in range(0, len(ds), B):
        batch = ds.select(range(i, min(i + B, len(ds))))
        codes = _as_text_list(batch["func"])  # ensure list[str]
        enc = tokenizer(
            codes,
            truncation=True,
            max_length=CTX_LIMIT,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        outs = model.generate(**enc, max_length=3, num_beams=1)
        dec = tokenizer.batch_decode(outs, skip_special_tokens=True)
        y_pred.extend([1 if t.strip().startswith("1") else 0 for t in dec])
        y_true.extend(batch["target"])
    return f1_score(y_true, y_pred, average="macro")


def run_for_ratio(
    ratio: float,
    splits: DatasetDict,
    tokenizer,
    epochs: int,
    tag: str,
    model_id: str
):
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Leakage ratio must be in [0,1] (got {ratio})")

    rng = np.random.default_rng(SPLIT_SEED)
    leak_n = int(ratio * len(splits["test"]))
    leak_idx = rng.choice(len(splits["test"]), size=leak_n, replace=False) if leak_n else []
    leak_ds = splits["test"].select(leak_idx) if leak_n else None

    train_ds = concatenate_datasets([splits["train"], leak_ds]).shuffle(seed=SPLIT_SEED) if leak_n else splits["train"]

    # Tokenise
    cols_strip = [c for c in ("func", "target", "n_tok") if c in train_ds.column_names]
    tokenised_train = train_ds.map(lambda ex: tokenize_func(ex, tokenizer),
                                   remove_columns=cols_strip, batched=True, num_proc=1)
    tokenised_val = splits["validation"].map(lambda ex: tokenize_func(ex, tokenizer),
                                             remove_columns=cols_strip, batched=True, num_proc=1)

    if len(tokenised_train) == 0:
        print(f"{tag:10s} | Leak {int(ratio*100):3d}% | *SKIPPED* (empty train set)")
        return None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, low_cpu_mem_usage=True).to(device)

    batch_size = 4 if epochs == 3 else 16
    
    model_tag = model_id.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        output_dir=f"./{model_tag}_{tag}_leak_{int(ratio*100)}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=3e-5,
        num_train_epochs=epochs,
        save_strategy="no",
        logging_strategy="no",
        fp16=(device.type == "cuda"),
        seed=SPLIT_SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        processing_class=tokenizer,  # future-proof: replaces deprecated `tokenizer=` arg
    )
    trainer.train()

    f1 = eval_f1(splits["test"], tokenizer, model)
    print(f"{tag:10s} | Leak {int(ratio*100):3d}% | F1(full test) = {f1:.3f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return f1


def main():
    parser = argparse.ArgumentParser(description="Data leakage experiment (lab-setting)")
    parser.add_argument(
        "--dataset",
        required=False,
        default=None,
        help="Dataset key (Devign/DiverseVul/PrimeVul) or HF path (owner/name). "
             "Omit to run all built-in datasets."
    )
    parser.add_argument("--epochs", type=int, default=10,
                        help="Fine-tuning epochs per leakage ratio (default 10)")
    parser.add_argument("--ratio", type=float, default=None, metavar="R",
                        help="Leakage ratio R ∈ [0,1]. If omitted run the preset sweep.")
    # FAST flag updated
    parser.add_argument("--fast", "--FAST", dest="fast", action="store_true",
                        help="Use google-t5/t5-small and set epochs=3 for a quick run.")

    args = parser.parse_args()

    # Decide model & epochs based on FAST
    model_id = "google-t5/t5-small" if args.fast else MODEL_ID
    epochs = 3 if args.fast else args.epochs
    if args.fast:
        print(">> FAST mode: using google-t5/t5-small and epochs=3")

    # Decide which datasets to run
    if args.dataset is None:
        dataset_keys = list(DATASETS.keys())  # run all built-ins
        
        if args.fast:
            dataset_keys = ["Devign"]
    else:
        dataset_keys = [args.dataset]

    ratios = [args.ratio] if args.ratio is not None else DEFAULT_LEAK_RATIOS

    # This print is informational; actual device selection happens inside run_for_ratio
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    )
    print(f"\n>> device: {device}")
    print(f">> model:  {model_id}")
    print(f">> running ratios: {', '.join(f'{r:.1f}' for r in ratios)}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    HF_TOKEN = os.getenv("HF_TOKEN")  # set in env if needed for gated datasets

    is_full_sweep_all = (args.dataset is None and args.ratio is None)

    for key in dataset_keys:
        # Resolve mapping or fall back to defaults for HF paths
        if key in DATASETS:
            cfg = DATASETS[key]
            hf_id, code_key, label_key, positive = cfg["hf_id"], cfg["code_key"], cfg["label_key"], cfg["positive"]
            tag = key
        else:
            hf_id, code_key, label_key, positive = key, "func", "target", 1
            tag = hf_id.split("/")[-1]

        print(f"\n==== Dataset: {tag}  (source: {hf_id}) ====")
        ds_all = load_dataset(hf_id, split="train+validation+test", token=HF_TOKEN)
        ds_all = standardise_columns(ds_all, code_key, label_key, positive)

        # --- FAST mode: random sample 300 and split 100/100/100 ---
        if args.fast:
            want_n = 600
            if len(ds_all) >= want_n:
                print(f">> FAST: randomly sampling {want_n} samples (out of {len(ds_all)})")
                ds_fast = ds_all.shuffle(seed=SPLIT_SEED).select(range(want_n))
                # Fixed 100/100/100 split (already shuffled)
                splits = DatasetDict({
                    "train": ds_fast.select(range(0, 200)),
                    "validation": ds_fast.select(range(200, 400)),
                    "test": ds_fast.select(range(400, 600)),
                })
            else:
                print(f">> FAST: dataset has only {len(ds_all)} rows; falling back to 60/20/20 split.")
                splits = add_toklen_and_split(ds_all.shuffle(seed=SPLIT_SEED), tokenizer)
        else:
            splits = add_toklen_and_split(ds_all, tokenizer)

        for r in ratios:
            f1 = run_for_ratio(r, splits, tokenizer, epochs, tag, model_id)
            if is_full_sweep_all and f1 is not None:
                RESULTS[tag].append((int(r * 100), f1))

    # If we ran a full sweep (no dataset + no ratio), generate plot
    if is_full_sweep_all:
        import matplotlib.pyplot as plt

        # Ensure output dir exists
        os.makedirs("plots", exist_ok=True)

        # Plot measured results
        plt.figure(figsize=(8, 6))
        for dataset, entries in RESULTS.items():
            if not entries:
                continue
            entries.sort(key=lambda x: x[0])  # by leakage %
            leak_percentages = [p for p, _ in entries]
            f1_scores = [s for _, s in entries]
            plt.plot(leak_percentages, f1_scores, marker='o', label=dataset)

        plt.xlabel("Leakage Percentage")
        plt.ylabel("F1 Score (Full Test)")
        plt.title("F1 Score vs Leakage Percentage (Experiment B-1)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        out_path = "plots/experiment_b_1.pdf"
        plt.savefig(out_path, format="pdf", dpi=300)
        print(f"\n✅ Plot saved to: {out_path}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)