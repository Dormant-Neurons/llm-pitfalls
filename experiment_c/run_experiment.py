#!/usr/bin/env python
"""
Token-length statistics for four vulnerability datasets
+ LaTeX table output.

Datasets  | code field | label field | positive label
----------|------------|-------------|----------------
bstee615/bigvul                         func_before   vul            1
google/code_x_glue_cc_defect_detection  func          target         True
bstee615/diversevul                     func          target         1
colin/PrimeVul                          func          target         1

Tokenizer  : Salesforce/codet5-small (fast)
Cut-offs   : 512 / 1 024 / 2 048 tokens
"""

import multiprocessing as mp
from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

# ────────── config ───────────────────────────────────────────────────────── #

CUT_OFFS = (512, 1024, 2048)

DATASETS = {
    "Devign": {
        "hf_id":     "google/code_x_glue_cc_defect_detection",
        "code_key":  "func",
        "label_key": "target",
        "positive":  True,
    },
    "DiverseVul": {
        "hf_id":     "bstee615/diversevul",
        "code_key":  "func",
        "label_key": "target",
        "positive":  1,
    },
    "PrimeVul": {
        "hf_id":     "colin/PrimeVul",
        "code_key":  "func",
        "label_key": "target",
        "positive":  1,
    },
}

# ────────── helpers ──────────────────────────────────────────────────────── #

def add_len_flags(example, code_key, cutoffs, tokenizer):
    """Compute token length and flags for 'longer than' each cut-off."""
    length = len(tokenizer(example[code_key]).input_ids)
    flags  = {f"gt_{c}": length > c for c in cutoffs}
    return {"tok_len": length, **flags}


def process_dataset(name, spec, tokenizer, cutoffs):
    """Return total vulnerable and counts > each cut-off."""
    print(f"\n▶  Loading {name} …")
    ds = load_dataset(spec["hf_id"], split="train+validation+test")

    # keep only vulnerable functions
    ds = ds.filter(lambda x: x[spec["label_key"]] == spec["positive"])

    fn = partial(add_len_flags,
                 code_key=spec["code_key"],
                 cutoffs=cutoffs,
                 tokenizer=tokenizer)
    ds = ds.map(fn,
                num_proc=mp.cpu_count(),
                desc=f"Tokenising ({len(ds):,} rows)")

    total  = len(ds)
    counts = {c: int(sum(ds[f"gt_{c}"])) for c in cutoffs}   # ← fixed
    return total, counts


def make_latex(results, cutoffs):
    """
    Build a LaTeX table using siunitx S-columns with paired
    count/(percent) cells for each cut-off and an 'Average' row.
    The averages are unweighted means of the per-dataset percentages.
    """
    # Column spec: l + one S for totals + (count S, percent S) per cut-off
    cols_spec = [
        "l@{\\,}",
        "S[table-format=6.0]",
    ]
    for _ in cutoffs:
        cols_spec.append("S[table-format=5.0]@{~(}S[table-format=2.1, table-space-text-post={\\%}]<{\\%)})")
    cols_spec_str = "\n    ".join(cols_spec)

    # Header rows
    tokens_span = 2 * len(cutoffs)
    cmid_end = 2 + tokens_span  # columns start at 1: dataset=1, total=2, then pairs

    lines = [
        r"\begin{table}[ht]",
        r"  %\scriptsize",
        r"  \setlength{\tabcolsep}{2.5pt}",
        r"  \centering",
        r"  \caption{Proportion of functions labeled as \emph{vulnerable} whose \model{CodeT5}-tokenized length exceeds context-window sizes used in the papers on vulnerability detection identified by our pitfall study (\num{512}, \num{1024}, and \num{2048}~tokens).\label{tab:token-cutoffs}}",
        r"  \begin{tabular}{",
        f"    {cols_spec_str}",
        r"  }",
        r"  \toprule",
        r"   \bfseries Dataset & \bfseries\#~Funcs & \multicolumn{" + f"{tokens_span}" + r"}{c}{\bfseries\# Tokens}  \\",
        r"                                           \cmidrule(lr){" + f"3-{cmid_end}" + r"}",
        "                     &                   " + " ".join(
            f"& \\multicolumn{{2}}{{c}}{{${{{'>{c}'}}}$}} " for c in cutoffs
        ) + r"\\",
        r"  \midrule",
    ]

    # Body rows + collect percentages for averages
    # Preserve insertion order of 'results' for row order
    percs_for_avg = {c: [] for c in cutoffs}
    for name, (total, counts) in results.items():
        # counts[c] is absolute count > c; percent is with one decimal
        row_cells = [f"{name:>16}", f"{int(total)}"]
        for c in cutoffs:
            pct = (counts[c] / total * 100) if total else 0.0
            percs_for_avg[c].append(pct)
            row_cells.extend([f"{int(counts[c])}", f"{pct:.1f}"])
        lines.append("    " + " & ".join(row_cells) + r" \\")
    lines.append(r"  \midrule")

    # Average row (unweighted mean of percentages across datasets)
    avg_cells = [r"    \bfseries", r"    Average          & {--}    "]
    for c in cutoffs:
        if percs_for_avg[c]:
            avg = sum(percs_for_avg[c]) / len(percs_for_avg[c])
        else:
            avg = 0.0
        avg_cells.append(
            r"& \multicolumn{2}{c}{\bfseries\num{" + f"{avg:.1f}" + r"}\%} "
        )
    lines.append(" ".join(avg_cells) + r"\\")

    lines.extend([
        r"  \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ────────── main ─────────────────────────────────────────────────────────── #

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "Salesforce/codet5-small", use_fast=True
    )

    results = {}
    for name, spec in DATASETS.items():
        total, counts = process_dataset(name, spec, tokenizer, CUT_OFFS)
        results[name] = (total, counts)

        # Console summary
        print(f"\n{name}  – vulnerable functions: {total:,}")
        for c in CUT_OFFS:
            pct = counts[c] / total * 100
            print(f"   > {c:4d} tokens : {counts[c]:8,}  ({pct:6.2f} %)")
        print()

    # Build LaTeX table
    latex = make_latex(results, CUT_OFFS)
    print("\n────────── LaTeX table ──────────\n")
    print(latex)
    print("\n────────── end LaTeX ────────────")

    # Write to file (create parent dir if needed)
    out_path = Path("plots/token_cutoff_table.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex, encoding="utf-8")
    print(f"\nLaTeX table saved to {out_path}")

if __name__ == "__main__":
    main()