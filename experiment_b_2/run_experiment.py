from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ── Optional .env loader ──
try:  # type: ignore
    load_dotenv()
except ImportError:  # pragma: no cover
    print("[INFO] python-dotenv not installed – skipping .env loading.", file=sys.stderr)

# ── Similarity helper ──
def jaccard_similarity(a: str, b: str) -> float:
    s1, s2 = set(a.split()), set(b.split())
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0


# ── Prompt templates ──
COMMIT_MESSAGE_SYSTEM_PROMPT = (
    "You are a commit message assistant. I will give you project+commit+partial message. "
    "Predict the full original commit message only. No markdown or explanation."
)

CODE_SYSTEM_PROMPT = (
    "You are a code assistant. I will give you a project name, a commit ID, and the first half of a C/C++ function.\n"
    "Predict the full original function only. No explanation or formatting."
)


# ── Unified backend helpers ──
class CompletionBackend:
    """Wrapper that hides backend-specific API calls."""

    def __init__(self, name: str, model: str, max_tokens: int = 1024):
        self.name = name.lower()
        self.model = model
        self.max_tokens = max_tokens

        if self.name == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set.")
            self.client = OpenAI(api_key=api_key)
            self._call = self._openai_call  # type: ignore[attr-defined]

        elif self.name == "anthropic":
            from anthropic import Anthropic  # type: ignore
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set.")
            self.client = Anthropic(api_key=api_key)
            self._call = self._anthropic_call  # type: ignore[attr-defined]

        elif self.name == "deepseek":
            from openai import OpenAI
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY not set.")
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            self._call = self._openai_call  # type: ignore[attr-defined]

        elif self.name == "local":
            from openai import OpenAI
            base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234")
            api_key = os.getenv("LLM_API_KEY", "lm-studio")
            self.client = OpenAI(api_key=api_key, base_url=f"{base_url.rstrip('/')}/v1")
            self._call = self._openai_call  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unsupported backend: {name}")

    # ---- backend-specific call implementations ----
    def _openai_call(self, user_prompt: str, system_prompt: str) -> str:  # noqa: D401
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def _anthropic_call(self, user_prompt: str, system_prompt: str) -> str:  # noqa: D401
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    # ---- public API ----
    def complete(self, project: str, commit: str, partial: str, system_prompt: str, suffix: str = "") -> str:
        user_prompt = (
            f"Project: {project}\nCommit: {commit}\nPartial commit message: \"{partial}\""
        )
        if suffix:
            user_prompt += f"\n{suffix}"
        try:
            return self._call(user_prompt, system_prompt)  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover – network/API errors
            print(f"[WARN] Completion error ({self.name}) – returning empty string: {exc}", file=sys.stderr)
            return ""


# ── Hugging Face dataset loader (replaces local files) ──
RANDOM_SEED = 42
N_SAMPLES = 100  # fixed sample size

def _hf_load_all_splits() -> Dict[str, pd.DataFrame]:
    """
    Load the 'colin/PrimeVul' dataset from Hugging Face using an HF token if provided.

    Expected columns (based on the original script's usage):
      - commit_message (str)
      - project_url or project (str)
      - commit_id (str)
      - func (str)  # full function text for code completion
    """
    from datasets import load_dataset

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("[INFO] HF_TOKEN not set. Attempting to load 'colin/PrimeVul' without authentication.", file=sys.stderr)

    # datasets>=2.13: use token=...
    ds = load_dataset("colin/PrimeVul", token=hf_token)

    frames: Dict[str, pd.DataFrame] = {}
    for split_name, dset in ds.items():
        # Convert to pandas
        frames[split_name] = dset.to_pandas()  # type: ignore[attr-defined]
    return frames


def load_dataset_df() -> pd.DataFrame:
    """
    Load & combine HF splits into a single DataFrame of unique commits.
    Removes rows with missing/placeholder commit messages.
    """
    frames = _hf_load_all_splits()

    # Accept common split names; ignore any that aren't present.
    concat_candidates = []
    for key in ("train", "validation", "valid", "test"):
        if key in frames:
            concat_candidates.append(frames[key])

    if not concat_candidates:
        raise RuntimeError("No usable splits found in 'colin/PrimeVul'.")

    df_all = pd.concat(concat_candidates, ignore_index=True)

    # Normalize field names if needed
    if "project_url" not in df_all.columns and "project" in df_all.columns:
        df_all = df_all.rename(columns={"project": "project_url"})

    # Filter unusable commit messages
    if "commit_message" not in df_all.columns:
        raise RuntimeError("Dataset is missing 'commit_message' column.")
    df_all = df_all[df_all["commit_message"].notna() & (df_all["commit_message"] != "None")]

    # Ensure 'commit_id' exists
    if "commit_id" not in df_all.columns:
        raise RuntimeError("Dataset is missing 'commit_id' column.")

    # Drop duplicate commits by commit_id (no commit date join anymore)
    df = df_all.drop_duplicates("commit_id").reset_index(drop=True)
    return df


def sample_commits(df: pd.DataFrame, n: int, rng_seed: int) -> pd.DataFrame:
    if len(df) < n:
        raise RuntimeError(f"Dataset only has {len(df)} unique commits (<{n}).")
    return df.sample(n, random_state=rng_seed).reset_index(drop=True)


# ── Text helpers ──
def normalise(txt: str) -> str:
    return "\n".join(l.rstrip() for l in txt.strip().splitlines())


def strip_formatting(txt: str) -> str:
    return txt.replace("\n", " ").replace("\t", " ").strip()


# ── Evaluation loop ──
def evaluate_commit(df: pd.DataFrame, backend: CompletionBackend, show_commits: bool = False, suffix: str = "") -> Dict[str, float | int]:
    exact, jaccs = [], []

    # Be tolerant if dataset uses 'project' instead of 'project_url'
    proj_col = "project_url" if "project_url" in df.columns else ("project" if "project" in df.columns else None)
    if proj_col is None:
        raise RuntimeError("Dataset missing 'project_url'/'project' column.")

    iterator: List[Tuple[str, str, str]] = list(zip(df[proj_col], df["commit_id"], df["commit_message"]))

    for proj, cid, true_msg in tqdm(iterator, desc="Commit Message Completion", total=len(df)):
        norm_truth = normalise(true_msg)
        hint_words = norm_truth.split()
        hint = " ".join(hint_words[: len(hint_words) // 2])  # first half as partial

        pred = normalise(backend.complete(proj, cid, hint, COMMIT_MESSAGE_SYSTEM_PROMPT, suffix=suffix))

        truth_stripped = strip_formatting(norm_truth)
        pred_stripped = strip_formatting(pred)

        if show_commits:
            print("\n=== COMMIT MESSAGE ===")
            print(f"Hint     : {hint}")
            print(f"Actual   : {truth_stripped}")
            print(f"Predicted: {pred_stripped}")

        match = pred_stripped == truth_stripped
        exact.append(int(match))
        jaccs.append(int(jaccard_similarity(truth_stripped, pred_stripped) > 0.75))

    return {
        "accuracy": float(np.mean(exact)),
        "jaccard": float(np.mean(jaccs)),
        "correct": int(np.sum(exact)),
        "jaccard_correct": int(np.sum(jaccs)),
        "total": len(df),
    }


def evaluate_code(df: pd.DataFrame, backend: CompletionBackend, show_commits: bool = False, suffix: str = "") -> Dict[str, float | int]:
    exact, jaccs = [], []

    if "func" not in df.columns:
        raise RuntimeError("Dataset is missing 'func' column required for code completion.")

    # Be tolerant if dataset uses 'project' instead of 'project_url'
    proj_col = "project_url" if "project_url" in df.columns else ("project" if "project" in df.columns else None)
    if proj_col is None:
        raise RuntimeError("Dataset missing 'project_url'/'project' column.")

    iterator: List[Tuple[str, str, str]] = list(zip(df[proj_col], df["commit_id"], df["func"]))

    for proj, cid, full_func in tqdm(iterator, desc="Code Completion", total=len(df)):
        norm_truth = normalise(full_func)
        lines = norm_truth.splitlines()
        split_point = len(lines) // 2
        prefix = "\n".join(lines[:split_point])
        ground_truth = "\n".join(lines)

        predicted = backend.complete(proj, cid, prefix, system_prompt=CODE_SYSTEM_PROMPT, suffix=suffix)
        predicted_norm = normalise(predicted)

        pred_stripped = strip_formatting(predicted_norm)
        truth_stripped = strip_formatting(ground_truth)

        if show_commits:
            print("\n=== FUNCTION COMPLETION ===")
            print(f"Prefix   :\n{prefix}")
            print(f"Ground   : {truth_stripped}")
            print(f"Predicted: {pred_stripped}")

        match = pred_stripped == truth_stripped
        exact.append(int(match))
        jaccs.append(int(jaccard_similarity(truth_stripped, pred_stripped) > 0.75))

    return {
        "accuracy": float(np.mean(exact)),
        "jaccard": float(np.mean(jaccs)),
        "correct": int(np.sum(exact)),
        "jaccard_correct": int(np.sum(jaccs)),
        "total": len(df),
    }


# ── CLI interface ──
def parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(description="PrimeVul commit-message benchmark (HF dataset, multi-backend)")
    parser.add_argument(
        "--backend",
        choices=["openai", "anthropic", "deepseek", "local"],
        default=os.getenv("BACKEND"),
        help="Which backend to use (env BACKEND overrides).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        help="Model name (env MODEL_NAME).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional text to append to the end of the prompt.",
    )
    parser.add_argument("--show", action="store_true", help="Print commit messages + predictions.")
    parser.add_argument("--samples", type=int, default=N_SAMPLES, help="How many unique commits to sample.")
    return parser.parse_args()


# ── Main entrypoint ──
def main() -> None:  # noqa: D401
    args = parse_args()

    # Auto-detect backend if not provided
    backend_name = args.backend or (
        "anthropic"
        if os.getenv("ANTHROPIC_API_KEY")
        else (
            "deepseek"
            if os.getenv("DEEPSEEK_API_KEY")
            else ("openai" if os.getenv("OPENAI_API_KEY") else "local")
        )
    )

    backend = CompletionBackend(backend_name, args.model)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    df_all = load_dataset_df()
    sample_df = sample_commits(df_all, args.samples, RANDOM_SEED)

    commit_message_metrics = evaluate_commit(sample_df, backend, show_commits=args.show, suffix=args.suffix)
    code_metrics = evaluate_code(sample_df, backend, show_commits=args.show, suffix=args.suffix)

    print("\n===== Commit Completion Summary =====")
    print(
        f"Backend={backend.name}  Model={backend.model}\n"
        f"Matches               : {commit_message_metrics['accuracy']:.2%} "
        f"({commit_message_metrics['correct']}/{commit_message_metrics['total']})\n"
        f"Jaccard over 75%      : {commit_message_metrics['jaccard']:.2%} "
        f"({commit_message_metrics['jaccard_correct']}/{commit_message_metrics['total']})\n"
    )

    print("\n===== Code Completion Summary =====")
    print(
        f"Backend={backend.name}  Model={backend.model}\n"
        f"Matches               : {code_metrics['accuracy']:.2%} "
        f"({code_metrics['correct']}/{code_metrics['total']})\n"
        f"Jaccard over 75%      : {code_metrics['jaccard']:.2%} "
        f"({code_metrics['jaccard_correct']}/{code_metrics['total']})\n"
    )


if __name__ == "__main__":
    main()