from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

SPLITS = ("train", "validation", "test")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data" / "tweet_eval_hate"


def load_local_splits() -> Dict[str, pd.DataFrame]:
    root = data_dir()
    frames: Dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = root / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")
        df = pd.read_csv(path)
        frames[split] = df[["text", "label"]].copy()
    return frames


def load_hf_splits() -> Dict[str, pd.DataFrame]:
    from datasets import load_dataset

    ds = load_dataset("cardiffnlp/tweet_eval", "hate")
    frames: Dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        frames[split] = ds[split].to_pandas()[["text", "label"]].copy()
    return frames
