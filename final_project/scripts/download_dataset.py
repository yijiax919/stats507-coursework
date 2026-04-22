from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.tweeteval_hate.data_utils import SPLITS, data_dir, load_hf_splits, load_local_splits, project_root


def main() -> None:
    root = project_root()
    out_data = data_dir()
    out_art = root / "artifacts"
    out_fig = root / "figures"
    out_data.mkdir(parents=True, exist_ok=True)
    out_art.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    try:
        frames = load_hf_splits()
        source = "huggingface"
    except Exception:
        frames = load_local_splits()
        source = "local_csv_fallback"

    for split in SPLITS:
        frames[split].to_csv(out_data / f"{split}.csv", index=False)

    train = frames["train"].copy()
    stats = {
        "source": source,
        "splits": {k: int(v.shape[0]) for k, v in frames.items()},
        "label_counts_train": train["label"].value_counts().sort_index().to_dict(),
        "avg_text_len_chars_train": float(train["text"].astype(str).str.len().mean()),
        "median_text_len_chars_train": float(train["text"].astype(str).str.len().median()),
        "max_text_len_chars_train": int(train["text"].astype(str).str.len().max()),
    }
    (out_art / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    counts = train["label"].value_counts().sort_index()
    plt.figure(figsize=(5.2, 3.2))
    plt.bar(["Non-hate (0)", "Hate (1)"], counts.values)
    plt.ylabel("Count")
    plt.title("Training label distribution")
    plt.tight_layout()
    plt.savefig(out_fig / "label_distribution.png", dpi=220)
    plt.close()

    lengths = pd.DataFrame(
        {
            "split": sum([[split] * len(df) for split, df in frames.items()], []),
            "length": pd.concat([df["text"].astype(str).str.len() for df in frames.values()], axis=0).values,
        }
    )
    order = ["train", "validation", "test"]
    plt.figure(figsize=(5.6, 3.4))
    groups = [lengths.loc[lengths["split"] == split, "length"].values for split in order]
    plt.boxplot(groups, tick_labels=order, showfliers=False)
    plt.ylabel("Tweet length (characters)")
    plt.title("Tweet length across splits")
    plt.tight_layout()
    plt.savefig(out_fig / "tweet_length_boxplot.png", dpi=220)
    plt.close()

    print(json.dumps(stats, indent=2))
    print("[OK] dataset assets ready")


if __name__ == "__main__":
    main()
