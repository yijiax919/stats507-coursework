from __future__ import annotations

import json
import re
import random
from pathlib import Path
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix

from src.tweeteval_hate.data_utils import load_local_splits, project_root
from src.tweeteval_hate.evaluation import metric_bundle, plot_confusion_matrix
from src.tweeteval_hate.preprocessing import normalize_tweet

MASK_TERMS = [
    "bitch", "bitches", "whore", "hoe", "illegals", "illegal", "womensuck", "buildthatwall", "buildthewall"
]


def sanitize_text(text: str) -> str:
    text = str(text)
    for term in MASK_TERMS:
        text = re.sub(term, term[:1] + "*" * max(1, len(term) - 1), text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+|www\.\S+", "URL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:170]


def augment_text(text: str, aug_type: str = "none") -> str:
    """Simple text augmentation to improve generalization."""
    if aug_type == "none":
        return text
    elif aug_type == "dropout":
        # Random word dropout (15% chance)
        words = text.split()
        words = [w for w in words if random.random() > 0.15]
        return " ".join(words) if words else text
    elif aug_type == "swap":
        # Random adjacent word swap
        words = text.split()
        if len(words) > 1:
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return " ".join(words)
    return text


def build_vectorizer(kind: str, max_features: int = 10000) -> TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),  # Increased to capture more context
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
            lowercase=False,
            norm='l2',  # Add L2 normalization
        )
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),  # Increased range
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
            lowercase=False,
            norm='l2',
        )
    raise ValueError(kind)


def plot_model_comparison(df: pd.DataFrame, out_path: Path, title: str = "Model comparison") -> None:
    x = np.arange(len(df))
    width = 0.36
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, df["val_macro_f1"], width, label="Validation macro-F1")
    plt.bar(x + width / 2, df["test_macro_f1"], width, label="Test macro-F1")
    plt.xticks(x, df["experiment"], rotation=25, ha='right')
    plt.ylim(0.35, 0.8)
    plt.ylabel("Macro-F1")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def cross_validate_model(model, X_train: csr_matrix, y_train: np.ndarray, cv: int = 5) -> Dict[str, float]:
    """Perform stratified k-fold cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro')
    return {
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist()
    }


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["val_macro_f1", "cv_mean", "test_macro_f1"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def majority_vote(predictions: List[np.ndarray]) -> np.ndarray:
    return (np.asarray(predictions).mean(axis=0) >= 0.5).astype(int)


def load_original_baseline(art_dir: Path) -> Dict[str, float | str]:
    summary_path = art_dir / "experiment_summary.csv"
    if summary_path.exists():
        baseline_df = pd.read_csv(summary_path)
        if not baseline_df.empty:
            baseline_df = baseline_df.sort_values(
                ["val_macro_f1", "test_macro_f1"],
                ascending=[False, False],
            ).reset_index(drop=True)
            row = baseline_df.iloc[0]
            return {
                "model": str(row["experiment"]),
                "val_macro_f1": float(row["val_macro_f1"]),
                "test_macro_f1": float(row["test_macro_f1"]),
                "gap": float(row["val_macro_f1"] - row["test_macro_f1"]),
            }

    return {
        "model": "word_lr",
        "val_macro_f1": 0.7312,
        "test_macro_f1": 0.4780,
        "gap": 0.2532,
    }


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    
    root = project_root()
    art_dir = root / "artifacts"
    fig_dir = root / "figures"
    art_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    splits = load_local_splits()
    train = splits["train"]
    val = splits["validation"]
    test = splits["test"]

    x_train_raw = train["text"].fillna("").astype(str)
    x_val_raw = val["text"].fillna("").astype(str)
    x_test_raw = test["text"].fillna("").astype(str)

    x_train = x_train_raw.map(normalize_tweet)
    x_val = x_val_raw.map(normalize_tweet)
    x_test = x_test_raw.map(normalize_tweet)

    y_train = train["label"].astype(int)
    y_val = val["label"].astype(int)
    y_test = test["label"].astype(int)

    # New experiments with different strategies
    experiments = [
        {
            "experiment": "word_lr_balanced",
            "vectorizer": build_vectorizer("word"),
            "model": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=1.0, class_weight='balanced'),
        },
        {
            "experiment": "word_svc_balanced",
            "vectorizer": build_vectorizer("word"),
            "model": LinearSVC(C=0.5, random_state=42, class_weight='balanced'),
        },
        {
            "experiment": "word_lr_higher_c",
            "vectorizer": build_vectorizer("word"),
            "model": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=10.0),
        },
        {
            "experiment": "word_lr_tri_gram",
            "vectorizer": build_vectorizer("word"),  # Already uses tri-grams
            "model": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=2.0),
        },
        {
            "experiment": "char_wider_range",
            "vectorizer": build_vectorizer("char"),  # Already uses wider range
            "model": LinearSVC(C=0.3, random_state=42),
        },
        {
            "experiment": "combined_features",
            "vectorizer_word": build_vectorizer("word", max_features=6000),
            "vectorizer_char": build_vectorizer("char", max_features=4000),
            "model": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=1.0),
            "use_combined": True,
        },
        {
            "experiment": "word_lr_l2_strong",
            "vectorizer": build_vectorizer("word"),
            "model": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42, C=0.1, penalty='l2'),
        },
    ]

    rows = []
    payload = {}
    trained = {}
    
    for exp in experiments:
        print(f"\nTraining {exp['experiment']}...")
        
        if exp.get("use_combined", False):
            vec_word = exp["vectorizer_word"]
            vec_char = exp["vectorizer_char"]
            X_train_word = vec_word.fit_transform(x_train)
            X_val_word = vec_word.transform(x_val)
            X_test_word = vec_word.transform(x_test)
            
            X_train_char = vec_char.fit_transform(x_train)
            X_val_char = vec_char.transform(x_val)
            X_test_char = vec_char.transform(x_test)
            
            X_train = hstack([X_train_word, X_train_char])
            X_val = hstack([X_val_word, X_val_char])
            X_test = hstack([X_test_word, X_test_char])
            
            vec = {"word": vec_word, "char": vec_char}
        else:
            vec = exp["vectorizer"]
            X_train = vec.fit_transform(x_train)
            X_val = vec.transform(x_val)
            X_test = vec.transform(x_test)
        
        model = exp["model"]
        
        # Cross-validation
        cv_results = cross_validate_model(model, X_train, y_train, cv=5)
        print(f"  CV Macro-F1 = {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
        
        # Train on full training set
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        val_metrics = metric_bundle(y_val, val_pred)
        test_metrics = metric_bundle(y_test, test_pred)
        
        rows.append(
            {
                "experiment": exp["experiment"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "cv_mean": cv_results["cv_mean"],
                "cv_std": cv_results["cv_std"],
                "gap": val_metrics["macro_f1"] - test_metrics["macro_f1"],
            }
        )
        payload[exp["experiment"]] = {
            "validation": val_metrics, 
            "test": test_metrics,
            "cross_validation": cv_results
        }
        trained[exp["experiment"]] = {
            "vectorizer": vec, 
            "model": model, 
            "val_pred": val_pred, 
            "test_pred": test_pred
        }

    base_result_df = sort_results(pd.DataFrame(rows))

    print("\nCreating ensemble of top 3 models (by validation performance)...")
    top_3_models = base_result_df.head(3)["experiment"].tolist()

    ensemble_val_pred = majority_vote([trained[name]["val_pred"] for name in top_3_models])
    ensemble_test_pred = majority_vote([trained[name]["test_pred"] for name in top_3_models])
    ensemble_val_metrics = metric_bundle(y_val, ensemble_val_pred)
    ensemble_test_metrics = metric_bundle(y_test, ensemble_test_pred)

    ensemble_row = {
        "experiment": "voting_ensemble",
        "val_accuracy": ensemble_val_metrics["accuracy"],
        "val_macro_f1": ensemble_val_metrics["macro_f1"],
        "test_accuracy": ensemble_test_metrics["accuracy"],
        "test_macro_f1": ensemble_test_metrics["macro_f1"],
        "cv_mean": None,
        "cv_std": None,
        "gap": ensemble_val_metrics["macro_f1"] - ensemble_test_metrics["macro_f1"],
    }

    payload["voting_ensemble"] = {
        "validation": ensemble_val_metrics,
        "test": ensemble_test_metrics,
        "component_models": top_3_models,
        "selection_basis": "top_3_validation_macro_f1",
    }

    ensemble_df = pd.DataFrame([ensemble_row], columns=base_result_df.columns).astype(
        {
            "val_accuracy": "float64",
            "val_macro_f1": "float64",
            "test_accuracy": "float64",
            "test_macro_f1": "float64",
            "cv_mean": "float64",
            "cv_std": "float64",
            "gap": "float64",
        }
    )
    result_df = sort_results(pd.concat([base_result_df, ensemble_df], ignore_index=True))
    result_df.to_csv(art_dir / "experiment_summary_v2.csv", index=False)
    (art_dir / "experiment_metrics_v2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_model_comparison(result_df, fig_dir / "model_comparison_v2.png", "Model Comparison V2")

    # Print results
    print("\n" + "="*70)
    print("RESULTS V2 (sorted by validation performance):")
    print(result_df[["experiment", "val_macro_f1", "test_macro_f1", "gap"]].to_string(index=False))

    best_model = result_df.iloc[0]
    best_test_model = result_df.sort_values("test_macro_f1", ascending=False).iloc[0]
    print("\n" + "="*70)
    print(f"Validation-selected best model: {best_model['experiment']}")
    print(f"  Validation Macro-F1: {best_model['val_macro_f1']:.4f}")
    print(f"  Test Macro-F1: {best_model['test_macro_f1']:.4f}")
    print(f"  Gap: {best_model['gap']:.4f}")

    print(f"\nBest test-set model (diagnostic only): {best_test_model['experiment']}")
    print(f"  Validation Macro-F1: {best_test_model['val_macro_f1']:.4f}")
    print(f"  Test Macro-F1: {best_test_model['test_macro_f1']:.4f}")
    print(f"  Gap: {best_test_model['gap']:.4f}")

    original_baseline = load_original_baseline(art_dir)
    original_test_f1 = float(original_baseline["test_macro_f1"])
    original_gap = float(original_baseline["gap"])

    improvement = best_model['test_macro_f1'] - original_test_f1
    gap_improvement = original_gap - best_model['gap']

    print(f"\nComparison with original baseline:")
    print(f"  Original model: {original_baseline['model']}")
    print(f"  Original test F1: {original_test_f1:.4f}")
    print(f"  New test F1: {best_model['test_macro_f1']:.4f}")
    print(f"  Improvement: {improvement:+.4f}")
    print(f"  Original gap: {original_gap:.4f}")
    print(f"  New gap: {best_model['gap']:.4f}")
    print(f"  Gap reduction: {gap_improvement:+.4f}")
    
    summary = {
        "best_model": best_model["experiment"],
        "best_val_f1": float(best_model["val_macro_f1"]),
        "corresponding_test_f1": float(best_model["test_macro_f1"]),
        "gap": float(best_model["gap"]),
        "selection_metric": "val_macro_f1",
        "best_test_model": best_test_model["experiment"],
        "best_test_f1": float(best_test_model["test_macro_f1"]),
        "best_test_model_val_f1": float(best_test_model["val_macro_f1"]),
        "best_test_model_gap": float(best_test_model["gap"]),
        "original_best_model": original_baseline["model"],
        "original_test_f1": original_test_f1,
        "original_gap": original_gap,
        "improvement_over_original": float(improvement),
        "gap_reduction": float(gap_improvement),
    }
    (art_dir / "v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    print("\n[OK] V2 experiments complete")


if __name__ == "__main__":
    main()
