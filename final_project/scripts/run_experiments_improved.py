from __future__ import annotations

import json
import re
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
from sklearn.feature_selection import SelectFromModel
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


def build_vectorizer(kind: str, max_features: int = 8000) -> TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=3,  # Increased from 2 to reduce overfitting
            max_features=max_features,
            sublinear_tf=True,
            lowercase=False,
        )
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,  # Increased from 2
            max_features=max_features,
            sublinear_tf=True,
            lowercase=False,
        )
    raise ValueError(kind)


def plot_model_comparison(df: pd.DataFrame, out_path: Path, title: str = "Model comparison") -> None:
    x = np.arange(len(df))
    width = 0.36
    plt.figure(figsize=(7, 3.8))
    plt.bar(x - width / 2, df["val_macro_f1"], width, label="Validation macro-F1")
    plt.bar(x + width / 2, df["test_macro_f1"], width, label="Test macro-F1")
    plt.xticks(x, df["experiment"], rotation=20, ha='right')
    plt.ylim(0.35, 0.8)
    plt.ylabel("Macro-F1")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_improvement_comparison(original_df: pd.DataFrame, improved_df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    
    # Original results
    x = np.arange(len(original_df))
    width = 0.36
    ax1.bar(x - width / 2, original_df["val_macro_f1"], width, label="Validation")
    ax1.bar(x + width / 2, original_df["test_macro_f1"], width, label="Test")
    ax1.set_xticks(x)
    ax1.set_xticklabels(original_df["experiment"], rotation=15, ha='right')
    ax1.set_ylim(0.35, 0.8)
    ax1.set_ylabel("Macro-F1")
    ax1.set_title("Original Models")
    ax1.legend()
    
    # Improved results
    x2 = np.arange(len(improved_df))
    ax2.bar(x2 - width / 2, improved_df["val_macro_f1"], width, label="Validation")
    ax2.bar(x2 + width / 2, improved_df["test_macro_f1"], width, label="Test")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(improved_df["experiment"], rotation=20, ha='right')
    ax2.set_ylim(0.35, 0.8)
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("Improved Models")
    ax2.legend()
    
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


def select_features(X_train: csr_matrix, y_train: np.ndarray, X_val: csr_matrix, X_test: csr_matrix, 
                   selector_model: LogisticRegression, threshold: str = "median") -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Perform feature selection to reduce overfitting."""
    selector = SelectFromModel(selector_model, threshold=threshold, prefit=False)
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_val_selected, X_test_selected, selector


def ensemble_predictions(predictions: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
    """Weighted voting ensemble."""
    if weights is None:
        weights = [1.0] * len(predictions)
    weighted_preds = np.zeros_like(predictions[0], dtype=float)
    for pred, weight in zip(predictions, weights):
        weighted_preds += pred * weight
    return (weighted_preds >= 0.5).astype(int)


def main() -> None:
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

    # Original experiments (for comparison)
    original_experiments = [
        {
            "experiment": "word_lr_original",
            "vectorizer": build_vectorizer("word"),
            "model": LogisticRegression(max_iter=800, solver="liblinear", random_state=42, C=4),
        },
        {
            "experiment": "word_svc_original",
            "vectorizer": build_vectorizer("word"),
            "model": LinearSVC(C=1.0, random_state=42),
        },
        {
            "experiment": "char_svc_original",
            "vectorizer": build_vectorizer("char"),
            "model": LinearSVC(C=0.5, random_state=42),
        },
    ]

    # Improved experiments with stronger regularization and feature selection
    improved_experiments = [
        {
            "experiment": "word_lr_strong_reg",
            "vectorizer": build_vectorizer("word"),
            "model": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=0.5),  # Reduced C
            "use_feature_selection": True,
        },
        {
            "experiment": "word_svc_strong_reg",
            "vectorizer": build_vectorizer("word"),
            "model": LinearSVC(C=0.1, random_state=42),  # Reduced C
            "use_feature_selection": True,
        },
        {
            "experiment": "char_svc_strong_reg",
            "vectorizer": build_vectorizer("char"),
            "model": LinearSVC(C=0.1, random_state=42),  # Reduced C
            "use_feature_selection": True,
        },
        {
            "experiment": "word_lr_l1_reg",
            "vectorizer": build_vectorizer("word"),
            "model": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=0.5, penalty="l1"),
            "use_feature_selection": False,  # L1 does implicit feature selection
        },
        {
            "experiment": "combined_word_char",
            "vectorizer_word": build_vectorizer("word", max_features=5000),
            "vectorizer_char": build_vectorizer("char", max_features=3000),
            "model": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=0.5),
            "use_combined_features": True,
        },
    ]

    # Run original experiments
    print("Running original experiments...")
    original_rows = []
    original_payload = {}
    original_trained = {}
    
    for exp in original_experiments:
        vec = exp["vectorizer"]
        model = exp["model"]
        X_train = vec.fit_transform(x_train)
        X_val = vec.transform(x_val)
        X_test = vec.transform(x_test)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        val_metrics = metric_bundle(y_val, val_pred)
        test_metrics = metric_bundle(y_test, test_pred)
        original_rows.append(
            {
                "experiment": exp["experiment"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            }
        )
        original_payload[exp["experiment"]] = {"validation": val_metrics, "test": test_metrics}
        original_trained[exp["experiment"]] = {"vectorizer": vec, "model": model, "val_pred": val_pred, "test_pred": test_pred}

    original_df = pd.DataFrame(original_rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)

    # Run improved experiments
    print("\nRunning improved experiments...")
    improved_rows = []
    improved_payload = {}
    improved_trained = {}
    
    for exp in improved_experiments:
        if exp.get("use_combined_features", False):
            # Combined word + char features
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
        
        # Feature selection if enabled
        original_dim = X_train.shape[1]
        if exp.get("use_feature_selection", False):
            selector_model = LogisticRegression(max_iter=500, solver="liblinear", random_state=42, C=0.5)
            X_train, X_val, X_test, selector = select_features(X_train, y_train, X_val, X_test, selector_model)
            print(f"  {exp['experiment']}: Feature selection reduced dimensions from {original_dim} to {X_train.shape[1]}")
        
        # Cross-validation
        cv_results = cross_validate_model(model, X_train, y_train, cv=5)
        print(f"  {exp['experiment']}: CV Macro-F1 = {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
        
        # Train on full training set
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        val_metrics = metric_bundle(y_val, val_pred)
        test_metrics = metric_bundle(y_test, test_pred)
        
        improved_rows.append(
            {
                "experiment": exp["experiment"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "cv_mean": cv_results["cv_mean"],
                "cv_std": cv_results["cv_std"],
            }
        )
        improved_payload[exp["experiment"]] = {
            "validation": val_metrics, 
            "test": test_metrics,
            "cross_validation": cv_results
        }
        improved_trained[exp["experiment"]] = {
            "vectorizer": vec, 
            "model": model, 
            "val_pred": val_pred, 
            "test_pred": test_pred,
            "selector": selector if exp.get("use_feature_selection", False) else None,
            "original_dim": original_dim if exp.get("use_feature_selection", False) else None
        }

    improved_df = pd.DataFrame(improved_rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    
    # Save results
    original_df.to_csv(art_dir / "experiment_summary_original.csv", index=False)
    improved_df.to_csv(art_dir / "experiment_summary_improved.csv", index=False)
    
    (art_dir / "experiment_metrics_original.json").write_text(json.dumps(original_payload, indent=2), encoding="utf-8")
    (art_dir / "experiment_metrics_improved.json").write_text(json.dumps(improved_payload, indent=2), encoding="utf-8")
    
    plot_model_comparison(original_df, fig_dir / "model_comparison_original.png", "Original Models")
    plot_model_comparison(improved_df, fig_dir / "model_comparison_improved.png", "Improved Models")
    plot_improvement_comparison(original_df, improved_df, fig_dir / "original_vs_improved.png")
    
    # Create ensemble of best models
    print("\nCreating ensemble of top 3 models...")
    top_3_models = improved_df.head(3)["experiment"].tolist()
    ensemble_val_preds = []
    ensemble_test_preds = []
    ensemble_weights = []
    
    for model_name in top_3_models:
        trained = improved_trained[model_name]
        
        # Transform data
        if isinstance(trained["vectorizer"], dict):
            X_val = hstack([trained["vectorizer"]["word"].transform(x_val), trained["vectorizer"]["char"].transform(x_val)])
            X_test = hstack([trained["vectorizer"]["word"].transform(x_test), trained["vectorizer"]["char"].transform(x_test)])
        else:
            X_val = trained["vectorizer"].transform(x_val)
            X_test = trained["vectorizer"].transform(x_test)
        
        # Apply feature selection if needed
        if trained["selector"] is not None:
            X_val = trained["selector"].transform(X_val)
            X_test = trained["selector"].transform(X_test)
        
        # Get decision values for weighted voting
        if hasattr(trained["model"], "decision_function"):
            val_dec = trained["model"].decision_function(X_val)
            test_dec = trained["model"].decision_function(X_test)
            # Convert decision values to probabilities
            val_prob = 1 / (1 + np.exp(-val_dec))
            test_prob = 1 / (1 + np.exp(-test_dec))
        else:
            val_prob = trained["model"].predict_proba(X_val)[:, 1]
            test_prob = trained["model"].predict_proba(X_test)[:, 1]
        
        ensemble_val_preds.append(val_prob)
        ensemble_test_preds.append(test_prob)
        # Weight by validation performance
        weight = improved_df[improved_df["experiment"] == model_name]["val_macro_f1"].values[0]
        ensemble_weights.append(weight)
    
    # Normalize weights
    ensemble_weights = np.array(ensemble_weights)
    ensemble_weights = ensemble_weights / ensemble_weights.sum()
    
    ensemble_val_pred = (np.average(ensemble_val_preds, axis=0, weights=ensemble_weights) >= 0.5).astype(int)
    ensemble_test_pred = (np.average(ensemble_test_preds, axis=0, weights=ensemble_weights) >= 0.5).astype(int)
    
    ensemble_val_metrics = metric_bundle(y_val, ensemble_val_pred)
    ensemble_test_metrics = metric_bundle(y_test, ensemble_test_pred)
    
    ensemble_row = {
        "experiment": "ensemble_top3",
        "val_accuracy": ensemble_val_metrics["accuracy"],
        "val_macro_f1": ensemble_val_metrics["macro_f1"],
        "test_accuracy": ensemble_test_metrics["accuracy"],
        "test_macro_f1": ensemble_test_metrics["macro_f1"],
    }
    
    improved_payload["ensemble_top3"] = {
        "validation": ensemble_val_metrics,
        "test": ensemble_test_metrics,
        "component_models": top_3_models,
        "weights": ensemble_weights.tolist()
    }
    
    # Add ensemble to results
    improved_df = pd.concat([improved_df, pd.DataFrame([ensemble_row])], ignore_index=True)
    improved_df = improved_df.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    
    # Final summary
    print("\n" + "="*60)
    print("ORIGINAL RESULTS:")
    print(original_df.to_string(index=False))
    print("\nIMPROVED RESULTS:")
    print(improved_df.to_string(index=False))
    
    # Calculate improvements
    best_original = original_df.iloc[0]
    best_improved = improved_df.iloc[0]
    
    improvement_val = best_improved["val_macro_f1"] - best_original["val_macro_f1"]
    improvement_test = best_improved["test_macro_f1"] - best_original["test_macro_f1"]
    
    print("\n" + "="*60)
    print(f"Best original model: {best_original['experiment']}")
    print(f"  Validation Macro-F1: {best_original['val_macro_f1']:.4f}")
    print(f"  Test Macro-F1: {best_original['test_macro_f1']:.4f}")
    print(f"  Gap: {best_original['val_macro_f1'] - best_original['test_macro_f1']:.4f}")
    
    print(f"\nBest improved model: {best_improved['experiment']}")
    print(f"  Validation Macro-F1: {best_improved['val_macro_f1']:.4f}")
    print(f"  Test Macro-F1: {best_improved['test_macro_f1']:.4f}")
    print(f"  Gap: {best_improved['val_macro_f1'] - best_improved['test_macro_f1']:.4f}")
    
    print(f"\nImprovement:")
    print(f"  Validation: {improvement_val:+.4f}")
    print(f"  Test: {improvement_test:+.4f}")
    print(f"  Gap reduction: {(best_original['val_macro_f1'] - best_original['test_macro_f1']) - (best_improved['val_macro_f1'] - best_improved['test_macro_f1']):+.4f}")
    
    # Save final results
    improved_df.to_csv(art_dir / "experiment_summary_improved.csv", index=False)
    (art_dir / "experiment_metrics_improved.json").write_text(json.dumps(improved_payload, indent=2), encoding="utf-8")
    
    summary = {
        "best_original_model": best_original["experiment"],
        "best_original_val_f1": float(best_original["val_macro_f1"]),
        "best_original_test_f1": float(best_original["test_macro_f1"]),
        "best_improved_model": best_improved["experiment"],
        "best_improved_val_f1": float(best_improved["val_macro_f1"]),
        "best_improved_test_f1": float(best_improved["test_macro_f1"]),
        "test_improvement": float(improvement_test),
        "gap_reduction": float((best_original['val_macro_f1'] - best_original['test_macro_f1']) - (best_improved['val_macro_f1'] - best_improved['test_macro_f1'])),
    }
    (art_dir / "improvement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    print("\n[OK] Improved experiments complete")


if __name__ == "__main__":
    main()
