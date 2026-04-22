from __future__ import annotations

import json
import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

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


def build_vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=8000,
            sublinear_tf=True,
            lowercase=False,
        )
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=8000,
            sublinear_tf=True,
            lowercase=False,
        )
    raise ValueError(kind)


def plot_model_comparison(df: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(df))
    width = 0.36
    plt.figure(figsize=(6.8, 3.6))
    plt.bar(x - width / 2, df["val_macro_f1"], width, label="Validation macro-F1")
    plt.bar(x + width / 2, df["test_macro_f1"], width, label="Test macro-F1")
    plt.xticks(x, df["experiment"], rotation=15)
    plt.ylim(0.35, 0.8)
    plt.ylabel("Macro-F1")
    plt.title("Model comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_preprocessing_ablation(df: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(df))
    width = 0.36
    plt.figure(figsize=(5.8, 3.5))
    plt.bar(x - width / 2, df["val_macro_f1"], width, label="Validation macro-F1")
    plt.bar(x + width / 2, df["test_macro_f1"], width, label="Test macro-F1")
    plt.xticks(x, df["variant"], rotation=0)
    plt.ylim(0.4, 0.76)
    plt.ylabel("Macro-F1")
    plt.title("Preprocessing ablation (word logistic regression)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def top_feature_table(vectorizer: TfidfVectorizer, model: LogisticRegression) -> pd.DataFrame:
    names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    top_hate = pd.DataFrame({"feature": names[np.argsort(coef)[-15:][::-1]], "weight": np.sort(coef)[-15:][::-1], "class": "hate"})
    top_non_hate = pd.DataFrame({"feature": names[np.argsort(coef)[:15]], "weight": np.sort(coef)[:15], "class": "non_hate"})
    return pd.concat([top_hate, top_non_hate], ignore_index=True)


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

    experiments = [
        {
            "experiment": "word_lr",
            "vectorizer": build_vectorizer("word"),
            "model": LogisticRegression(max_iter=800, solver="liblinear", random_state=42, C=4),
        },
        {
            "experiment": "word_svc",
            "vectorizer": build_vectorizer("word"),
            "model": LinearSVC(C=1.0, random_state=42),
        },
        {
            "experiment": "char_svc",
            "vectorizer": build_vectorizer("char"),
            "model": LinearSVC(C=0.5, random_state=42),
        },
    ]

    rows = []
    payload = {}
    trained = {}
    for exp in experiments:
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
        rows.append(
            {
                "experiment": exp["experiment"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            }
        )
        payload[exp["experiment"]] = {"validation": val_metrics, "test": test_metrics}
        trained[exp["experiment"]] = {"vectorizer": vec, "model": model, "val_pred": val_pred, "test_pred": test_pred}

    result_df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    result_df.to_csv(art_dir / "experiment_summary.csv", index=False)
    (art_dir / "experiment_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_model_comparison(result_df, fig_dir / "model_comparison.png")

    ablation_rows = []
    for name, tr, va, te in [
        ("raw", x_train_raw, x_val_raw, x_test_raw),
        ("normalized", x_train, x_val, x_test),
    ]:
        vec = build_vectorizer("word")
        model = LogisticRegression(max_iter=800, solver="liblinear", random_state=42, C=4)
        X_train_a = vec.fit_transform(tr)
        X_val_a = vec.transform(va)
        X_test_a = vec.transform(te)
        model.fit(X_train_a, y_train)
        val_pred = model.predict(X_val_a)
        test_pred = model.predict(X_test_a)
        ablation_rows.append(
            {
                "variant": name,
                "val_accuracy": float((val_pred == y_val).mean()),
                "val_macro_f1": float(metric_bundle(y_val, val_pred)["macro_f1"]),
                "test_accuracy": float((test_pred == y_test).mean()),
                "test_macro_f1": float(metric_bundle(y_test, test_pred)["macro_f1"]),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(art_dir / "preprocessing_ablation.csv", index=False)
    plot_preprocessing_ablation(ablation_df, fig_dir / "preprocessing_ablation.png")

    best_name = result_df.iloc[0]["experiment"]
    best_vec = trained[best_name]["vectorizer"]
    best_model = trained[best_name]["model"]
    val_pred = trained[best_name]["val_pred"]
    test_pred = trained[best_name]["test_pred"]
    plot_confusion_matrix(confusion_matrix(y_val, val_pred), fig_dir / "confusion_matrix_validation.png", f"{best_name} validation")
    plot_confusion_matrix(confusion_matrix(y_test, test_pred), fig_dir / "confusion_matrix_test.png", f"{best_name} test")

    feature_df = top_feature_table(best_vec, best_model)
    feature_df.to_csv(art_dir / "top_features.csv", index=False)

    coef = best_model.coef_[0]
    names = np.array(best_vec.get_feature_names_out())
    order = np.argsort(np.abs(coef))[-12:]
    show_names = names[order]
    show_coef = coef[order]
    plt.figure(figsize=(6.6, 3.5))
    plt.barh(np.arange(len(show_names)), show_coef)
    plt.yticks(np.arange(len(show_names)), show_names)
    plt.title("Most influential lexical features (best validation model)")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(fig_dir / "top_feature_weights.png", dpi=220)
    plt.close()

    test_errors = test.copy()
    test_errors["prediction"] = test_pred
    test_errors["sanitized_text"] = test_errors["text"].map(sanitize_text)
    error_rows = []
    for error_type, cond in [
        ("false_positive", (test_errors["label"] == 0) & (test_errors["prediction"] == 1)),
        ("false_negative", (test_errors["label"] == 1) & (test_errors["prediction"] == 0)),
    ]:
        subset = test_errors.loc[cond, ["sanitized_text", "label", "prediction"]].head(8).copy()
        subset.insert(0, "error_type", error_type)
        error_rows.append(subset)
    error_df = pd.concat(error_rows, ignore_index=True)
    error_df.to_csv(art_dir / "error_examples.csv", index=False)

    summary = {
        "best_validation_model": best_name,
        "best_validation_macro_f1": float(result_df.iloc[0]["val_macro_f1"]),
        "corresponding_test_macro_f1": float(result_df.iloc[0]["test_macro_f1"]),
        "largest_drop_val_to_test_macro_f1": float((result_df["val_macro_f1"] - result_df["test_macro_f1"]).max()),
    }
    (art_dir / "best_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(result_df.to_string(index=False))
    print("\nAblation:")
    print(ablation_df.to_string(index=False))
    print("\n" + json.dumps(summary, indent=2))
    print("[OK] experiments complete")


if __name__ == "__main__":
    main()
