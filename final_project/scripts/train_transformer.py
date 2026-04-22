"""Transformer experiment script for TweetEval hate speech detection.

This script supports two strong workflows:
1. Evaluate an official task-specific Twitter model directly with `--eval_only`.
2. Fine-tune a Twitter encoder on the packaged TweetEval hate splits.
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tweeteval_hate.data_utils import load_hf_splits, load_local_splits, project_root
from src.tweeteval_hate.evaluation import metric_bundle, plot_confusion_matrix

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-dec2021-hate"
CARDIFF_TWITTER_MODEL_MARKERS = (
    "cardiffnlp/twitter-roberta-base",
    "cardiffnlp/twitter-roberta-large",
    "twitter-roberta-base-dec2021",
    "twitter-roberta-base-2021-124m",
    "twitter-roberta-base-hate",
    "twitter-roberta-base-hate-latest",
)
MENTION_RE = re.compile(r"^@\w+$")
URL_RE = re.compile(r"^(https?://|www\.)", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default="outputs/transformer")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--tune_threshold", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--preprocess_style", choices=["auto", "cardiff", "none"], default="auto")
    return parser.parse_args()


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def is_cardiff_twitter_model(model_name: str) -> bool:
    lower_name = model_name.lower()
    return any(marker in lower_name for marker in CARDIFF_TWITTER_MODEL_MARKERS)


def preprocess_cardiff_tweet(text: str) -> str:
    tokens = []
    for token in str(text).split():
        if MENTION_RE.match(token):
            tokens.append("@user")
        elif URL_RE.match(token):
            tokens.append("http")
        else:
            tokens.append(token)
    return " ".join(tokens)


def preprocess_text(text: str, model_name: str, preprocess_style: str) -> str:
    if preprocess_style == "cardiff":
        return preprocess_cardiff_tweet(text)
    if preprocess_style == "none":
        return str(text)
    if is_cardiff_twitter_model(model_name):
        return preprocess_cardiff_tweet(text)
    return str(text)


def tokenizer_kwargs(model_name: str) -> Dict[str, Any]:
    lower_name = model_name.lower()
    if "bertweet" in lower_name:
        return {"use_fast": False, "normalization": True}
    return {}


def load_splits() -> Tuple[Dict[str, pd.DataFrame], str]:
    try:
        return load_hf_splits(), "huggingface"
    except Exception:
        return load_local_splits(), "local_csv_fallback"


def frames_to_datasetdict(frames: Dict[str, pd.DataFrame], model_name: str, preprocess_style: str) -> DatasetDict:
    datasets_dict: Dict[str, Dataset] = {}
    for split_name, frame in frames.items():
        clean_frame = frame[["text", "label"]].copy()
        clean_frame["text"] = clean_frame["text"].fillna("").astype(str).map(lambda value: preprocess_text(value, model_name, preprocess_style))
        clean_frame["label"] = clean_frame["label"].astype(int)
        datasets_dict[split_name] = Dataset.from_pandas(clean_frame, preserve_index=False)
    return DatasetDict(datasets_dict)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probs[:, 1]


def threshold_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Tuple[Dict[str, Any], np.ndarray]:
    preds = (probs >= threshold).astype(int)
    return metric_bundle(y_true, preds), preds


def select_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict[str, Any], np.ndarray]:
    best_threshold = 0.5
    best_metrics, best_preds = threshold_metrics(y_true, probs, best_threshold)
    best_score = float(best_metrics["macro_f1"])

    for threshold in np.linspace(0.05, 0.95, 181):
        metrics, preds = threshold_metrics(y_true, probs, float(threshold))
        score = float(metrics["macro_f1"])
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_preds = preds
            best_score = score

    return best_threshold, best_metrics, best_preds


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def resolve_output_dir(root: Path, output_dir: str) -> Path:
    path = Path(output_dir)
    return path if path.is_absolute() else root / path


def build_training_arguments(args: argparse.Namespace, output_dir: Path, fp16: bool) -> TrainingArguments:
    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": 0 if args.eval_only else args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_strategy": "epoch" if not args.eval_only else "no",
        "logging_strategy": "steps",
        "logging_steps": 25,
        "save_total_limit": 2,
        "load_best_model_at_end": not args.eval_only,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": fp16,
        "dataloader_num_workers": 0,
    }

    training_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_signature.parameters:
        kwargs["evaluation_strategy"] = "epoch" if not args.eval_only else "no"
    else:
        kwargs["eval_strategy"] = "epoch" if not args.eval_only else "no"

    return TrainingArguments(**kwargs)


def build_trainer(
    model: AutoModelForSequenceClassification,
    training_args: TrainingArguments,
    encoded_dataset: DatasetDict,
    collator: DataCollatorWithPadding,
    tokenizer: AutoTokenizer,
    callbacks: list[EarlyStoppingCallback],
) -> Trainer:
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": encoded_dataset["train"],
        "eval_dataset": encoded_dataset["validation"],
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    return Trainer(**trainer_kwargs)


def save_predictions(
    out_path: Path,
    split_frame: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
) -> None:
    prediction_frame = split_frame[["text", "label"]].copy()
    prediction_frame["prob_hate"] = probs
    prediction_frame["prediction"] = preds
    prediction_frame.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root = project_root()
    output_dir = resolve_output_dir(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = root / "artifacts"
    figure_dir = root / "figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    model_slug = slugify_model_name(args.model_name)

    frames, data_source = load_splits()
    dataset = frames_to_datasetdict(frames, args.model_name, args.preprocess_style)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs(args.model_name))

    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    encoded_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    fp16 = torch.cuda.is_available() and not args.no_fp16
    training_args = build_training_arguments(args, output_dir, fp16)

    callbacks = []
    if not args.eval_only and args.patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer = build_trainer(model, training_args, encoded_dataset, collator, tokenizer, callbacks)

    if not args.eval_only:
        trainer.train()

    validation_output = trainer.predict(encoded_dataset["validation"], metric_key_prefix="validation")
    test_output = trainer.predict(encoded_dataset["test"], metric_key_prefix="test")

    y_val = np.asarray(frames["validation"]["label"].astype(int))
    y_test = np.asarray(frames["test"]["label"].astype(int))
    val_probs = probabilities_from_logits(validation_output.predictions)
    test_probs = probabilities_from_logits(test_output.predictions)

    selected_threshold = 0.5
    validation_metrics, val_preds = threshold_metrics(y_val, val_probs, selected_threshold)
    if args.tune_threshold:
        selected_threshold, validation_metrics, val_preds = select_best_threshold(y_val, val_probs)

    test_metrics, test_preds = threshold_metrics(y_test, test_probs, selected_threshold)

    plot_confusion_matrix(
        np.asarray(validation_metrics["confusion_matrix"]),
        figure_dir / f"confusion_matrix_{model_slug}_validation.png",
        f"{args.model_name} validation",
    )
    plot_confusion_matrix(
        np.asarray(test_metrics["confusion_matrix"]),
        figure_dir / f"confusion_matrix_{model_slug}_test.png",
        f"{args.model_name} test",
    )

    if args.save_predictions:
        save_predictions(artifact_dir / f"transformer_{model_slug}_validation_predictions.csv", frames["validation"], val_probs, val_preds)
        save_predictions(artifact_dir / f"transformer_{model_slug}_test_predictions.csv", frames["test"], test_probs, test_preds)

    summary = {
        "model_name": args.model_name,
        "data_source": data_source,
        "eval_only": args.eval_only,
        "threshold_strategy": "validation_macro_f1_search" if args.tune_threshold else "fixed_0.5",
        "selected_threshold": selected_threshold,
        "validation_macro_f1": float(validation_metrics["macro_f1"]),
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "output_dir": output_dir,
    }

    payload = {
        "config": vars(args),
        "summary": summary,
        "trainer_validation_metrics": validation_output.metrics,
        "trainer_test_metrics": test_output.metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    summary_path = artifact_dir / f"transformer_{model_slug}_summary.json"
    metrics_path = artifact_dir / f"transformer_{model_slug}_metrics.json"
    summary_path.write_text(json.dumps(to_serializable(summary), indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")

    print(json.dumps(to_serializable(summary), indent=2))
    print(f"[OK] transformer results saved to {summary_path}")


if __name__ == "__main__":
    main()
