# TweetEval Hate Speech Detection

This is the final project for STATS 507.

## Project Overview
This project studies hate speech detection on the TweetEval dataset. It compares classical machine learning models (such as TF-IDF with Logistic Regression and SVM) with transformer-based models (Twitter RoBERTa and fine-tuned versions). The main goal is to evaluate performance differences and understand model behavior.

## Main Methods
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Twitter RoBERTa baseline
- Fine-tuned Twitter RoBERTa

## Main Metric
- Macro-F1
- Accuracy

## Repository Structure
* scripts/ for running experiments
* src/ for preprocessing and evaluation code
* data/ for dataset files
* figures/ for plots used in the report
* artifacts/ for final results and summaries
