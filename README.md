# Credit Card Fraud Detection with TensorFlow/Keras

## Project Overview
This project builds a deep learning model to detect fraudulent credit card transactions. The primary challenge of this dataset is extreme class imbalance (fraud accounts for only 0.17% of all transactions). 

Instead of relying on standard accuracy—which is misleading for imbalanced data—this project optimizes for **Recall** and **Precision** using custom class weights.

## Key Engineering Highlights
* **Data Leakage Prevention:** Used `train_test_split` with `stratify=y` to ensure the exact ratio of fraud was maintained across splits. The `StandardScaler` was strictly fitted *only* on the training data before transforming the validation and test sets.
* **Deep Learning Architecture:** Built a 4-layer `Sequential` neural network in Keras using `ReLU` activations for hidden layers and a `Sigmoid` output layer for binary classification.
* **The Precision-Recall Trade-off:** Initially applied a mathematically proportional class weight of 568:1. While this caught almost all fraud (high Recall), it destroyed Precision (2%). By manually tuning the penalty down to 10:1, the model achieved a highly realistic balance.

## Final Test Set Results
After validating the model on a Dev set, it was evaluated against an unseen Test vault, achieving:
* **Recall:** ~87.7% (Successfully identifying the vast majority of actual fraud)
* **Precision:** ~75.4% (Ensuring innocent customers are rarely flagged)

## Dataset
The dataset used is the Kaggle Credit Card Fraud Detection dataset, which contains PCA-transformed numerical features for privacy. 
*(Note: The dataset CSV is excluded from this repository via `.gitignore` due to size constraints).*
