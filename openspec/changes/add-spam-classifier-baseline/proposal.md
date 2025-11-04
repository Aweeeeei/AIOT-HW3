# Change: Add Baseline Spam Classifier Implementation

## Why
Implement the core spam classification functionality with a basic TF-IDF + Logistic Regression pipeline that provides a reproducible baseline for text message classification.

## What Changes
- Add training script with TF-IDF vectorization and Logistic Regression
- Add prediction script for single and batch inference
- Add dataset loading and preprocessing utilities
- Add model serialization with joblib
- Add basic performance metrics reporting
- Add example dataset and usage documentation

## Impact
- Affected specs: text-classification, data-management, model-management, cli-interface
- Affected code: 
  - `scripts/train.py` (new)
  - `scripts/predict.py` (new)
  - `scripts/utils.py` (new)
  - `requirements.txt` (new)
  - `README.md` (new)
  - `datasets/` (example data)
  - `models/` (trained artifacts)