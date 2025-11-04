# Spam Message Classifier

A simple, reproducible machine learning solution to classify messages as spam or not spam (ham).

## Setup

1. Create a Python virtual environment (Python 3.9-3.12):
```bash
python -m venv venv
source venv/bin/activate  # Unix
# or
.\venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a new model using the provided dataset:

```bash
python scripts/train.py --input datasets/sms_spam_no_header.csv --output models/spam_classifier.joblib --seed 42
```

Options:
- `--input`: Path to input CSV file (no header, format: label,message)
- `--output`: Path to save trained model
- `--seed`: Random seed for reproducibility (default: 42)

### Prediction

Classify new messages using a trained model:

```bash
# Single message
python scripts/predict.py --model models/spam_classifier.joblib --text "Win a free iPhone now!"

# Batch prediction from file
python scripts/predict.py --model models/spam_classifier.joblib --input messages.csv --output predictions.csv
```

Options:
- `--model`: Path to trained model file
- `--text`: Single message to classify
- `--input`: Path to input CSV file for batch prediction
- `--output`: Path to save prediction results (for batch mode)

## Project Structure

- `datasets/`: Raw data files
- `scripts/`: Python scripts for training and prediction
- `models/`: Saved model artifacts
- `requirements.txt`: Python package dependencies

## Performance Metrics

The baseline model typically achieves:
- Accuracy: To be measured
- Precision: To be measured
- Recall: To be measured
- F1 Score: To be measured

Training completes in under 2 minutes on CPU.
