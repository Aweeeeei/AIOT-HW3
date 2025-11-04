"""Train spam classification model."""

import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

from utils import load_data, preprocess_texts


def train_model(
    input_path: str,
    output_path: str,
    random_seed: int = 42
) -> None:
    """Train and save spam classification model.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save model artifacts
        random_seed: Random seed for reproducibility
    """
    print("Loading data...")
    messages, labels = load_data(input_path)
    
    # Preprocess text
    print("Preprocessing messages...")
    messages = preprocess_texts(messages)
    
    # Split data
    print("Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        messages, labels,
        test_size=0.2,
        random_state=random_seed,
        stratify=labels
    )
    
    # Create and train vectorizer
    print("Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Limit vocabulary size
        min_df=2,  # Remove very rare words
        stop_words='english'  # Remove common English words
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train classifier
    print("Training logistic regression classifier...")
    start_time = time.time()
    classifier = LogisticRegression(
        random_state=random_seed,
        max_iter=200
    )
    classifier.fit(X_train_vec, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred,
        average='binary',
        pos_label=1
    )
    
    # Print metrics
    print("\nPerformance Metrics:")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_path}...")
    model = {
        'vectorizer': vectorizer,
        'classifier': classifier,
        'metadata': {
            'train_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': train_time
        }
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Train spam classification model")
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input CSV file (no header, format: label,message)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save trained model'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    train_model(args.input, args.output, args.seed)


if __name__ == '__main__':
    main()