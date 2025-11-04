"""Predict spam classification for new messages."""

import argparse
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Union

from utils import preprocess_text


def load_model(model_path: str) -> Dict:
    """Load trained model artifacts.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Dictionary containing vectorizer and classifier
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def predict_single(
    text: str,
    model: Dict
) -> Dict[str, Union[str, float]]:
    """Classify a single message.
    
    Args:
        text: Input message text
        model: Loaded model dictionary
        
    Returns:
        Dictionary with prediction and confidence
    """
    # Preprocess
    text = preprocess_text(text)
    
    # Vectorize
    X = model['vectorizer'].transform([text])
    
    # Predict
    label = model['classifier'].predict(X)[0]
    prob = model['classifier'].predict_proba(X)[0]
    confidence = prob[1] if label == 1 else prob[0]
    
    return {
        'text': text,
        'prediction': 'spam' if label == 1 else 'ham',
        'confidence': confidence
    }


def predict_batch(
    input_path: str,
    output_path: str,
    model: Dict
) -> None:
    """Classify messages from a CSV file.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save predictions
        model: Loaded model dictionary
    """
    # Load messages
    try:
        df = pd.read_csv(input_path)
        if 'message' not in df.columns:
            raise ValueError("Input CSV must have a 'message' column")
    except Exception as e:
        raise RuntimeError(f"Failed to read input file: {e}")
    
    # Preprocess
    messages = df['message'].apply(preprocess_text)
    
    # Vectorize
    X = model['vectorizer'].transform(messages)
    
    # Predict
    labels = model['classifier'].predict(X)
    probabilities = model['classifier'].predict_proba(X)
    
    # Save results
    results = pd.DataFrame({
        'message': df['message'],
        'prediction': ['spam' if l == 1 else 'ham' for l in labels],
        'confidence': [p[1] if l == 1 else p[0] for l, p in zip(labels, probabilities)]
    })
    results.to_csv(output_path, index=False)
    print(f"Saved predictions for {len(results)} messages to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict spam classification")
    parser.add_argument(
        '--model',
        required=True,
        help='Path to trained model file'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--text',
        help='Single message to classify'
    )
    group.add_argument(
        '--input',
        help='Path to input CSV file with messages'
    )
    parser.add_argument(
        '--output',
        help='Path to save prediction results (required for batch mode)'
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Single or batch prediction
    if args.text:
        result = predict_single(args.text, model)
        print(f"\nInput text: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        if not args.output:
            parser.error("--output is required for batch prediction")
        print(f"Processing messages from {args.input}...")
        predict_batch(args.input, args.output, model)


if __name__ == '__main__':
    main()
