"""Utility functions for spam classification."""

import pandas as pd
from typing import Tuple, Union
from pathlib import Path


def load_data(file_path: Union[str, Path]) -> Tuple[pd.Series, pd.Series]:
    """Load spam classification dataset from CSV.
    
    Args:
        file_path: Path to CSV file with no header, format: label,message
        
    Returns:
        Tuple of (messages, labels)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    try:
        # Read CSV without header
        df = pd.read_csv(file_path, header=None, names=['label', 'message'])
        
        # Basic validation
        if df.shape[1] != 2:
            raise ValueError("CSV must have exactly 2 columns: label,message")
        if df.isnull().any().any():
            raise ValueError("Dataset contains missing values")
            
        # Get messages and labels
        messages = df['message']
        labels = (df['label'] == 'spam').astype(int)  # Convert to binary
        
        print(f"Loaded {len(messages)} messages")
        print(f"Spam ratio: {labels.mean():.2%}")
        
        return messages, labels
        
    except pd.errors.EmptyDataError:
        raise ValueError("Empty CSV file")
    except pd.errors.ParserError:
        raise ValueError("Invalid CSV format")


def preprocess_text(text: str) -> str:
    """Apply basic preprocessing to text message.
    
    Args:
        text: Raw message text
        
    Returns:
        Preprocessed text
    """
    # Convert to string (handle non-string input)
    text = str(text)
    
    # Basic cleaning
    text = text.lower()  # lowercase
    text = ' '.join(text.split())  # normalize whitespace
    
    return text


def preprocess_texts(texts: pd.Series) -> pd.Series:
    """Preprocess a series of text messages.
    
    Args:
        texts: Series of raw messages
        
    Returns:
        Series of preprocessed messages
    """
    return texts.apply(preprocess_text)