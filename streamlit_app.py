import streamlit as st
import joblib
import pandas as pd
import io
from typing import Dict, Any
from pathlib import Path

# Reuse preprocessing from scripts/utils.py
try:
    from scripts.utils import preprocess_text
except Exception:
    # Fallback: simple lowercase normalizer if import fails
    def preprocess_text(text: str) -> str:
        return str(text).lower().strip()


@st.cache_resource
def load_model(path: str) -> Dict[str, Any]:
    """Load and return model artifacts (vectorizer, classifier, metadata).

    Cached so repeated UI interactions don't reload from disk.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    # Expecting a dict with 'vectorizer', 'classifier', and optional 'metadata'
    return model


def predict_single(text: str, model: Dict[str, Any]) -> Dict[str, Any]:
    text_proc = preprocess_text(text)
    vec = model['vectorizer'].transform([text_proc])
    label = model['classifier'].predict(vec)[0]
    if hasattr(model['classifier'], 'predict_proba'):
        probs = model['classifier'].predict_proba(vec)[0]
        confidence = float(probs[1] if label == 1 else probs[0])
    else:
        confidence = 1.0
    return {
        'raw_text': text,
        'processed_text': text_proc,
        'prediction': 'spam' if int(label) == 1 else 'ham',
        'confidence': confidence
    }


def predict_batch(df: pd.DataFrame, model: Dict[str, Any]) -> pd.DataFrame:
    # Accept either a 'message' column or headerless two-column CSV (label,message)
    if 'message' not in df.columns:
        # Try headerless: assume first column label, second column message
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ['label', 'message']
        else:
            raise ValueError("Input CSV must contain a 'message' column or be a 2-column CSV (label,message)")

    messages = df['message'].astype(str).apply(preprocess_text)
    X = model['vectorizer'].transform(messages)
    labels = model['classifier'].predict(X)
    if hasattr(model['classifier'], 'predict_proba'):
        probs = model['classifier'].predict_proba(X)
        confidences = [float(p[1] if l == 1 else p[0]) for l, p in zip(labels, probs)]
    else:
        confidences = [1.0] * len(labels)

    results = df.copy()
    results['prediction'] = ['spam' if int(l) == 1 else 'ham' for l in labels]
    results['confidence'] = confidences
    return results


def main():
    st.set_page_config(page_title="Spam Classifier", layout="wide")
    st.title("Spam Message Classifier")

    st.sidebar.header("Model and Input")
    default_model = "models/spam_classifier.joblib"
    model_path = st.sidebar.text_input("Model path", value=default_model)

    load_btn = st.sidebar.button("Load model")

    model = None
    if load_btn:
        try:
            with st.spinner("Loading model..."):
                model = load_model(model_path)
            st.sidebar.success("Model loaded")
            # show metadata if present
            meta = model.get('metadata', {}) if isinstance(model, dict) else {}
            if meta:
                st.sidebar.markdown("**Model metadata**")
                for k, v in meta.items():
                    st.sidebar.write(f"- **{k}**: {v}")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")

    # If model hasn't been loaded yet but file exists, try auto-loading silently
    if model is None and Path(model_path).exists():
        try:
            model = load_model(model_path)
        except Exception:
            model = None

    st.header("Single message prediction")
    col1, col2 = st.columns([3, 1])

    with col1:
        text = st.text_area("Enter message to classify", height=120)
        if st.button("Predict on text"):
            if model is None:
                st.error("Load a model first (use the sidebar or make sure the default model path is correct)")
            elif not text.strip():
                st.warning("Please enter a message to classify.")
            else:
                try:
                    res = predict_single(text, model)
                    st.success(f"Prediction: {res['prediction'].upper()} (confidence: {res['confidence']:.4f})")
                    st.markdown("**Processed text**")
                    st.write(res['processed_text'])
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    with col2:
        st.markdown("### Quick examples")
        st.write("- URGENT! You have won a free iPhone. Click now!")
        st.write("- Hi Mom, what time should I come over?")
        st.write("- 20% off at your local store this weekend!")

    st.markdown("---")
    st.header("Batch prediction (CSV)")
    st.write("Upload a CSV with a 'message' column or a headerless 2-column CSV (label,message).")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"]) 

    if uploaded is not None:
        try:
            # Read with pandas; allow headerless
            df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, header=None)

        st.write(f"Loaded {len(df)} rows")
        st.dataframe(df.head())

        if model is None:
            st.error("Load a model first before running batch predictions.")
        else:
            if st.button("Run batch prediction"):
                try:
                    with st.spinner("Running predictions..."):
                        results = predict_batch(df, model)
                    st.success(f"Predicted {len(results)} messages")
                    st.dataframe(results.head())

                    # Prepare CSV for download
                    csv_buf = io.StringIO()
                    results.to_csv(csv_buf, index=False)
                    csv_bytes = csv_buf.getvalue().encode('utf-8')
                    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

    st.markdown("---")
    st.caption("Tip: If you're missing packages, install them with `pip install -r requirements.txt`.")


if __name__ == '__main__':
    main()
