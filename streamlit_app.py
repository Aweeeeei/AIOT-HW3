import streamlit as st
import joblib
import pandas as pd
import io
import requests
import os
from typing import Dict, Any
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import TruncatedSVD

# Reuse preprocessing from scripts/utils.py when available
try:
    from scripts.utils import preprocess_text
except Exception:
    def preprocess_text(text: str) -> str:
        return str(text).lower().strip()


@st.cache_resource
def load_model_from_path(path: str) -> Dict[str, Any]:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    return model


def load_model_from_bytes(b: bytes) -> Dict[str, Any]:
    bio = io.BytesIO(b)
    model = joblib.load(bio)
    return model


def download_model_to_bytes(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def predict_single(text: str, model: Dict[str, Any]) -> Dict[str, Any]:
    text_proc = preprocess_text(text)
    vec = model['vectorizer'].transform([text_proc])
    label = model['classifier'].predict(vec)[0]
    if hasattr(model['classifier'], 'predict_proba'):
        probs = model['classifier'].predict_proba(vec)[0]
        confidence = float(probs[1] if int(label) == 1 else probs[0])
    else:
        confidence = 1.0
    return {
        'raw_text': text,
        'processed_text': text_proc,
        'prediction': 'spam' if int(label) == 1 else 'ham',
        'confidence': confidence,
    }


def predict_batch(df: pd.DataFrame, model: Dict[str, Any]) -> pd.DataFrame:
    if 'message' not in df.columns:
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
        confidences = [float(p[1] if int(l) == 1 else p[0]) for l, p in zip(labels, probs)]
    else:
        confidences = [1.0] * len(labels)
    results = df.copy()
    results['prediction'] = ['spam' if int(l) == 1 else 'ham' for l in labels]
    results['confidence'] = confidences
    return results


def compute_embedding(model: Dict[str, Any], messages: pd.Series, sample: int = 500) -> pd.DataFrame:
    n = len(messages)
    if n == 0:
        return pd.DataFrame()
    if n > sample:
        messages = messages.sample(sample, random_state=42).reset_index(drop=True)
    X = model['vectorizer'].transform(messages.astype(str))
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(X)
    df = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'message': messages})
    return df


def main():
    st.set_page_config(page_title="Spam Classifier", layout="wide")
    st.title("Spam Message Classifier")

    st.sidebar.header("Model and Input")
    default_model = "models/spam_classifier.joblib"
    model_source = st.sidebar.selectbox("Model source", ["Repo path", "Upload .joblib", "Model URL"], index=0)
    model = None

    if model_source == "Repo path":
        model_path = st.sidebar.text_input("Model path (in repo)", value=default_model)
        if st.sidebar.button("Load model from repo path"):
            try:
                with st.spinner("Loading model from path..."):
                    model = load_model_from_path(model_path)
                st.sidebar.success("Model loaded from path")
            except Exception as e:
                st.sidebar.error(f"Failed to load model: {e}")
        if model is None and Path(model_path).exists():
            try:
                model = load_model_from_path(model_path)
            except Exception:
                model = None
    elif model_source == "Upload .joblib":
        uploaded_model = st.sidebar.file_uploader("Upload .joblib file", type=["joblib", "pkl"])
        if uploaded_model is not None:
            try:
                with st.spinner("Loading uploaded model..."):
                    b = uploaded_model.read()
                    model = load_model_from_bytes(b)
                st.sidebar.success("Model loaded from upload")
            except Exception as e:
                st.sidebar.error(f"Failed to load uploaded model: {e}")
    else:
        model_url = st.sidebar.text_input("Model URL (https)")
        if st.sidebar.button("Download & load model"):
            if not model_url:
                st.sidebar.error("Please provide a model URL.")
            else:
                try:
                    with st.spinner("Downloading model..."):
                        b = download_model_to_bytes(model_url)
                        model = load_model_from_bytes(b)
                    st.sidebar.success("Model downloaded and loaded")
                except Exception as e:
                    st.sidebar.error(f"Failed to download/load model: {e}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Note: On deployed Streamlit apps the repo may not contain local model files. Use Upload or Model URL if the default path isn't present.")

    tabs = st.tabs(["Predictions", "Analytics", "Model"])
    pred_tab, analytics_tab, model_tab = tabs

    # ---------------- Predictions tab ----------------
    with pred_tab:
        st.header("Single message prediction")
        col1, col2 = st.columns([3, 1])
        with col1:
            # 初始化 session_state 儲存文字框內容
            if 'input_text' not in st.session_state:
                st.session_state['input_text'] = ''

            # 按鈕載入範例 spam email
            if st.button("Load sample spam email"):
                st.session_state['input_text'] = "Congratulations! You have won a free iPhone. Click here to claim."

            # 文字輸入框綁定 session_state
            text = st.text_area("Enter message to classify", value=st.session_state['input_text'], height=120)

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
                            if 'message' not in df.columns:
                                df.rename(columns={df.columns[-1]: 'message'}, inplace=True)
                            results = predict_batch(df, model)
                        st.success(f"Predicted {len(results)} messages")
                        st.dataframe(results.head())
                        csv_buf = io.StringIO()
                        results.to_csv(csv_buf, index=False)
                        csv_bytes = csv_buf.getvalue().encode('utf-8')
                        st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
                        st.session_state['last_results'] = results
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")

    # ---------------- Analytics tab ----------------
    with analytics_tab:
        st.header("Prediction Analytics")

        # 選擇資料來源
        data_source = st.selectbox(
            "Select dataset for analytics",
            options=[
                "Default processed dataset (sms_spam_clean.csv)",
                "No-header dataset (sms_spam_no_header.csv)"
            ],
            index=0
        )

        results = st.session_state.get('last_results')

        # 如果沒有 batch prediction 結果，載入選擇的 CSV
        if results is None:
            if data_source == "Default processed dataset (sms_spam_clean.csv)":
                dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "processed", "sms_spam_clean.csv")
            else:
                dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "sms_spam_no_header.csv")

            if os.path.exists(dataset_path):
                try:
                    results = pd.read_csv(dataset_path, header=0 if "clean" in dataset_path else None)
                    st.success(f"✅ Loaded dataset: {os.path.basename(dataset_path)}")
                    if "no_head" in dataset_path:
                        if results.shape[1] >= 2:
                            results.columns = ['label', 'message']
                        else:
                            results.columns = ['message']
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")
                    results = None
            else:
                st.warning(f"Dataset not found: {dataset_path}")
                results = None

        if results is not None:
            if 'message' not in results.columns:
                if 'text' in results.columns:
                    results.rename(columns={'text':'message'}, inplace=True)
                else:
                    results.rename(columns={results.columns[-1]:'message'}, inplace=True)

            if 'prediction' not in results.columns or results['prediction'].isnull().all():
                if model is not None:
                    try:
                        results['prediction'] = predict_batch(results[['message']], model)['prediction']
                        if 'confidence' not in results.columns:
                            results['confidence'] = predict_batch(results[['message']], model)['confidence']
                    except Exception:
                        results['prediction'] = 'unknown'
                        results['confidence'] = 1.0
                else:
                    results['prediction'] = 'unknown'
                    results['confidence'] = 1.0

            if 'label' not in results.columns:
                results['label'] = results['prediction']

            # Message count distribution
            st.subheader('Message count distribution')
            counts = results['prediction'].value_counts().reset_index()
            counts.columns = ['label', 'count']
            fig_counts = px.bar(counts, x='label', y='count', color='label', title='Message count distribution')
            st.plotly_chart(fig_counts, use_container_width=True)

            # Confidence distribution
            if 'confidence' in results.columns:
                st.subheader('Confidence distribution')
                fig_conf = px.histogram(results, x='confidence', nbins=30, title='Prediction confidence distribution')
                st.plotly_chart(fig_conf, use_container_width=True)

            # Confusion matrix, ROC, PR, calibration
            if 'label' in results.columns and model is not None:
                y_true = results['label'].copy()
                if y_true.dtype == object:
                    y_true = (y_true == 'spam').astype(int)
                y_pred = (results['prediction'] == 'spam').astype(int)
                y_score = results.get('confidence', pd.Series(np.ones(len(results)))).values

                st.subheader('Confusion Matrix')
                cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
                fig_cm, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['spam', 'ham'], yticklabels=['spam', 'ham'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig_cm)

                # ROC
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                fig_roc = px.area(x=fpr, y=tpr, title=f'ROC curve (AUC={roc_auc:.3f})', labels={'x':'FPR','y':'TPR'})
                fig_roc.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash'))
                st.plotly_chart(fig_roc, use_container_width=True)

                # Precision-Recall
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                fig_pr = px.area(x=recall, y=precision, title=f'Precision-Recall (AP={ap:.3f})', labels={'x':'Recall','y':'Precision'})
                st.plotly_chart(fig_pr, use_container_width=True)

                # Calibration
                try:
                    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
                    fig_cal = px.line(x=prob_pred, y=prob_true, markers=True, title='Calibration curve')
                    fig_cal.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash'))
                    fig_cal.update_xaxes(title='Mean predicted probability')
                    fig_cal.update_yaxes(title='Fraction of positives')
                    st.plotly_chart(fig_cal, use_container_width=True)
                except Exception as e:
                    st.info(f'Calibration plot skipped: {e}')

            # Top tokens
            if model is not None and hasattr(model['classifier'], 'coef_') and hasattr(model['vectorizer'], 'get_feature_names_out'):
                st.subheader('Top tokens contributing to spam/ham')
                coefs = model['classifier'].coef_[0]
                feature_names = model['vectorizer'].get_feature_names_out()
                top_pos_idx = np.argsort(coefs)[-20:]
                top_neg_idx = np.argsort(coefs)[:20]
                top_pos = pd.DataFrame({'token': feature_names[top_pos_idx], 'weight': coefs[top_pos_idx]})
                top_neg = pd.DataFrame({'token': feature_names[top_neg_idx], 'weight': coefs[top_neg_idx]})
                fig_pos = px.bar(top_pos.sort_values('weight'), x='weight', y='token', orientation='h', title='Top positive tokens (spam)')
                fig_neg = px.bar(top_neg.sort_values('weight'), x='weight', y='token', orientation='h', title='Top negative tokens (ham)')
                st.plotly_chart(fig_pos, use_container_width=True)
                st.plotly_chart(fig_neg, use_container_width=True)

            # Embedding
            st.subheader('Feature-space embedding (TruncatedSVD, sampled)')
            try:
                msgs = results['message'].astype(str)
                emb_df = compute_embedding(model, msgs, sample=1000)
                if not emb_df.empty:
                    fig_emb = px.scatter(emb_df, x='x', y='y', hover_data=['message'], title='2D embedding (sampled)')
                    st.plotly_chart(fig_emb, use_container_width=True)
            except Exception as e:
                st.info(f'Embedding visualization skipped: {e}')

    # ---------------- Model tab ----------------
    with model_tab:
        st.header('Model information')
        if model is None:
            st.info('No model loaded')
        else:
            meta = model.get('metadata', {}) if isinstance(model, dict) else {}
            if meta:
                st.markdown('**Model metadata**')
                for k, v in meta.items():
                    st.write(f'- **{k}**: {v}')
            st.write('Classifier type: ', type(model['classifier']).__name__)
            st.write('Vectorizer type: ', type(model['vectorizer']).__name__)

    st.markdown('---')
    st.caption("Tip: If you're missing packages, install them with `pip install -r requirements.txt`.")


if __name__ == '__main__':
    main()