import streamlit as st
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

# -------------------------
# Load Dataset (Google Drive)
# -------------------------
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?export=download&id=1fhuqUbqF9eSAUepYVg7iULCxLnmTy8KK"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return None

df = load_data()

if df is not None:
    st.session_state["data"] = df
    st.success("Dataset Loaded Successfully ✅")
else:
    st.warning("Dataset not loaded")

# -------------------------
# Load BERT Model
# -------------------------
@st.cache_resource
def load_bert():
    try:
        model_name = "distilbert-base-uncased"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        return tokenizer, model

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None


tokenizer, model = load_bert()

# -------------------------
# Prediction Function
# -------------------------
def predict_fake_job(text):
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        confidence = torch.max(probs).item()
        label = torch.argmax(probs).item()

        if label == 1:
            return "Fake", confidence
        else:
            return "Real", confidence

    except Exception as e:
        return "Error", 0

# -------------------------
# Sidebar
# -------------------------
st.sidebar.success("🚀 Fake Job Detection System")

# -------------------------
# Load CSS (optional)
# -------------------------
def load_css():
    css_file = os.path.join("assets", "styles.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -------------------------
# Navigation
# -------------------------
page = st.navigation([
    st.Page("pages/home.py", title="Home", icon="🏠"),
    st.Page("pages/results.py", title="Results", icon="📊"),
])

page.run()
