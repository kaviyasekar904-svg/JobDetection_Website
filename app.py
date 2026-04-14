import streamlit as st
import pandas as pd
import gdown
import os

# -------------------------
# Page Config (MUST BE FIRST)
# -------------------------
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

# -------------------------
# Load Dataset (Google Drive FIXED)
# -------------------------
@st.cache_data
def load_data():
    file_id = "1mcyb_HcW21QiD56mMjuFj-pDIW7GPE_f"
    output = "dataset.csv"

    try:
        # Download only once
        if not os.path.exists(output):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=False)

        df = pd.read_csv(output)
        return df

    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return None


df = load_data()

# -------------------------
# Store Dataset in Session
# -------------------------
if df is not None:
    st.session_state["data"] = df
    st.success("Dataset Loaded Successfully ✅")
else:
    st.warning("Dataset not loaded")

# -------------------------
# Sidebar Info (optional)
# -------------------------
st.sidebar.success("🚀 Fake Job Detection System")

# -------------------------
# Load CSS
# -------------------------
def load_css():
    css_file = os.path.join("assets", "styles.css")
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------
# Navigation
# -------------------------
page = st.navigation([
    st.Page("pages/home.py", title="Home", icon="🏠"),
    st.Page("pages/results.py", title="Results", icon="📊"),
])

page.run()
