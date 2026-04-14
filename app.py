import streamlit as st
import pandas as pd
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
# Load Dataset (YOUR LINK FIXED)
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

# -------------------------
# Store Dataset
# -------------------------
if df is not None:
    st.session_state["data"] = df
    st.success("Dataset Loaded Successfully ✅")
else:
    st.warning("Dataset not loaded")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.success("🚀 Fake Job Detection System")

# -------------------------
# Load CSS
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
