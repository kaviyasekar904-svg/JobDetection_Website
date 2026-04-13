import streamlit as st
import os
import pandas as pd
st.session_state["data"] = df
# -------------------------
# Page Config (MUST BE FIRST)
# -------------------------
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

# -------------------------
# Load Dataset (Cached ✅)
# -------------------------
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1mcyb_HcW21QiD56mMjuFj-pDIW7GPE_f"
    return pd.read_csv(url)

df = load_data()

# Optional (debug)
st.write("Dataset Loaded Successfully ✅")
st.write(df.head())

# -------------------------
# Load CSS
# -------------------------
def load_css():
    css_file = os.path.join("assets", "styles.css")
    if os.path.exists(css_file):   # ✅ avoid crash
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
