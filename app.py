import streamlit as st
import pandas as pd

# -------------------------
# Page Config (MUST BE FIRST)
# -------------------------
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

# -------------------------
# Load Dataset (SAFE)
# -------------------------
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?export=download&id=1mcyb_HcW21QiD56mMjuFj-pDIW7GPE_f"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        return None

df = load_data()

# -------------------------
# Store in Session
# -------------------------
if df is not None:
    st.session_state["data"] = df
else:
    st.warning("Dataset not loaded")

# -------------------------
# Navigation
# -------------------------
page = st.navigation([
    st.Page("pages/home.py", title="Home", icon="🏠"),
    st.Page("pages/results.py", title="Results", icon="📊"),
])

page.run()
