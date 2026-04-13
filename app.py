import streamlit as st
import os
import pandas as pd
import streamlit as st

url = "https://drive.google.com/uc?id=1mcyb_HcW21QiD56mMjuFj-pDIW7GPE_f"

df = pd.read_csv(url)

st.write("Dataset Loaded Successfully ✅")
st.write(df.head())
# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

# -------------------------
# Load CSS
# -------------------------
def load_css():
    css_file = os.path.join("assets", "styles.css")
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

