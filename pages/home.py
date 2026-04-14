import streamlit as st
import requests
from bs4 import BeautifulSoup
import os

# -------------------------
# Disable Playwright (safe for deployment)
# -------------------------
PLAYWRIGHT_AVAILABLE = False
ENABLE_JS_RENDER = False

# -------------------------
# Blocked phrases
# -------------------------
BLOCKED_PHRASES = [
    "access denied",
    "enable javascript",
    "not a robot",
    "captcha",
    "unusual traffic",
    "verify you are human",
    "forbidden",
]

# -------------------------
# Global BERT variables
# -------------------------
tokenizer = None
bert_model = None
bert_loaded = False


# -------------------------
# Load BERT Model (FIXED)
# -------------------------
def load_bert_model():
    global tokenizer, bert_model, bert_loaded

    if not bert_loaded:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            model_name = "distilbert-base-uncased"

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            model.eval()
            bert_model = model
            bert_loaded = True
            return True

        except Exception as e:
            st.error(f"Error loading BERT model: {e}")
            return False

    return True


# -------------------------
# Extract text from HTML
# -------------------------
def extract_text_from_html(html_text):
    soup = BeautifulSoup(html_text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""

    # remove scripts
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    keywords = [
        "job",
        "description",
        "responsibilit",
        "qualification",
        "role",
        "about",
        "summary",
        "details",
    ]

    candidates = []

    for tag in soup.find_all(["article", "main", "section", "div"]):
        attr = " ".join(tag.get("class", [])) + " " + (tag.get("id", "") or "")
        attr = attr.lower()

        if any(k in attr for k in keywords):
            txt = tag.get_text(" ", strip=True)

            if len(txt) > 200:
                candidates.append(txt)

    if candidates:
        text = max(candidates, key=len)
    else:
        text = soup.get_text(" ", strip=True)

    text = " ".join(text.split())
    text_lower = text.lower()

    if any(p in text_lower for p in BLOCKED_PHRASES):
        return None, {
            "error": "Page blocked or requires JavaScript/CAPTCHA",
            "title": title,
        }

    return text, {"title": title}


# -------------------------
# Extract text from URL
# -------------------------
def extract_text_from_url(url):
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None, {"error": f"HTTP {response.status_code}"}

        return extract_text_from_html(response.text)

    except Exception as e:
        return None, {"error": str(e)}


# -------------------------
# UI SECTION
# -------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Fake Job Detection</div>
        <div class="hero-sub">
            Analyze job postings using AI to detect fraudulent listings.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

url = st.text_input("Enter Job Posting URL")

analyze_button = st.button("Analyze Job")

# Settings
with st.expander("Settings"):
    decision_threshold = st.slider("Threshold", 0.5, 0.9, 0.65)
    show_uncertain = st.checkbox("Show uncertain", True)

# -------------------------
# MAIN LOGIC
# -------------------------
if analyze_button:

    if url == "":
        st.warning("Please enter a URL")

    else:
        with st.spinner("Extracting job content..."):
            text, info = extract_text_from_url(url)

        if text is None:
            st.error(info.get("error", "Extraction failed"))

        else:
            with st.spinner("Loading AI model..."):

                if load_bert_model():

                    # Store everything for results page
                    st.session_state.analysis_results = {
                        "url": url,
                        "text": text,
                        "info": info,
                        "decision_threshold": decision_threshold,
                        "show_uncertain": show_uncertain,
                    }

                    st.success("Analysis Ready ✅")

                    st.switch_page("pages/results.py")

                else:
                    st.error("Model loading failed")
