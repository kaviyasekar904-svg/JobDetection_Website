import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
df = st.session_state.get("data")

st.write(df.head())

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

ENABLE_JS_RENDER = PLAYWRIGHT_AVAILABLE

BLOCKED_PHRASES = [
    "access denied",
    "enable javascript",
    "not a robot",
    "captcha",
    "unusual traffic",
    "verify you are human",
    "forbidden",
]

# Global variables for BERT model
tokenizer = None
bert_model = None
bert_loaded = False

def load_bert_model():
    global tokenizer, bert_model, bert_loaded
    if not bert_loaded:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained("saved_model")
            model = AutoModelForSequenceClassification.from_pretrained("saved_model")
            model.eval()
            bert_model = model
            bert_loaded = True
            return True
        except ImportError as e:
            st.error(f"Missing dependencies: {e}. Please install torch and transformers:")
            st.code("pip install torch transformers")
            return False
        except Exception as e:
            st.error(f"Error loading BERT model: {e}")
            return False
    return True

def extract_text_from_html(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

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
            "length": len(text),
        }

    return text, {"title": title, "length": len(text)}


def extract_text_with_playwright(url):
    if not PLAYWRIGHT_AVAILABLE:
        return None, {
            "error": "Playwright not available. Install it to enable JS-rendered pages."
        }
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=20000)
            html_text = page.content()
            browser.close()

        text, info = extract_text_from_html(html_text)
        if info is None:
            info = {}
        info["rendered"] = True
        return text, info

    except Exception as exc:
        return None, {"error": f"JS render failed: {exc}"}


def extract_text_from_url(url):
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            if ENABLE_JS_RENDER:
                return extract_text_with_playwright(url)
            return None, {"error": f"HTTP {response.status_code}"}

        text, info = extract_text_from_html(response.text)
        if text is None and ENABLE_JS_RENDER:
            return extract_text_with_playwright(url)
        return text, info

    except Exception as exc:
        if ENABLE_JS_RENDER:
            return extract_text_with_playwright(url)
        return None, {"error": f"Request failed: {exc}"}

# -------------------------
# Hero
# -------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI Risk Screening • Job Posting Verification</div>
        <div class="hero-title">Fake Job Detection</div>
        <div class="hero-sub">
            Evaluate job postings in seconds with a deep-learning classifier and
            red-flag keyword analysis. Built to help teams and job seekers identify
            suspicious listings quickly.
        </div>
        <div class="badge-row">
            <span class="badge">Research-grade model</span>
            <span class="badge">Explainable signals</span>
            <span class="badge">Privacy-first</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="trust-note">Use this as a screening tool and always verify with official sources.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-title">How it works</div>
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-title">1. Paste a job URL</div>
            <div class="feature-text">We extract the job description from the page.</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">2. Analyze signals</div>
            <div class="feature-text">A fine-tuned model + keyword heuristics assess risk.</div>
        </div>
        <div class="feature-card">
            <div class="feature-title">3. Review results</div>
            <div class="feature-text">Get a label, confidence score, and key reasons.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="section-title">Analyze a job post</div>', unsafe_allow_html=True)

# -------------------------
# URL Input Form
# -------------------------
col1, col2 = st.columns([4, 1])
with col1:
    url = st.text_input("Enter Job Posting URL", key="url_input")
with col2:
    analyze_button = st.button("Analyze Job", use_container_width=True)

# -------------------------
# Detection Settings
# -------------------------
with st.expander("Detection settings"):
    decision_threshold = st.slider(
        "Fraud risk threshold",
        min_value=0.5,
        max_value=0.9,
        value=0.65,
        step=0.01,
        help="Higher values reduce false positives but may miss some fraud.",
    )
    show_uncertain = st.checkbox(
        "Show Needs Review when confidence is low",
        value=True,
    )
    flip_label_mapping = st.checkbox(
        "Flip label mapping (advanced)",
        value=False,
        help="Use this only if your model's label order is reversed.",
    )

# -------------------------
# Analyze and store results in session state
# -------------------------
if analyze_button:
    if url == "":
        st.warning("Please enter a job URL")
    else:
        with st.spinner("Analyzing job posting..."):
            text, info = extract_text_from_url(url)

        if text is None:
            error_msg = info.get("error", "Could not extract job description from the URL")
            st.error(f"Could not extract job description from the URL ({error_msg})")
            if "Playwright not available" in error_msg:
                st.info("Tip: install Playwright to handle JavaScript-heavy job sites.")
            st.info("Please try another reachable job URL.")
        else:
            # Check if BERT model can be loaded
            if load_bert_model():
                # Save to session state and navigate to results
                st.session_state.analysis_results = {
                    "url": url,
                    "text": text,
                    "info": info,
                    "decision_threshold": decision_threshold,
                    "show_uncertain": show_uncertain,
                    "flip_label_mapping": flip_label_mapping,
                }
                st.switch_page("pages/results.py")
            else:
                st.error("Cannot proceed without BERT model. Please install the required dependencies.")

# -------------------------
# Analyze and store results in session state
# -------------------------
if analyze_button:
    if url == "":
        st.warning("Please enter a job URL")
    else:
        with st.spinner("Analyzing job posting..."):
            text, info = extract_text_from_url(url)

        if text is None:
            error_msg = info.get("error", "Could not extract job description from the URL")
            st.error(f"Could not extract job description from the URL ({error_msg})")
            if "Playwright not available" in error_msg:
                st.info("Tip: install Playwright to handle JavaScript-heavy job sites.")
            st.info("Please try another reachable job URL.")
        else:
            # Check if BERT model can be loaded
            if load_bert_model():
                # Save to session state and navigate to results
                st.session_state.analysis_results = {
                    "url": url,
                    "text": text,
                    "info": info,
                    "decision_threshold": decision_threshold,
                    "show_uncertain": show_uncertain,
                    "flip_label_mapping": flip_label_mapping,
                }
                st.switch_page("pages/results.py")
            else:
                st.error("Cannot proceed without BERT model. Please install the required dependencies.")
