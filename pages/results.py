import streamlit as st
import html
import re

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

# -------------------------
# Load BERT Model
# -------------------------
@st.cache_resource
def load_bert():
    if not DEPENDENCIES_AVAILABLE:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained("saved_model")
        model = AutoModelForSequenceClassification.from_pretrained("saved_model")
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None

tokenizer, bert_model = load_bert()

# -------------------------
# Suspicious Keywords
# -------------------------
SUSPICIOUS_KEYWORDS = [
    "registration fee",
    "upfront payment",
    "deposit",
    "payment to apply",
    "pay to apply",
    "quick money",
    "earn per day",
    "easy money",
    "no experience",
    "immediate joining",
    "urgent hiring",
    "limited slots",
    "work from home",
    "guaranteed income",
    "wire transfer",
    "bank account",
    "gift card",
    "crypto",
    "whatsapp",
    "telegram",
    "skype",
    "confidential",
]

CONTACT_KEYWORDS = [
    "whatsapp",
    "telegram",
    "skype",
    "contact us on",
    "message us on",
]


def build_reasons(text):
    text_lower = text.lower()
    matched = [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]
    contact_hits = [kw for kw in CONTACT_KEYWORDS if kw in text_lower]

    reasons = []
    if matched:
        reasons.append(
            "Suspicious keywords detected: " + ", ".join(sorted(set(matched)))
        )
    if contact_hits:
        reasons.append(
            "External contact methods found (e.g., WhatsApp/Telegram/Skype)."
        )
    if "no experience" in text_lower and ("high salary" in text_lower or "salary" in text_lower):
        reasons.append(
            "Mentions no experience with salary promises, which is a common fraud pattern."
        )
    if not reasons:
        reasons.append("No obvious red-flag keywords were found in the extracted text.")

    return reasons, matched


def highlight_keywords(text, keywords):
    safe_text = html.escape(text)
    if not keywords:
        return safe_text
    keywords = sorted(set(keywords), key=len, reverse=True)
    pattern = re.compile(r"(" + "|".join(re.escape(k) for k in keywords) + r")", re.IGNORECASE)
    return pattern.sub(r"<span class='kw'>\1</span>", safe_text)


def build_preview(text, max_chars=900):
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return trimmed + "..."


def predict_with_bert(text):
    if tokenizer is None or bert_model is None:
        return 0, [0.5, 0.5]  # Return neutral prediction

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    if not isinstance(probs, list):
        probs = [float(probs)]

    if len(probs) < 2:
        probs = [1 - probs[0], probs[0]]

    # Choose label by highest probability to avoid class index mismatch
    pred_id = 1 if probs[1] >= probs[0] else 0

    return pred_id, probs


# -------------------------
# Results Page
# -------------------------
compact_mode = st.sidebar.toggle(
    "Single-page view (for screenshots)",
    value=True,
    help="Hides long sections and keeps the output compact.",
)

if compact_mode:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1 {
            font-size: 1.7rem;
            margin-bottom: 0.2rem;
        }
        h2 {
            font-size: 1.1rem;
            margin-top: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-title'>Analysis Results</div>", unsafe_allow_html=True)
else:
    st.title("Analysis Results")

# Check if we have analysis results in session state
if "analysis_results" not in st.session_state:
    st.warning("No analysis results found. Please go back to the Home page and analyze a job posting.")
    st.page_link("pages/home.py", label="← Back to Home")
    st.stop()

# Check if dependencies are available
if not DEPENDENCIES_AVAILABLE:
    st.error("⚠️ **Missing Dependencies**")
    st.write("The BERT model analysis requires additional packages. Please install them using:")
    st.code("pip install torch transformers")
    st.page_link("pages/home.py", label="← Back to Home")
    st.stop()

# Get analysis results
results = st.session_state.analysis_results
text = results["text"]
url = results["url"]
decision_threshold = results["decision_threshold"]
show_uncertain = results["show_uncertain"]
flip_label_mapping = results["flip_label_mapping"]



# Display URL
if compact_mode:
    st.caption(f"Job URL: {url}")
else:
    st.markdown(f"**Job URL:** {url}")
    st.divider()

# Perform BERT prediction
prediction, probs = predict_with_bert(text)
real_prob = probs[0]
fake_prob = probs[1]
if flip_label_mapping:
    real_prob, fake_prob = fake_prob, real_prob

reasons, matched_keywords = build_reasons(text)
preview_max = 320 if compact_mode else 900
preview_text = build_preview(text, max_chars=preview_max)
highlighted_preview = highlight_keywords(preview_text, matched_keywords)

if not compact_mode:
    st.write("")

# Determine result label
if show_uncertain and max(real_prob, fake_prob) < decision_threshold:
    result_label = "NEEDS REVIEW"
    result_class = "review"
    confidence = max(real_prob, fake_prob)
    reasons.append(
        f"Model confidence below threshold ({decision_threshold:.2f})."
    )
elif fake_prob >= real_prob and fake_prob >= decision_threshold:
    result_label = "FAKE JOB DETECTED"
    result_class = "fake"
    confidence = fake_prob
else:
    result_label = "REAL JOB POSTING"
    result_class = "real"
    confidence = real_prob

# Display result card and summary
if compact_mode:
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(
            f"""
            <div class="result-card {result_class}">
                <div class="result-title">{result_label}</div>
                <div class="result-confidence">Confidence Score: {confidence * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        st.subheader("Probabilities")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Real", f"{real_prob * 100:.1f}%")
        with prob_col2:
            st.metric("Fake", f"{fake_prob * 100:.1f}%")
else:
    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="result-title">{result_label}</div>
            <div class="result-confidence">Confidence Score: {confidence * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

reason_items = "".join(f"<li>{html.escape(r)}</li>" for r in reasons)
preview_class = "preview-card compact-preview" if compact_mode else "preview-card"

if not compact_mode:
    # Reasons
    st.subheader("Reasons & Keywords")

    st.markdown(
        f"""
        <div class="reason-card">
            <ul class="reason-list">{reason_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if matched_keywords:
        chips = "".join(
            f"<span class='chip'>{html.escape(k)}</span>"
            for k in sorted(set(matched_keywords))
        )
        st.markdown(
            f"<div class='chip-wrap'>{chips}</div>",
            unsafe_allow_html=True,
        )

    # Prediction Probabilities
    st.subheader("Prediction Probabilities")
    st.progress(float(real_prob))
    st.write(f"Real Job Probability: **{real_prob:.2f}**")

    st.progress(float(fake_prob))
    st.write(f"Fake Job Probability: **{fake_prob:.2f}**")

    # Extracted Preview
    st.subheader("Extracted Preview")
    st.markdown(
        f"<div class='{preview_class}'>{highlighted_preview}</div>",
        unsafe_allow_html=True,
    )

# Back button
if not compact_mode:
    st.divider()
    st.page_link("pages/home.py", label="← Analyze Another Job")

