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
    "registration fee", "upfront payment", "deposit", "payment to apply",
    "pay to apply", "quick money", "earn per day", "easy money",
    "no experience", "immediate joining", "urgent hiring",
    "limited slots", "work from home", "guaranteed income",
    "wire transfer", "bank account", "gift card", "crypto",
    "whatsapp", "telegram", "skype", "confidential",
]

CONTACT_KEYWORDS = [
    "whatsapp", "telegram", "skype",
    "contact us on", "message us on",
]

# -------------------------
# Helper Functions
# -------------------------
def build_reasons(text):
    text_lower = text.lower()
    matched = [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]
    contact_hits = [kw for kw in CONTACT_KEYWORDS if kw in text_lower]

    reasons = []

    if matched:
        reasons.append("Suspicious keywords detected: " + ", ".join(sorted(set(matched))))

    if contact_hits:
        reasons.append("External contact methods found (WhatsApp/Telegram/Skype).")

    if "no experience" in text_lower and ("salary" in text_lower):
        reasons.append("No experience + salary promise pattern detected.")

    if not reasons:
        reasons.append("No major red flags found.")

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
        return 0, [0.5, 0.5]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        logits = bert_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().tolist()

    if not isinstance(probs, list):
        probs = [float(probs)]

    if len(probs) < 2:
        probs = [1 - probs[0], probs[0]]

    pred_id = 1 if probs[1] >= probs[0] else 0

    return pred_id, probs

# -------------------------
# UI SETTINGS
# -------------------------
compact_mode = st.sidebar.toggle("Compact View", value=True)

if compact_mode:
    st.markdown("<h2>Analysis Results</h2>", unsafe_allow_html=True)
else:
    st.title("Analysis Results")

# -------------------------
# SESSION CHECK
# -------------------------
if "analysis_results" not in st.session_state:
    st.warning("No results found. Please analyze a job first.")
    st.page_link("pages/home.py", label="← Back to Home")
    st.stop()

results = st.session_state.analysis_results

text = results["text"]
url = results["url"]
decision_threshold = results["decision_threshold"]
show_uncertain = results["show_uncertain"]

# ✅ FIXED (no crash)
flip_label_mapping = results.get("flip_label_mapping", False)

# -------------------------
# DISPLAY URL
# -------------------------
st.caption(f"🔗 {url}")

# -------------------------
# PREDICTION
# -------------------------
prediction, probs = predict_with_bert(text)

real_prob = probs[0]
fake_prob = probs[1]

if flip_label_mapping:
    real_prob, fake_prob = fake_prob, real_prob

# -------------------------
# ANALYSIS
# -------------------------
reasons, matched_keywords = build_reasons(text)

preview_text = build_preview(text, 300 if compact_mode else 900)
highlighted_preview = highlight_keywords(preview_text, matched_keywords)

# -------------------------
# DECISION LOGIC
# -------------------------
if show_uncertain and max(real_prob, fake_prob) < decision_threshold:
    label = "⚠️ NEEDS REVIEW"
    css_class = "review"
    confidence = max(real_prob, fake_prob)

elif fake_prob >= real_prob and fake_prob >= decision_threshold:
    label = "🚨 FAKE JOB DETECTED"
    css_class = "fake"
    confidence = fake_prob

else:
    label = "✅ REAL JOB POSTING"
    css_class = "real"
    confidence = real_prob

# -------------------------
# RESULT CARD
# -------------------------
st.markdown(
    f"""
    <div class="result-card {css_class}">
        <h3>{label}</h3>
        <p>Confidence: {confidence*100:.2f}%</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# PROBABILITIES
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Real", f"{real_prob*100:.2f}%")

with col2:
    st.metric("Fake", f"{fake_prob*100:.2f}%")

# -------------------------
# REASONS
# -------------------------
if not compact_mode:
    st.subheader("Reasons")

    for r in reasons:
        st.write("•", r)

# -------------------------
# PREVIEW
# -------------------------
if not compact_mode:
    st.subheader("Extracted Text Preview")

    st.markdown(
        f"<div class='preview-box'>{highlighted_preview}</div>",
        unsafe_allow_html=True
    )

# -------------------------
# BACK BUTTON
# -------------------------
st.divider()
st.page_link("pages/home.py", label="← Analyze Another Job")
