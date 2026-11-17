from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "phishing_model.joblib"

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ Phishing Email Detection")
st.write("Paste an email below to estimate whether it is **Phishing** or **Safe**.")

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Run `python train_model.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
email = st.text_area(
    "Email content",
    height=230,
    placeholder="Paste the subject and body of an email here..."
)

if st.button("Analyze Email", type="primary", use_container_width=True):
    if not email.strip():
        st.error("Please paste an email first.")
    else:
        prediction = int(model.predict([email])[0])
        probabilities = model.predict_proba([email])[0]
        confidence = float(probabilities[prediction]) * 100

        if prediction == 1:
            st.error("🚨 Prediction: PHISHING")
        else:
            st.success("✅ Prediction: SAFE")
        st.metric("Model confidence", f"{confidence:.1f}%")
        st.caption(
            "Confidence is the model's estimated probability, not a guarantee. "
            "Do not click links or open attachments based only on this result."
        )

with st.expander("What does this model inspect?"):
    st.markdown("""
    - **Text patterns:** TF-IDF word and phrase features.
    - **URL indicators:** number and length of URLs, IP-address URLs, unusual subdomains,
      hyphens and selected uncommon top-level domains.
    - **Language indicators:** urgency, account-verification wording, punctuation and message length.

    This is a learning/demo detector. It does not visit URLs, verify sender identity,
    inspect attachments, or replace a mail security gateway.
    """)
