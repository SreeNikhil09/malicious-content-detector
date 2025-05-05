import streamlit as st
import pickle
import requests
import io
import joblib
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Malicious Content Detector", layout="centered")

# ========== Load Files from GitHub ==========
@st.cache_resource
def load_pickle_from_github(url):
    response = requests.get(url)
    return pickle.load(io.BytesIO(response.content))

@st.cache_resource
def load_model_from_github(url, filename="temp_model.h5"):
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    return load_model(filename)

# Base URL of your raw GitHub files
base_url = "https://raw.githubusercontent.com/SreeNikhil09/malicious-content-detector/main/"

# Load models
url_model = load_pickle_from_github(base_url + "malicious_url_model.pkl")
email_model = load_pickle_from_github(base_url + "email_spam_model.pkl")
vectorizer = load_pickle_from_github(base_url + "tfidf_vectorizer.pkl")
sms_model = load_model_from_github(base_url + "sms_spam_model.h5")
sms_tokenizer = load_pickle_from_github(base_url + "sms_tokenizer.pkl")

# ========== UI ==========
st.title("🚨 Malicious Content Detection System")

tab1, tab2, tab3 = st.tabs(["📧 Email Detection", "🔗 URL Detection", "📱 SMS Detection"])

# --------- Email Detection ----------
with tab1:
    st.subheader("Detect if an Email is Spam")
    email_input = st.text_area("Enter Email Content:")
    if st.button("Check Email"):
        if email_input.strip() == "":
            st.warning("Please enter an email.")
        else:
            transformed = vectorizer.transform([email_input])
            prediction = email_model.predict(transformed)[0]
            st.success("Result: SPAM ❌" if prediction == 1 else "Result: HAM ✅")

# --------- URL Detection ----------
with tab2:
    st.subheader("Check if a URL is Malicious")
    url_input = st.text_input("Enter the URL:")
    if st.button("Check URL"):
        if url_input.strip() == "":
            st.warning("Please enter a URL.")
        else:
            # Feature extraction (based on your model)
            def extract_features(url):
                return np.array([
                    len(url),
                    url.count('.'),
                    url.count('/'),
                    int(bool(re.search(r'\d', url))),  # Has numbers
                    int("https" in url),
                    int("@" in url)
                ]).reshape(1, -1)

            features = extract_features(url_input)
            prediction = url_model.predict(features)[0]
            st.success("Result: MALICIOUS ❌" if prediction == 1 else "Result: SAFE ✅")

# --------- SMS Detection ----------
with tab3:
    st.subheader("Detect if an SMS is Spam")
    sms_input = st.text_area("Enter SMS Content:")
    if st.button("Check SMS"):
        if sms_input.strip() == "":
            st.warning("Please enter an SMS message.")
        else:
            sequences = sms_tokenizer.texts_to_sequences([sms_input])
            padded = pad_sequences(sequences, maxlen=100)
            prediction = sms_model.predict(padded)[0][0]
            st.success("Result: SPAM ❌" if prediction > 0.5 else "Result: HAM ✅")
