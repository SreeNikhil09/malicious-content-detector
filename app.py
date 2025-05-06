import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Set the base URL of your GitHub raw content
BASE_URL = "https://raw.githubusercontent.com/SreeNikhil09/malicious-content-detector/main/"

@st.cache_resource
def load_joblib_model(file_name):
    return joblib.load(file_name)

@st.cache_resource
def load_pickle_file(file_name):
    import pickle
    with open(file_name, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_keras_model(file_name):
    return load_model(file_name)

# Load all models and vectorizers
email_model = load_joblib_model("email_spam_model.pkl")
tfidf_vectorizer = load_joblib_model("tfidf_vectorizer.pkl")
sms_model = load_keras_model("sms_spam_model.h5")
sms_tokenizer = load_joblib_model("sms_tokenizer.pkl")
url_model = load_joblib_model("malicious_url_model.pkl")
feature_columns = load_joblib_model("feature_columns.pkl")

# --- Streamlit UI ---
st.set_page_config(page_title="Malicious Content Detector", layout="centered")
st.title("Malicious Content Detection App")

tabs = st.tabs(["Malicious Email", "Malicious URL", "Malicious SMS"])

# --- Email Tab ---
with tabs[0]:
    st.header("Email Spam Detection")
    email_input = st.text_area("Enter Email Text")
    if st.button("Detect Email"):
        if email_input.strip():
            vectorized_input = tfidf_vectorizer.transform([email_input])
            prediction = email_model.predict(vectorized_input)
            result = "Spam" if prediction == 1 else "Not Spam"
            st.success(f"Prediction: {result}")
        else:
            st.warning("Please enter some text.")

# --- URL Tab ---
with tabs[1]:
    st.header("Malicious URL Detection")
    url_input = st.text_input("Enter URL")

    def extract_features_from_url(url, feature_columns):
        import re
        import tldextract
        import pandas as pd

        features = {}
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['has_https'] = int('https' in url)
        features['has_ip'] = int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url)))

        extracted = tldextract.extract(url)
        features['domain_length'] = len(extracted.domain)
        suffix = extracted.suffix

        df = pd.DataFrame([features])

        # One-hot encode suffix
        for col in feature_columns:
            if col.startswith("suffix_"):
                df[col] = 1 if f"suffix_{suffix}" == col else 0

        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_columns]
        return df

    if st.button("Detect URL"):
        if url_input.strip():
            try:
                input_df = extract_features_from_url(url_input, feature_columns)
                prediction = url_model.predict(input_df)[0]
                result = "Malicious" if prediction == 1 else "Safe"
                st.success(f"Prediction: {result}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            st.warning("Please enter a URL.")

# --- SMS Tab ---
with tabs[2]:
    st.header("Malicious SMS Detection")
    sms_input = st.text_area("Enter SMS Text")
    if st.button("Detect SMS"):
        if sms_input.strip():
            sequence = sms_tokenizer.texts_to_sequences([sms_input])
            padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
            prediction = sms_model.predict(padded)[0][0]
            result = "Spam" if prediction > 0.5 else "Not Spam"
            st.success(f"Prediction: {result}")
        else:
            st.warning("Please enter an SMS message.")
