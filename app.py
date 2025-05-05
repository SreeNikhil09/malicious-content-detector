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
    if st.button("Detect URL"):
        if url_input.strip():
            import pandas as pd
            from urllib.parse import urlparse

            def extract_features(url):
                return {
                    'url_length': len(url),
                    'hostname_length': len(urlparse(url).hostname or ''),
                    'path_length': len(urlparse(url).path),
                    'count_@': url.count('@'),
                    'count_-': url.count('-'),
                    'count_?': url.count('?'),
                    'count_=': url.count('='),
                    'count_.': url.count('.'),
                    'count_http': url.count('http'),
                    'count_https': url.count('https'),
                    'count_www': url.count('www'),
                    'count_digits': sum(c.isdigit() for c in url),
                    'count_letters': sum(c.isalpha() for c in url),
                }

            input_df = pd.DataFrame([extract_features(url_input)], columns=feature_columns)
            prediction = url_model.predict(input_df)[0]
            result = "Malicious" if prediction == 1 else "Safe"
            st.success(f"Prediction: {result}")
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
