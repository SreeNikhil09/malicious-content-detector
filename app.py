import streamlit as st
import pickle
import joblib
import gdown
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Download malicious_url_model.pkl from Google Drive if not present
url_model_id = '1TUWuxkGtJD8kSEI9wP4UJxI_RXpt4bmh'
url_model_path = 'malicious_url_model.pkl'

if not os.path.exists(url_model_path):
    gdown.download(f"https://drive.google.com/uc?id={url_model_id}", url_model_path, quiet=False)

# Load models and vectorizers
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('email_spam_model.pkl', 'rb') as f:
    email_model = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

sms_model = load_model('sms_spam_model.h5')

with open('sms_tokenizer.pkl', 'rb') as f:
    sms_tokenizer = pickle.load(f)

with open('malicious_url_model.pkl', 'rb') as f:
    url_model = pickle.load(f)

# Streamlit App UI
st.title("Multi-Modal Threat Detection System")
tab1, tab2, tab3 = st.tabs(["Email Spam", "Malicious URL", "SMS Spam"])

with tab1:
    st.subheader("Email Spam Detection")
    user_input = st.text_area("Enter Email Text")
    if st.button("Check Email"):
        if user_input.strip():
            vec_input = tfidf_vectorizer.transform([user_input])
            result = email_model.predict(vec_input)[0]
            st.success("Spam Email Detected!" if result == 1 else "Not Spam.")
        else:
            st.warning("Please enter text.")

with tab2:
    st.subheader("Malicious URL Detection")
    url_input = st.text_input("Enter URL")
    if st.button("Check URL"):
        if url_input.strip():
            # Create features based on your dataset’s columns
            import pandas as pd
            input_df = pd.DataFrame([[url_input]], columns=['url'])
            input_df = input_df.reindex(columns=feature_columns, fill_value=0)
            result = url_model.predict(input_df)[0]
            st.success("Malicious URL!" if result == 1 else "Safe URL.")
        else:
            st.warning("Please enter a URL.")

with tab3:
    st.subheader("SMS Spam Detection")
    sms_input = st.text_area("Enter SMS Text")
    if st.button("Check SMS"):
        if sms_input.strip():
            sequence = sms_tokenizer.texts_to_sequences([sms_input])
            padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
            result = sms_model.predict(padded)[0][0]
            st.success("Spam SMS!" if result > 0.5 else "Not Spam.")
        else:
            st.warning("Please enter SMS text.")

st.sidebar.markdown("Developed as a Final Year Project")