# app.py

import streamlit as st
import joblib
import re 
# Model aur vectorizer load karo
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Email Spam Classifier")
st.write("Enter an email message to check if it's spam or not.")

# User input
email_text = st.text_area("✉️ Email Text", height=200)

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Text ko vectorize karo
        X = vectorizer.transform([email_text])
        prediction = model.predict(X)[0]

        if prediction == 1:
            st.error("🚫 This email is **SPAM**!")
        else:
            st.success("✅ This email is **NOT SPAM**.")
