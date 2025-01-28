import streamlit as st
import joblib as jb
import numpy as np
from preprocessing import tweets_cleaning


model = jb.load("hate_speech_model.pkl")
vectorizer = jb.load("vectorizer.pkl")

label_mapping = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}

st.title("Hate Speech Detection")
st.write("Enter a tweet below and classify it in real-time.")

user_input = st.text_area("Enter a tweet:")


if st.button("Classify"):
    if user_input.strip():
        processed_input = tweets_cleaning(user_input)

        input_vector = vectorizer.transform([processed_input])
        prediction_probs = model.predict_proba(input_vector)[0]
        prediction = np.argmax(prediction_probs)

        predicted_label = label_mapping.get(prediction, "Unknown")
        st.write(f"Prediction: **{predicted_label}**")
    else:
        st.write("Please enter a valid tweet.")
