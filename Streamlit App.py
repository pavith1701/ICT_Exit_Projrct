import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load from local folder
MODEL_PATH = "./tweet_urgency_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Tweet Urgency Classifier", layout="centered")
st.title("🚨 Tweet Urgency Classifier")
st.markdown("This app predicts if a tweet needs an **urgent** reply (within 1 hour) or is **non-urgent**.")

tweet = st.text_area("✍️ Enter tweet text below")

if tweet:
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = float(probs[0][pred]) * 100

    label = "🔴 Urgent" if pred == 1 else "🟢 Non-Urgent"
    st.subheader(f"Prediction: {label}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    if st.checkbox("Show model details"):
        st.text(model)
