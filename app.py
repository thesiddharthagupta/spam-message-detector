import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    "text": [
        "Win money now",
        "Hello how are you",
        "Claim your prize",
        "Let's meet tomorrow",
        "Click this link",
        "Good morning bro"
    ],
    "label": [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# UI
st.title("📩 Spam Message Detector")

user_input = st.text_input("Enter a message:")

if st.button("Check"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        result = model.predict(input_vec)

        if result[0] == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT spam")
    else:
        st.warning("Please enter a message")