import streamlit as st
import joblib
import os
from preprocess import clean_text

# Check if model files exist
if not os.path.exists("model/sentiment_model.pkl") or not os.path.exists("model/tfidf_vectorizer.pkl"):
    st.error("Model files not found. Please run train.py first.")
    st.stop()

# Load model
model = joblib.load("model/sentiment_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Flipkart Sentiment Analyzer")

st.title("🛒 Flipkart Product Review Sentiment Analysis")

review = st.text_area("Enter a product review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        try:
            cleaned = clean_text(review)
            if not cleaned:
                st.warning("Review could not be processed. Please enter valid text.")
            else:
                vectorized = tfidf.transform([cleaned])
                prediction = model.predict(vectorized)[0]

                if prediction == 1:
                    st.success("✅ Positive Review")
                else:
                    st.error("❌ Negative Review")
        except Exception as e:
            st.error(f"Error analyzing review: {str(e)}")
