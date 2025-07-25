import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess(text):
    text = text.lower()
    return ' '.join([word for word in text.split() if word not in stop_words])

st.title("Flipkart Review Sentiment Analyzer")
user_input = st.text_area("Enter your review")

if st.button("Predict Sentiment"):
    cleaned = preprocess(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    st.write(f"**Sentiment:** {sentiment}")
