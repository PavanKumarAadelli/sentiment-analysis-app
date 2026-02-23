import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# --- LOAD MODEL FILES ---
try:
    model = pickle.load(open('sentiment_model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found! Please run train_model.py first.")
    st.stop()

# --- PREPROCESSING FUNCTION ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# --- STREAMLIT UI DESIGN ---
st.set_page_config(page_title="Flipkart Sentiment Analyzer", page_icon="🏸")

st.title("🏸 YONEX MAVIS 350 Review Sentiment")
st.write("Predict whether a review is Positive or Negative.")

st.markdown("---")

user_input = st.text_area("Enter the Review Text here:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # 1. Preprocess
        cleaned_review = preprocess_text(user_input)
        
        # 2. Vectorize
        vectorized_review = tfidf.transform([cleaned_review]).toarray()
        
        # 3. Predict
        prediction = model.predict(vectorized_review)[0]
        
        # 4. Display
        if prediction == 1:
            st.success("✅ Result: POSITIVE Review")
        else:
            st.error("❌ Result: NEGATIVE Review")