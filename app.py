import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


st.set_page_config(page_title="Toxic Comment Detector", layout="centered")


model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('tfidf.pkl','rb'))


nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)


st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
     Toxic Comment Classification Challenge
    </h1>
    <p style='text-align: center;'>
    Detect whether a comment is toxic or not using Machine Learning 🤖
    </p>
    """,
    unsafe_allow_html=True
)


user_input = st.text_area(" Enter your comment here:")


if st.button("Analyze Comment"):
    if user_input.strip() == "":
        st.warning(" Please enter a comment first!")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0][1]

        st.markdown("---")

        if prediction == 1:
            st.error(f" Toxic Comment Detected!\n\nConfidence: {prob:.2f}")
        else:
            st.success(f"Non-Toxic Comment\n\nConfidence: {prob:.2f}")

        
        with st.expander("🔍 See Processed Text"):
            st.write(cleaned)





import os
print(os.getcwd())