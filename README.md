## Comment-Toxicity-Detection
“Developed a machine learning-based NLP application to classify user comments as toxic or non-toxic, deployed using Streamlit with real-time predictions and text preprocessing.”
# Toxic Comment Detection App

## Project Overview

This project is a machine learning-based web application that detects whether a given comment is toxic or non-toxic. It uses Natural Language Processing (NLP) techniques for text preprocessing and a trained ML model for prediction, deployed using Streamlit.

---

##  Features

* User input for comment analysis
*  Text preprocessing (lowercase, stopword removal, stemming)
*  ML-based classification (toxic / non-toxic)
*  Confidence score display
*  View processed text
*  Interactive UI with Streamlit

---

##  Tech Stack

* Python
* Streamlit
* Scikit-learn
* NLTK
* Pickle

---

##  Project Structure

```id="pt5l2m"
toxic-comment-detector/
│
├── app.py              # Main Streamlit app
├── model.pkl           # Trained ML model
├── tfidf.pkl           # Vectorizer
├── README.md
└── .gitignore
```

---

##  Installation & Setup

###  Clone the repository

```id="k0q9vs"
git clone https://github.com/your-username/toxic-comment-detector.git
cd toxic-comment-detector
```

###  Create virtual environment

```id="b6i1gi"
python -m venv env
env\Scripts\activate
```

###  Install dependencies

```id="g5qv2q"
pip install streamlit scikit-learn nltk
```

---

## Run the Application

```id="9m8u6s"
streamlit run app.py
```

---

## Screenshots

Add your screenshots in an `assets` folder and display like this:

```id="mjqx2r"
![App Screenshot](assets/dashboard.png)
```

---

##  How it Works

1. User enters a comment
2. Text is cleaned and preprocessed
3. TF-IDF vectorization is applied
4. ML model predicts toxicity
5. Result + confidence score is displayed

---

##  Future Enhancements

* Multi-class toxicity detection
* Deep Learning models (LSTM, BERT)
* Deployment on cloud
* Real-time API integration

---


##  Acknowledgment

Dataset inspired by Toxic Comment Classification Challenge.
