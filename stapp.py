import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn

nltk.download('stopwords')

nltk.download('wordnet')


# PAGE CONFIG

st.set_page_config(

    page_title="Toxicity Detection System",

    page_icon="⚠️",

    layout="wide"
)
# TITLE
st.title("Comment Toxicity Detection")

with open("vocab.pkl", "rb") as f:

    vocab = pickle.load(f)


stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [

        word for word in words

        if word not in stop_words
    ]

    words = [

        lemmatizer.lemmatize(word)

        for word in words
    ]

    return words

def text_to_sequence(tokens):

    return [

        vocab[word]

        for word in tokens

        if word in vocab
    ]

# PADDING FUNCTION
max_len = 100

def pad_sequence(seq, max_len):

    if len(seq) < max_len:

        seq = seq + [0] * (max_len - len(seq))

    else:

        seq = seq[:max_len]

    return seq

class LSTMModel(nn.Module):

    def __init__(self, vocab_size):

        super(LSTMModel, self).__init__()


        self.embedding = nn.Embedding(

            vocab_size,

            128
        )


        self.lstm = nn.LSTM(

            input_size=128,

            hidden_size=64,

            batch_first=True
        )


        self.dropout = nn.Dropout(0.3)


        self.fc1 = nn.Linear(64, 32)

        self.fc2 = nn.Linear(32, 1)


        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        hidden = hidden[-1]

        x = self.dropout(hidden)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        return x



vocab_size = len(vocab) + 1

model = LSTMModel(vocab_size)

model.load_state_dict(

    torch.load(

        "lstm_model.pth",

        map_location=torch.device('cpu')
    )
)

model.eval()


def predict_toxicity(text):

    tokens = clean_text(text)

    sequence = text_to_sequence(tokens)

    padded = pad_sequence(sequence, max_len)


    tensor = torch.tensor(

        [padded],

        dtype=torch.long
    )


    with torch.no_grad():

        output = model(tensor).item()


    if output < 0.3:

        label = "🟢 Safe Comment"

    elif output < 0.6:

        label = "🟡 Suspicious Comment"

    else:

        label = "🔴 Toxic Comment"


    return label, output

# SIDEBAR

menu = st.sidebar.selectbox(

    "Menu",

    [

        "Home",

        "Single Prediction",

        "Bulk Prediction"
    ]
)


# HOME PAGE
if menu == "Home":

    st.header("Project Overview")

    st.write("""

    This project uses:

    - PyTorch
    - Deep Learning
    - LSTM Neural Networks
    - NLP Preprocessing
    - Toxicity Scoring

    to detect toxic online comments.

    """)

# SINGLE PREDICTION

elif menu == "Single Prediction":

    st.header("Single Comment Prediction")


    user_input = st.text_area(

        "Enter Comment"
    )


    if st.button("Predict"):


        label, score = predict_toxicity(user_input)


        st.subheader("Prediction Result")


        st.write(label)


        st.write(

            "Toxicity Score:",

            round(score * 100, 2),

            "%"
        )

# BULK PREDICTION
elif menu == "Bulk Prediction":

    st.header("Bulk CSV Prediction")


    uploaded_file = st.file_uploader(

        "Upload CSV File",

        type=["csv"]
    )


    if uploaded_file is not None:


        data = pd.read_csv(uploaded_file)


        st.write(data.head())


        predictions = []

        scores = []


        for comment in data['comment_text']:


            label, score = predict_toxicity(comment)


            predictions.append(label)

            scores.append(score)


        data['Prediction'] = predictions

        data['Toxicity Score'] = scores


        st.write(data.head())


        csv = data.to_csv(index=False)


        st.download_button(

            label="Download Predictions",

            data=csv,

            file_name="toxicity_predictions.csv",

            mime="text/csv"
        )