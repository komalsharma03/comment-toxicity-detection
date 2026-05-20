

import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(

    page_title="Comment Toxicity Detection",

    layout="centered"
)


st.markdown("""

<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00FFAA;
}

.subtitle {
    text-align: center;
    color: white;
    font-size: 18px;
    margin-bottom: 30px;
}

.stTextArea textarea {
    border-radius: 10px;
    border: 2px solid #00FFAA;
}

.result-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #1E1E1E;
    margin-top: 15px;
}

</style>

""", unsafe_allow_html=True)


st.markdown(

    '<div class="title">Comment Toxicity Detection</div>',

    unsafe_allow_html=True
)

st.markdown(

    '<div class="subtitle">Deep Learning NLP Application using PyTorch + Streamlit</div>',

    unsafe_allow_html=True
)


label_names = [

    'toxic',

    'severe_toxic',

    'obscene',

    'threat',

    'insult',

    'identity_hate'
]


with open(

    "tokenizer.pkl",

    "rb"

) as f:

    tokenizer = pickle.load(f)


def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\\S+', '', text)

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    return text


class ToxicityModel(nn.Module):


    def __init__(self, vocab_size):

        super(ToxicityModel, self).__init__()


        # Embedding Layer
        self.embedding = nn.Embedding(

            vocab_size,

            128
        )


        # BiLSTM Layer
        self.lstm = nn.LSTM(

            input_size=128,

            hidden_size=64,

            batch_first=True,

            bidirectional=True
        )


        # Dropout
        self.dropout = nn.Dropout(0.3)


        # Fully Connected Layers
        self.fc1 = nn.Linear(

            128,

            64
        )


        self.fc2 = nn.Linear(

            64,

            6
        )


        self.relu = nn.ReLU()


    def forward(self, x):


        x = self.embedding(x)


        output, (hidden, cell) = self.lstm(x)


        hidden_forward = hidden[-2]

        hidden_backward = hidden[-1]


        hidden = torch.cat(

            (hidden_forward, hidden_backward),

            dim=1
        )


        x = self.dropout(hidden)


        x = self.fc1(x)

        x = self.relu(x)


        x = self.fc2(x)


        return x


model = ToxicityModel(50000)

model.load_state_dict(

    torch.load(

        "toxicity_model.pth",

        map_location='cpu'
    )
)

model.eval()


comment = st.text_area(

    "Enter a Comment",

    height=150,

    placeholder="Type your comment here..."
)


if st.button(

    "Predict Toxicity",

    key="predict_btn"
):


    # Empty Input Check
    if comment.strip() == "":

        st.warning(" Please enter a comment.")

    else:


        cleaned = clean_text(comment)


        sequence = tokenizer.texts_to_sequences(

            [cleaned]
        )

        padded = pad_sequences(

            sequence,

            maxlen=150,

            padding='post'
        )

        # CONVERT TO TENSOR

        tensor = torch.tensor(

            padded,

            dtype=torch.long
        )


        
        # MODEL PREDICTION
        

        with torch.no_grad():

            outputs = model(tensor)


            probs = torch.sigmoid(

                outputs
            ).numpy()[0]


       

        overall_score = float(np.max(probs))


     

        st.subheader(" Overall Toxicity Meter")


        st.progress(overall_score)


        st.write(

            f"### Toxicity Score: {overall_score:.2%}"
        )


        # TOXICITY LEVEL
        if overall_score < 0.20:

            st.success(" Safe Comment")


        elif overall_score < 0.50:

            st.warning(" Suspicious Comment")

        elif overall_score < 0.80:

            st.warning(" Toxic Comment")


        else:

            st.error(" Highly Toxic Comment")

        st.subheader("Detailed Predictions")


        for i, label in enumerate(label_names):


            score = float(probs[i])


            st.markdown(

                f"""
                <div class="result-box">
                <b>{label.upper()}</b> : {score:.2%}
                </div>
                """,

                unsafe_allow_html=True
            )

# SIDEBAR

st.sidebar.title(" About Project")

st.sidebar.info(

    """
    This project detects toxic comments 

    Categories:
    - Toxic
    - Severe Toxic
    - Obscene
    - Threat
    - Insult
    - Identity Hate
    """
)
st.markdown("---")

