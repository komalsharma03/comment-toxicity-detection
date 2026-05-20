## Comment-Toxicity-Detection
“Developed a machine learning-based NLP application to classify user comments as toxic or non-toxic, deployed using Streamlit with real-time predictions and text preprocessing.”
# Toxic Comment Detection App

# Deep Learning for Comment Toxicity Detection with Streamlit

##  Project Overview

Online communities and social media platforms generate millions of comments every day. Toxic comments such as hate speech, insults, threats, and abusive language negatively impact user experience and platform reputation.

This project builds a **Deep Learning–based Comment Toxicity Detection System** using **PyTorch, NLP, and Streamlit** that automatically predicts whether a comment contains toxic content.

The application supports **real-time predictions** and helps moderators identify harmful comments efficiently.

---

# Problem Statement

Develop a deep learning model capable of detecting and classifying toxic comments into multiple categories.

The model predicts:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

The system provides toxicity probability scores and an interactive web application for real-time inference.

---

#  Skills Gained

- Deep Learning
- Natural Language Processing (NLP)
- Text Preprocessing
- PyTorch
- Model Training
- Multi-label Classification
- Streamlit Web App Development
- Model Deployment
- Data Visualization

---

# Domain

Online Community Management & Content Moderation

---

#  Tech Stack

### Programming Language
- Python

### Libraries
- PyTorch
- Streamlit
- Pandas
- NumPy
- NLTK
- TensorFlow (Tokenizer only)
- Scikit-learn

### Deep Learning
- Embedding Layer
- Bidirectional LSTM
- BCEWithLogitsLoss

---

#  Dataset

Dataset contains comments and toxicity labels.

Columns:

| Column |
|---------|
| comment_text |
| toxic |
| severe_toxic |
| obscene |
| threat |
| insult |
| identity_hate |

Dataset Size:

- Total Comments: 159,571
- Multi-label Classification

---

# Project Structure

```text
commenttoxicity/

│
├── train.csv
├── notebook.ipynb
├── app.py
├── tokenizer.pkl
├── toxicity_model.pth
├── requirements.txt
├── README.md
```

---

#  Project Workflow

## 1. Data Loading

Load dataset and inspect structure.

---

## 2. Text Preprocessing

- Lowercase conversion
- Remove URLs
- Remove punctuation
- Stopword removal
- Tokenization
- Padding

---

## 3. Model Development

Deep Learning Architecture:

```text
Input
↓
Embedding Layer
↓
BiLSTM
↓
Dropout
↓
Dense Layer
↓
Output Layer
```

---

## 4. Model Training

Loss Function:

```python
BCEWithLogitsLoss()
```

Optimizer:

```python
Adam
```

Epochs:

```python
10
```

---

## 5. Model Evaluation

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report

---

## 6. Deployment

Deploy model using Streamlit.

Features:

 Single Comment Prediction  
 Toxicity Meter  
 Multi-label Detection  
 Interactive Dashboard  

---

# Sample Predictions

### Example 1

Input:

```text
You are amazing.
```

Output:

```text
Safe
Toxicity Score: 0.01%
```

---

### Example 2

Input:

```text
I hate you idiot.
```

Output:

```text
Highly Toxic
Toxicity Score: 99.08%
```

#  Run Application

Start Streamlit:

```bash
streamlit run app.py
```

#  requirements.txt
```text
streamlit
torch
tensorflow
numpy
pandas
nltk
scikit-learn
```
Install:

```bash
pip install -r requirements.txt
```
#  Business Use Cases
### Social Media Platforms
Automatic moderation.
### Community Forums
Reduce abusive content.
### E-Learning Platforms
Create safe learning environments.
### Content Moderation Services
Automate toxic content detection.
### Brand Safety
Prevent harmful interactions.

# Future Improvements
- Attention Mechanism
- Transformer Models (BERT)
- CSV Batch Prediction
- Cloud Deployment
- Docker Support
- Explainable AI Dashboard

# Model Highlights
-Multi-label Classification  
-Deep Learning Architecture  
-Real-Time Prediction  
-Streamlit Deployment  
-Production-Ready Structure  

---
