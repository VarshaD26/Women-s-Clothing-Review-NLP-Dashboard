import pandas as pd
import numpy as np
import re
import nltk
import streamlit as st

# -------------------------------
# DOWNLOAD ALL REQUIRED NLTK DATA
# -------------------------------
def download_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import plotly.express as px
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import io

# -------------------------------
# LOAD DATA (FIXED FOR DEPLOYMENT)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
    df["Review Text"] = df["Review Text"].astype(str)
    df = df.dropna(subset=["Review Text"])
    return df.reset_index(drop=True)

# -------------------------------
# TEXT CLEANING
# -------------------------------
stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemm.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def get_clean_texts(df):
    return df["Review Text"].apply(clean_text)

# -------------------------------
# VADER SENTIMENT
# -------------------------------
sia = SentimentIntensityAnalyzer()

def add_vader_sentiment(df):
    df = df.copy()

    df["compound"] = df["Review Text"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    def label(x):
        if x >= 0.05:
            return "Positive"
        elif x <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment"] = df["compound"].apply(label)
    return df

# -------------------------------
# WORDCLOUD
# -------------------------------
def generate_wordcloud(texts):
    vec = CountVectorizer(stop_words="english")
    bag = vec.fit_transform(texts)

    freqs = dict(zip(vec.get_feature_names_out(), bag.sum(axis=0).A1))

    wc = WordCloud(width=1200, height=600, background_color="white")
    return wc.generate_from_frequencies(freqs)

# -------------------------------
# TOPIC MODELING (LDA)
# -------------------------------
def run_lda(texts, n_topics=5):
    vec = CountVectorizer(stop_words="english", max_features=2000)
    X = vec.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    words = vec.get_feature_names_out()
    topics = []

    for topic in lda.components_:
        top_words = [words[i] for i in topic.argsort()[-10:]]
        topics.append(top_words)

    return topics

# -------------------------------
# MODEL TRAINING
# -------------------------------
def train_model(df):
    df = add_vader_sentiment(df)

    df["label"] = df["sentiment"].map({
        "Positive": 1,
        "Negative": 0,
        "Neutral": 1
    })

    X_train, X_test, y_train, y_test = train_test_split(
        df["Review Text"], df["label"], test_size=0.2
    )

    tfidf = TfidfVectorizer(stop_words="english")
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)

# -------------------------------
# PDF REPORT
# -------------------------------
def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("NLP Dashboard Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Generated successfully.", styles["Normal"]))

    doc.build(elements)

    return buffer.getvalue()