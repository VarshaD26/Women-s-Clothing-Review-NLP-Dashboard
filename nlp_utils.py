import pandas as pd
import numpy as np
import re
import nltk
import streamlit as st
import io

# -------------------------------
# DOWNLOAD NLTK DATA
# -------------------------------
def download_nltk():
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('sentiment/vader_lexicon', 'vader_lexicon'),
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/wordnet', 'wordnet')
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

download_nltk()

# -------------------------------
# IMPORTS
# -------------------------------
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------
# OPTIONAL HF SUPPORT
# -------------------------------
HF_AVAILABLE = False

def add_hf_sentiment(df):
    return df

# -------------------------------
# LOAD DATA
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

    df["vader_compound"] = df["Review Text"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    def label(x):
        if x >= 0.05:
            return "Positive"
        elif x <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["vader_label"] = df["vader_compound"].apply(label)
    return df

# -------------------------------
# WORDCLOUD
# -------------------------------
def generate_wordcloud(texts):
    vec = CountVectorizer(stop_words="english")
    bag = vec.fit_transform(texts)

    freqs = dict(zip(vec.get_feature_names_out(), bag.sum(axis=0).A1))

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    )

    return wc.generate_from_frequencies(freqs)

# -------------------------------
# TRENDING COMPLAINTS
# -------------------------------
def freq_ngrams(texts, ngram_range=(2,2)):
    vec = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english"
    )

    bag = vec.fit_transform(texts)

    words = vec.get_feature_names_out()
    freqs = bag.sum(axis=0).A1

    df_ng = pd.DataFrame({
        "ngram": words,
        "freq": freqs
    })

    return df_ng.sort_values("freq", ascending=False)

def trending_complaints(df):
    df = add_vader_sentiment(df)

    neg = df[df["vader_label"] == "Negative"]

    clean_neg = get_clean_texts(neg).tolist()

    bi = freq_ngrams(clean_neg, (2,2)).head(20)
    tri = freq_ngrams(clean_neg, (3,3)).head(15)

    return bi, tri

# -------------------------------
# TOPIC MODELING
# -------------------------------
def run_lda(texts, n_topics=5):
    vec = CountVectorizer(
        stop_words="english",
        max_features=2000
    )

    X = vec.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )

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

    df["label"] = df["vader_label"].map({
        "Positive": 1,
        "Negative": 0,
        "Neutral": 1
    })

    X_train, X_test, y_train, y_test = train_test_split(
        df["Review Text"],
        df["label"],
        test_size=0.2
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

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph(
            "NLP Dashboard Report",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            "Generated successfully.",
            styles["Normal"]
        )
    )

    doc.build(elements)

    return buffer.getvalue()