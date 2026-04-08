import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.colors import HexColor
import io
import streamlit as st

# Transformers (optional)
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        r"C:/Users/varsh/OneDrive/Desktop/nlp_dashboard/Womens Clothing E-Commerce Reviews.csv"
    )
    df = df.rename(columns={"Review Text": "Review Text", "Rating": "Rating"})
    df["Review Text"] = df["Review Text"].astype(str)
    df = df.dropna(subset=["Review Text"])
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------
stop_words = set(stopwords.words("english"))
lemm = WordNetLemmatizer()

def clean_text(text, lemmatize=True):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    if lemmatize:
        tokens = [lemm.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def get_clean_texts(df, lemmatize=True):
    return df["Review Text"].apply(lambda x: clean_text(str(x), lemmatize))


# ---------------------------------------------------------
# VADER SENTIMENT
# ---------------------------------------------------------
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def add_vader_sentiment(df):
    df = df.copy()
    df["vader_compound"] = df["Review Text"].apply(lambda x: sia.polarity_scores(str(x))["compound"])

    def label(x):
        if x >= 0.05: return "Positive"
        elif x <= -0.05: return "Negative"
        else: return "Neutral"

    df["vader_label"] = df["vader_compound"].apply(label)
    df["ReviewIndex"] = df.index
    return df


# ---------------------------------------------------------
# HUGGINGFACE BERT SENTIMENT (Inference Only)
# ---------------------------------------------------------
@st.cache_resource
def hf_pipeline():
    if not HF_AVAILABLE:
        return None
    return pipeline("sentiment-analysis")


def add_hf_sentiment(df, max_samples=400):
    if not HF_AVAILABLE:
        return df

    df_hf = df.head(max_samples).copy()
    clf = hf_pipeline()

    preds = clf(df_hf["Review Text"].tolist())
    df_hf["hf_label"] = [p["label"] for p in preds]
    df_hf["hf_score"] = [p["score"] for p in preds]

    return df_hf


# ---------------------------------------------------------
# WORDCLOUD
# ---------------------------------------------------------
def generate_wordcloud(texts, ngram_range=(1,1)):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    bag = vec.fit_transform(texts)
    freqs = dict(zip(vec.get_feature_names_out(), bag.sum(axis=0).A1))

    wc = WordCloud(
        width=1400,
        height=700,
        background_color="white",
        colormap="Purples"
    ).generate_from_frequencies(freqs)

    return wc


# ---------------------------------------------------------
# N-GRAM FREQUENCY
# ---------------------------------------------------------
def freq_ngrams(texts, ngram_range=(2,2)):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    bag = vec.fit_transform(texts)
    words = vec.get_feature_names_out()
    freqs = bag.sum(axis=0).A1
    df_ng = pd.DataFrame({"ngram": words, "freq": freqs})
    df_ng = df_ng.sort_values("freq", ascending=False)
    return df_ng


# ---------------------------------------------------------
# LDA TOPIC MODELING
# ---------------------------------------------------------
def run_lda(corpus, n_topics=5, ngram_range=(1,1), max_features=2000):
    vec = CountVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
    )
    X = vec.fit_transform(corpus)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch"
    )
    lda.fit(X)

    terms = vec.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [terms[i] for i in topic.argsort()[-10:]]
        topics.append(top_words)

    return lda, vec, topics


# ---------------------------------------------------------
# LDA INTERTOPIC DISTANCE MAP (PCA)
# ---------------------------------------------------------
from sklearn.decomposition import PCA
import pandas as pd

def lda_distance_map(lda_model):
    comps = lda_model.components_
    pca = PCA(n_components=2)
    coords = pca.fit_transform(comps)
    df = pd.DataFrame({"x": coords[:,0], "y": coords[:,1]})
    df["topic"] = ["Topic "+str(i+1) for i in range(len(df))]
    return df


# ---------------------------------------------------------
# LOGISTIC REGRESSION MODEL (TF-IDF)
# ---------------------------------------------------------
def train_logreg_model(df):
    df = add_vader_sentiment(df)

    df["label"] = df["vader_label"].map({"Positive":1, "Negative":0, "Neutral":1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["Review Text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=300)
    model.fit(X_train_tfidf, y_train)

    y_pred_train = model.predict(X_train_tfidf)
    y_pred_test = model.predict(X_test_tfidf)

    return {
        "train_acc": accuracy_score(y_train, y_pred_train),
        "test_acc": accuracy_score(y_test, y_pred_test),
        "train_report": classification_report(y_train, y_pred_train, output_dict=True),
        "test_report": classification_report(y_test, y_pred_test, output_dict=True),
        "train_conf": confusion_matrix(y_train, y_pred_train),
        "test_conf": confusion_matrix(y_test, y_pred_test)
    }


# ---------------------------------------------------------
# TRENDING COMPLAINTS (NEGATIVE N-GRAMS)
# ---------------------------------------------------------
def trending_complaints(df):
    df = add_vader_sentiment(df)
    neg = df[df["vader_label"]=="Negative"]

    clean_neg = get_clean_texts(neg).tolist()
    bi = freq_ngrams(clean_neg, (2,2)).head(20)
    tri = freq_ngrams(clean_neg, (3,3)).head(15)

    return bi, tri


# ---------------------------------------------------------
# PDF REPORT (UNICODE-SAFE via REPORTLAB)
# ---------------------------------------------------------
def build_pdf_report(overall_stats, sentiment_stats, topic_summary):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    body = styles["Normal"]
    title_style = styles["Title"]
    title_style.textColor = HexColor("#7C4DFF")

    elements = []
    elements.append(Paragraph("Women's Clothing – NLP Insights Report", title_style))
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("<b>Dataset Overview</b>", body))
    for k, v in overall_stats.items():
        elements.append(Paragraph(f"{k}: {v}", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Sentiment Summary</b>", body))
    for k, v in sentiment_stats.items():
        elements.append(Paragraph(f"{k}: {v}", body))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Key Themes Identified</b>", body))
    for t in topic_summary:
        elements.append(Paragraph(f"• {t}", body))
    elements.append(Spacer(1, 16))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
