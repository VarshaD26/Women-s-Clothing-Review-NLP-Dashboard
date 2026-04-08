import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from nlp_utils import (
    load_data,
    get_clean_texts,
    run_lda,
    lda_distance_map,
)

# ------------------------------------------------------------
# PAGE CONFIG + GLOBAL CSS
# ------------------------------------------------------------
st.set_page_config(page_title="Topic Modeling", layout="wide")

with open("styles/lux_ui.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="lux-header">
    <h1>Topic Modeling</h1>
    <p>Extract hidden themes from reviews to understand fit, quality, comfort, stitching, and design concerns.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = load_data()
df["clean_text"] = get_clean_texts(df)


# ------------------------------------------------------------
# USER CONTROLS
# ------------------------------------------------------------
st.write("### LDA Configuration")

colA, colB = st.columns(2)

with colA:
    n_topics = st.slider("Number of Topics", 3, 10, 5)

with colB:
    ngram_range = st.selectbox("N-gram Range", [(1,1), (1,2), (2,2)], index=1)


# ------------------------------------------------------------
# HIGH vs LOW RATING TOPICS
# ------------------------------------------------------------
st.markdown("## Topic Modeling by Rating Group")

high = df[df["Rating"] >= 4]
low = df[df["Rating"] <= 2]

# Run LDA
lda_high, vec_high, topics_high = run_lda(high["clean_text"], n_topics=n_topics, ngram_range=ngram_range)
lda_low, vec_low, topics_low = run_lda(low["clean_text"], n_topics=n_topics, ngram_range=ngram_range)

# ------------------------------------------------------------
# SHOW TOPICS — HIGH RATING
# ------------------------------------------------------------
st.write("### Topics from High-Rating Reviews (Positive Themes)")

for i, t in enumerate(topics_high):
    st.markdown(f"""
    <div class="lux-card">
        <h3>Topic {i+1} — Customer Likes</h3>
        <b>Top Words:</b> {", ".join(t)}
        <br><br>
        <i>Insight:</i> This theme focuses on **comfort**, **premium feel**, **fit accuracy**, or **stylish design**, as commonly expressed in positive reviews.
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# SHOW TOPICS — LOW RATING
# ------------------------------------------------------------
st.write("### Topics from Low-Rating Reviews (Complaint Themes)")

for i, t in enumerate(topics_low):
    st.markdown(f"""
    <div class="lux-card" style="border-left: 6px solid #FF80AB;">
        <h3>Topic {i+1} — Customer Complaints</h3>
        <b>Top Words:</b> {", ".join(t)}
        <br><br>
        <i>Insight:</i> These words highlight issues such as **fit inconsistency**, **thin or itchy fabric**, **stitching defects**, **color mismatch**, or **poor comfort**.
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# ------------------------------------------------------------
# INTERTOPIC DISTANCE MAP
# ------------------------------------------------------------
st.write("## Intertopic Distance Map (Simplified PCA View)")

coords_high = lda_distance_map(lda_high)
coords_low = lda_distance_map(lda_low)

tab1, tab2 = st.tabs(["High Rating Topics", "Low Rating Topics"])

with tab1:
    fig1 = px.scatter(
        coords_high,
        x="x", y="y",
        text="topic",
        color="topic",
        title="High Rating Topic Distance Map",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig1.update_traces(textposition="top center")
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.scatter(
        coords_low,
        x="x", y="y",
        text="topic",
        color="topic",
        title="Low Rating Topic Distance Map",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig2.update_traces(textposition="top center")
    st.plotly_chart(fig2, use_container_width=True)


st.markdown("---")
