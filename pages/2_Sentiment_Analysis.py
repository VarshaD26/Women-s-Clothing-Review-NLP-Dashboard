import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from nlp_utils import (
    load_data,
    add_vader_sentiment,
    get_clean_texts,
    trending_complaints,
)

# ------------------------------------------------------------
# PAGE CONFIG + CSS
# ------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

with open("styles/lux_ui.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="lux-header">
    <h1>Sentiment Analysis</h1>
    <p>
        Analyze emotional tone, rating alignment,
        trending complaints, and sentiment flow
        across clothing reviews.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = load_data()
df_vader = add_vader_sentiment(df)

# ------------------------------------------------------------
# OVERVIEW METRICS
# ------------------------------------------------------------
st.write("## Dashboard Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Reviews", f"{len(df_vader):,}")

with col2:
    positive_pct = (df_vader["vader_label"] == "Positive").mean() * 100
    st.metric("Positive %", f"{positive_pct:.1f}%")

with col3:
    negative_pct = (df_vader["vader_label"] == "Negative").mean() * 100
    st.metric("Negative %", f"{negative_pct:.1f}%")

# ------------------------------------------------------------
# SENTIMENT DISTRIBUTION
# ------------------------------------------------------------
st.write("## Sentiment Distribution (VADER)")

sent_counts = df_vader["vader_label"].value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]

fig1 = px.bar(
    sent_counts,
    x="Sentiment",
    y="Count",
    color="Sentiment",
    color_discrete_map={
        "Positive": "#7C4DFF",
        "Negative": "#FF80AB",
        "Neutral": "#B388FF"
    },
    title="Sentiment Class Distribution"
)

st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------
# SENTIMENT VS RATING
# ------------------------------------------------------------
st.write("## Sentiment vs Rating")

grouped = (
    df_vader.groupby(["Rating", "vader_label"])
    .size()
    .reset_index(name="Count")
)

fig2 = px.bar(
    grouped,
    x="Rating",
    y="Count",
    color="vader_label",
    barmode="group",
    color_discrete_map={
        "Positive": "#7C4DFF",
        "Negative": "#FF80AB",
        "Neutral": "#B388FF"
    },
    title="Sentiment Across Ratings"
)

st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------
# SENTIMENT TIMELINE
# ------------------------------------------------------------
st.write("## Sentiment Timeline")

df_vader = df_vader.copy()
df_vader["index"] = df_vader.index

fig3 = px.line(
    df_vader,
    x="index",
    y="vader_compound",
    animation_frame="Rating",
    markers=True,
    color="Rating"
)

fig3.update_layout(template="plotly_white")

st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------
# REVIEW EXPLORER
# ------------------------------------------------------------
st.write("## View Review Relationships")

fig4 = px.scatter(
    df_vader,
    x="Rating",
    y="vader_compound",
    color="vader_label",
    hover_data=["Review Text"],
    color_discrete_map={
        "Positive": "#7C4DFF",
        "Negative": "#FF80AB",
        "Neutral": "#B388FF"
    },
    title="Hover over points to inspect reviews"
)

st.plotly_chart(fig4, use_container_width=True)

st.info(
    "Use hover, zoom, and selection tools above "
    "to explore sentiment relationships."
)

# ------------------------------------------------------------
# TRENDING COMPLAINTS
# ------------------------------------------------------------
st.write("## Weekly Trending Complaints")

bi, tri = trending_complaints(df)

colA, colB = st.columns(2)

with colA:
    st.write("### Top Bigrams")

    fig5 = px.bar(
        bi.head(15),
        x="freq",
        y="ngram",
        title="Most Frequent Negative Bigrams",
        color="freq",
        color_continuous_scale="Purples"
    )

    st.plotly_chart(fig5, use_container_width=True)

with colB:
    st.write("### Top Trigrams")

    fig6 = px.bar(
        tri.head(12),
        x="freq",
        y="ngram",
        title="Most Frequent Negative Trigrams",
        color="freq",
        color_continuous_scale="Pinkyl"
    )

    st.plotly_chart(fig6, use_container_width=True)

# ------------------------------------------------------------
# COMPLAINT HEATMAP (NO SEABORN)
# ------------------------------------------------------------
st.write("## Complaint Heatmap")

top_neg = bi.head(12)

fig_hm, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    top_neg["ngram"],
    top_neg["freq"]
)

ax.set_title("Top Complaint Phrases")
ax.set_xlabel("Frequency")
ax.set_ylabel("Complaint Phrase")

st.pyplot(fig_hm)