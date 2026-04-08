import streamlit as st
import pandas as pd
import plotly.express as px
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

from nlp_utils import (
    load_data,
    add_vader_sentiment,
    get_clean_texts,
    add_hf_sentiment,
    trending_complaints,
    HF_AVAILABLE
)


# ------------------------------------------------------------
# PAGE CONFIG + CSS
# ------------------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")

with open("styles/lux_ui.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="lux-header">
    <h1>Sentiment Analysis</h1>
    <p>Analyze emotional tone, rating alignment, trending complaints, and sentiment flow across clothing reviews.</p>
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
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Reviews", f"{len(df_vader):,}")

with col2:
    st.metric("Positive %", f"{(df_vader['vader_label']=='Positive').mean()*100:.1f}%")

with col3:
    st.metric("Negative %", f"{(df_vader['vader_label']=='Negative').mean()*100:.1f}%")


# ------------------------------------------------------------
# BASIC SENTIMENT DISTRIBUTION
# ------------------------------------------------------------
st.write("## Sentiment Distribution (VADER)")

sent_counts = df_vader["vader_label"].value_counts().reset_index()
sent_counts.columns = ["Sentiment", "Count"]

fig = px.bar(
    sent_counts,
    x="Sentiment",
    y="Count",
    color="Sentiment",
    color_discrete_map={"Positive":"#7C4DFF","Negative":"#FF80AB","Neutral":"#B388FF"},
    title="Sentiment Class Distribution"
)
st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# SENTIMENT VS RATING
# ------------------------------------------------------------
st.write("## Sentiment vs Rating")

grouped = df_vader.groupby(["Rating", "vader_label"]).size().reset_index(name="Count")

fig2 = px.bar(
    grouped,
    x="Rating",
    y="Count",
    color="vader_label",
    barmode="group",
    color_discrete_map={"Positive":"#7C4DFF","Negative":"#FF80AB","Neutral":"#B388FF"},
    title="Sentiment Across Ratings"
)
st.plotly_chart(fig2, use_container_width=True)


# ------------------------------------------------------------
# ANIMATED SENTIMENT TIMELINE
# ------------------------------------------------------------
st.write("## Sentiment Timeline")

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
# CLICK ANY DATA POINT TO SEE REVIEW
# ------------------------------------------------------------
st.write("## View Review")

fig4 = px.scatter(
    df_vader,
    x="Rating",
    y="vader_compound",
    color="vader_label",
    hover_data=["Review Text"],
    color_discrete_map={"Positive":"#7C4DFF","Negative":"#FF80AB","Neutral":"#B388FF"},
    title="Click any point to see details"
)
st.plotly_chart(fig4, use_container_width=True)

st.info("Use the hover and zoom tools above to explore review sentiment relationships.")

# ------------------------------------------------------------
# WEEKLY TRENDING COMPLAINTS (SAME PAGE)
# ------------------------------------------------------------
st.write("## Weekly Trending Complaints (Auto Detected)")

bi, tri = trending_complaints(df)

colA, colB = st.columns(2)

with colA:
    st.write("### Top Bigrams")
    fig5 = px.bar(
        bi.head(15),
        x="freq", y="ngram",
        title="Most Frequent Negative Bigrams",
        color="freq", color_continuous_scale="Purples"
    )
    st.plotly_chart(fig5, use_container_width=True)

with colB:
    st.write("### Top Trigrams")
    fig6 = px.bar(
        tri.head(12),
        x="freq", y="ngram",
        title="Most Frequent Negative Trigrams",
        color="freq", color_continuous_scale="Pinkyl"
    )
    st.plotly_chart(fig6, use_container_width=True)

# ------------------------------------------------------------
# HEATMAP OF COMPLAINT PHRASES
# ------------------------------------------------------------
st.write("## Complaint Heatmap")

neg_texts = get_clean_texts(df_vader[df_vader["vader_label"]=="Negative"]).tolist()
neg_bi = bi.head(12)

fig_hm, ax = plt.subplots(figsize=(10,5))
sns.heatmap(
    neg_bi.pivot_table(values="freq", index="ngram", aggfunc="sum"),
    cmap="Purples",
    cbar=True
)
st.pyplot(fig_hm)
