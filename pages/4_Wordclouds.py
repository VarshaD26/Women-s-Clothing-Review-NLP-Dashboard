import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from nlp_utils import (
    load_data,
    get_clean_texts,
    add_vader_sentiment,
    generate_wordcloud,
    freq_ngrams
)

# ------------------------------------------------------------
# PAGE CONFIG + GLOBAL CSS
# ------------------------------------------------------------
st.set_page_config(page_title="WordClouds & Keywords", layout="wide")

with open("styles/lux_ui.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown("""
<div class="lux-header">
    <h1>WordClouds & Keyword Analysis</h1>
    <p>Visualize most frequent words, bigrams, and trigrams across positive and negative clothing reviews.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# LOAD DATA + CLEANSING
# ------------------------------------------------------------
df = load_data()
df = add_vader_sentiment(df)
df["clean_text"] = get_clean_texts(df)


# ------------------------------------------------------------
# USER OPTIONS
# ------------------------------------------------------------
st.write("### WordCloud Configuration")

colA, colB = st.columns(2)

with colA:
    sentiment_choice = st.selectbox(
        "Select Sentiment Group", 
        ["All", "Positive", "Neutral", "Negative"]
    )

with colB:
    ngram_choice = st.radio(
        "N-gram Type",
        ["Unigrams (single words)", "Bigrams (two-word phrases)", "Trigrams (three-word phrases)"],
        index=0
    )


# ------------------------------------------------------------
# FILTER DATA
# ------------------------------------------------------------
df_filtered = df.copy()

if sentiment_choice != "All":
    df_filtered = df_filtered[df_filtered["vader_label"] == sentiment_choice]


# Determine ngram range
if "Unigrams" in ngram_choice:
    ngram_range = (1,1)
elif "Bigrams" in ngram_choice:
    ngram_range = (2,2)
else:
    ngram_range = (3,3)


# ------------------------------------------------------------
# WORDCLOUD GENERATION
# ------------------------------------------------------------
texts = df_filtered["clean_text"].tolist()
wc = generate_wordcloud(texts, ngram_range=ngram_range)

st.write("### WordCloud Visualization")

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")

st.pyplot(fig)


# ------------------------------------------------------------
# FREQUENCY BAR CHART
# ------------------------------------------------------------
st.write("### Most Frequent Words / N-grams")

ng_df = freq_ngrams(texts, ngram_range).head(20)

fig2 = px.bar(
    ng_df,
    x="freq",
    y="ngram",
    color="freq",
    color_continuous_scale="Purples",
    title=f"Top {len(ng_df)} Frequent N-grams"
)
st.plotly_chart(fig2, use_container_width=True)


# ------------------------------------------------------------
# DOMINANT THEMES INSIGHTS (FASHION DOMAIN)
# ------------------------------------------------------------
st.markdown("## Insights from the Keywords")

if sentiment_choice == "Positive":
    st.success("""
### What Customers Love
- Comfortable fit  
- Soft / premium fabric  
- Accurate sizing  
- Stylish colors & patterns  
- High-quality stitching  
- Great value for money  
""")

elif sentiment_choice == "Negative":
    st.error("""
### What Customers Complain About
- Runs small / runs large  
- Fabric feels cheap or thin  
- Itchy or uncomfortable material  
- Loose stitching / defects  
- Color mismatch vs website  
- Short length / tight sleeves  
""")

elif sentiment_choice == "Neutral":
    st.info("""
### Neutral Feedback Patterns
- Fit not as expected  
- Mixed feelings on material quality  
- Some inconsistencies across product batches  
""")

else:
    st.info("""
### Mixed Sentiment Themes (All Reviews)
- Fit and fabric dominate conversation  
- Color accuracy is frequently mentioned  
- Stitching quality varies by product batch  
- Comfort plays a major role in ratings  
""")
