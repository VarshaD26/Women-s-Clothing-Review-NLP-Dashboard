import streamlit as st
from nlp_utils import load_data

# ------------------------------------------
# GLOBAL CONFIG
# ------------------------------------------
st.set_page_config(
    page_title="Women's Clothing NLP Dashboard",
    layout="wide"
)

# ------------------------------------------
# LOAD GLOBAL CSS
# ------------------------------------------
with open("styles/lux_ui.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------
# HERO SECTION
# ------------------------------------------
hero_css = """
<style>
.hero {
    padding: 60px 30px;
    border-radius: 18px;
    background: linear-gradient(135deg, #B388FF 0%, #7C4DFF 50%, #FF80AB 100%);
    color: white;
    text-align: center;
    margin-bottom: 40px;
}
.hero h1 {
    font-size: 48px;
    font-weight: 900;
    margin-bottom: 0px;
}
.hero p {
    font-size: 20px;
    margin-top: 10px;
    opacity: 0.95;
}
.card-btn {
    width: 100%;
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.5);
    color: #7C4DFF;
    font-weight: 800;
    font-size: 18px;
    transition: 0.18s ease;
}
.card-btn:hover {
    transform: translateY(-7px);
    box-shadow: 0px 14px 28px rgba(0,0,0,0.20);
    cursor: pointer;
}
</style>
"""
st.markdown(hero_css, unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Women's Clothing Review Intelligence</h1>
    <p>AI-powered insights on fit, comfort, material quality, sentiment trends, and emerging customer complaints.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------
# FEATURE NAVIGATION CARDS
# ------------------------------------------
st.write("### Dashboard Features")

col1, col2, = st.columns(2)

with col1:
    if st.button("Sentiment Analysis", key="sent_btn", use_container_width=True):
        st.switch_page("pages/2_Sentiment_Analysis.py")

with col2:
    if st.button("Topic Modeling", key="topic_btn", use_container_width=True):
        st.switch_page("pages/3_Topic_Modeling.py")

col3, col4, = st.columns(2)

with col3:
    if st.button("Trending Complaints", key="trend_btn", use_container_width=True):
        st.switch_page("pages/2_Sentiment_Analysis.py")  # same page, bottom section

with col4:
    if st.button("WordClouds", key="wc_btn", use_container_width=True):
        st.switch_page("pages/4_Wordclouds.py")

# ------------------------------------------
# LOAD DATA SUMMARY
# ------------------------------------------
df = load_data()

st.write("### Dataset Overview")

colA, colB, colC = st.columns(3)

with colA:
    st.metric("Total Reviews", f"{len(df):,}")

with colB:
    st.metric("Average Rating", f"{df['Rating'].mean():.2f}")

with colC:
    st.metric("Status", "✔ Loaded")


st.success("Use the left sidebar or cards above to explore dashboard sections.")
