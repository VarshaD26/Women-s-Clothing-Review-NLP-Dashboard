import streamlit as st
from nlp_utils import load_data

# ------------------------------------------------------------
# PAGE CONFIG + LOAD CSS
# ------------------------------------------------------------
st.set_page_config(page_title="Overview & Objectives", page_icon="📌", layout="wide")

with open("styles/lux_ui.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------
# HEADER (Luxury Gradient Banner)
# ------------------------------------------------------------
st.markdown("""
<div class="lux-header">
    <h1> NLP Dashboard Overview & Strategic Objectives</h1>
    <p>Understand customer sentiment, recurring themes, and product experience across Women's Clothing reviews.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = load_data()


# ------------------------------------------------------------
# INTRO SUMMARY
# ------------------------------------------------------------
st.write("## Dataset Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Reviews", f"{len(df):,}")

with col2:
    st.metric("Average Rating", f"{df['Rating'].mean():.2f}")

with col3:
    st.metric("Unique Products", df["Class Name"].nunique() if "Class Name" in df.columns else "N/A")


st.markdown("---")


# ------------------------------------------------------------
# OBJECTIVES (Your Updated Clothing Domain Version)
# ------------------------------------------------------------
st.write("## Analysis Objectives (Women’s Clothing)")

st.markdown("""
### Identify Key Themes in Customer Reviews  
Use NLP techniques to uncover themes focused on:
- Fit & sizing issues  
- Fabric/material quality  
- Comfort & wearability  
- Stitching/durability  
- Style, design, and color accuracy  
- Delivery & return concerns  

---

### Perform Sentiment Analysis  
Analyze emotional tone across thousands of reviews to reveal:
- Which clothing attributes drive **positive sentiment**  
- What triggers **negative sentiment** (tight fit, poor stitching, thin fabric)  
- Sentiment patterns aligned with **star ratings**  

---

### Extract Actionable Insights  
Transform customer wording into clear insights to guide:
- Sizing standardization  
- Material upgrades  
- Improving stitching/durability  
- Better & more accurate product descriptions  

---

### Provide Data-Driven Recommendations  
Offer evidence-backed recommendations such as:
- Improve **size consistency**  
- Provide better **fabric descriptions**  
- Address recurring issues (tight sleeves, see-through material)  
- Highlight strengths in marketing (comfort, premium feel, flattering fit)  

---

### Improve Product Quality  
Fix problem areas frequently mentioned in **negative reviews**:
- Fabric feels cheap / thin  
- Fit runs small or large  
- Color differs from photos  
- Stitching defects  

---

### Enhance Marketing Strategies  
Leverage insights from **5-star reviews**:
- Comfort & flattering cut  
- Premium-feel materials  
- Stylish modern design  

---

### Guide Product Development & Innovation  
Use themes extracted to:
- Improve product lines  
- Launch new variants (sizes, fits, colors)  
- Develop customer-preferred materials  

---

### Monitor Customer Sentiment Proactively  
Continuously analyze incoming reviews to:
- Detect emerging issues early  
- Identify product batch defects  
- Improve customer satisfaction faster  
""")

st.markdown("---")


# ------------------------------------------------------------
# WHY THIS DASHBOARD MATTERS
# ------------------------------------------------------------
st.write("## Why This Dashboard Is Important")

st.markdown("""
This system helps product teams, designers, and business analysts uncover patterns that  
directly influence **customer satisfaction**, **conversion rates**, and **product returns**.

It converts thousands of unstructured reviews into:

- Clear visual insights  
- Actionable business recommendations  
- Intelligent early-warning indicators  
- Competitive advantage for product strategy  

Use the sidebar or homepage buttons to explore Sentiment, Topics, Trends, and AI Recommendations.
""")
