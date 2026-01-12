import streamlit as st

# Inject custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success { background-color: #d1fae5; border-left: 4px solid #10b981; }
    .status-error   { background-color: #fee2e2; border-left: 4px solid #ef4444; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Use the custom classes
st.markdown('<h1 class="main-header">Satellite Water Segmentation</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyzing Sentinel-1 & Sentinel-2 images</p>', unsafe_allow_html=True)
st.markdown('<div class="status-box status-success">✅ Model loaded successfully</div>', unsafe_allow_html=True)
st.markdown('<div class="metric-card"><h2>Accuracy</h2><p>97.8%</p></div>', unsafe_allow_html=True)