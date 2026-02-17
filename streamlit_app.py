"""
Fake News Detection System - Streamlit Application
Main entry point with multi-page navigation
"""
import streamlit as st
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fake-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .real-alert {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    
    page = st.sidebar.radio(
        "Select a page:",
        [
            "🏠 Home",
            "📝 Article Analysis",
            "📊 Batch Analysis",
            "🎯 Model Training",
            "📈 Visualization Dashboard",
            "ℹ️ About"
        ]
    )
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📝 Article Analysis":
        from streamlit_pages import article_analysis
        article_analysis.show()
    elif page == "📊 Batch Analysis":
        from streamlit_pages import batch_analysis
        batch_analysis.show()
    elif page == "🎯 Model Training":
        from streamlit_pages import model_training
        model_training.show()
    elif page == "📈 Visualization Dashboard":
        from streamlit_pages import visualization
        visualization.show()
    elif page == "ℹ️ About":
        from streamlit_pages import about
        about.show()
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    
    # Model selection
    model_dir = "models/saved_models"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'hybrid' in f]
        if model_files:
            selected_model = st.sidebar.selectbox(
                "Select Model:",
                model_files,
                index=0
            )
            st.session_state['selected_model'] = selected_model
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Train a model first if you haven't already!")


def show_home_page():
    """Display the home page"""
    
    # Header
    st.markdown('<div class="main-header">🔍 Fake News Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced AI-powered fake news detection with explainability</div>', unsafe_allow_html=True)
    
    # Welcome section
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📝 Article Analysis")
        st.write("Analyze individual articles for credibility and get detailed explanations.")
        if st.button("Start Analyzing →", key="nav_analysis"):
            st.session_state['page'] = 'Article Analysis'
            st.rerun()
    
    with col2:
        st.markdown("### 📊 Batch Processing")
        st.write("Upload CSV files to analyze multiple articles at once.")
        if st.button("Upload Dataset →", key="nav_batch"):
            st.session_state['page'] = 'Batch Analysis'
            st.rerun()
    
    with col3:
        st.markdown("### 🎯 Model Training")
        st.write("Train custom models on your own datasets.")
        if st.button("Train Model →", key="nav_training"):
            st.session_state['page'] = 'Model Training'
            st.rerun()
    
    st.markdown("---")
    
    # Features section
    st.markdown("## ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🧠 Deep Analysis
        - **Linguistic Features**: Lexical, syntactic, semantic analysis
        - **Psychological Indicators**: Sentiment patterns, emotional manipulation
        - **Structural Analysis**: Headline-body consistency, clickbait detection
        - **Coherence Scoring**: Logical flow and contextual consistency
        """)
        
        st.markdown("""
        #### 🎯 Custom Models
        - **No Pretrained LLMs**: Fully local, transparent system
        - **Hybrid Architecture**: TF-IDF + Neural Networks
        - **Custom Word Embeddings**: Domain-specific Word2Vec
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Explainability
        - **Feature Importance**: Understand what drives predictions
        - **Contribution Analysis**: See which aspects matter most
        - **Confidence Scores**: Transparent uncertainty estimates
        - **Visual Reports**: Interactive charts and graphs
        """)
        
        st.markdown("""
        #### 🚀 Easy to Use
        - **Interactive Interface**: User-friendly web application
        - **Batch Processing**: Handle multiple articles efficiently
        - **Export Results**: Download analysis reports as CSV/JSON
        """)
    
    st.markdown("---")
    
    # Quick Start
    st.markdown("## 🚀 Quick Start Guide")
    
    with st.expander("1️⃣ First-time Setup", expanded=False):
        st.markdown("""
        If you haven't trained a model yet:
        
        1. Go to **Model Training** page
        2. Upload your training dataset (CSV with 'text', 'headline', 'label' columns)
        3. Click "Start Training"
        4. Wait for training to complete
        
        Or use the quick training script:
        ```bash
        python train_example.py
        ```
        """)
    
    with st.expander("2️⃣ Analyze Your First Article", expanded=False):
        st.markdown("""
        1. Go to **Article Analysis** page
        2. Enter the article headline and text
        3. Click "Analyze Article"
        4. Review the results and explanations
        """)
    
    with st.expander("3️⃣ Batch Processing", expanded=False):
        st.markdown("""
        1. Prepare a CSV file with columns: 'text', 'headline' (optional), 'label' (optional)
        2. Go to **Batch Analysis** page
        3. Upload your CSV file
        4. Click "Process Batch"
        5. Download results
        """)
    
    st.markdown("---")
    
    # System Status
    st.markdown("## 📊 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_exists = os.path.exists("models/saved_models/hybrid_model.pkl")
        status = "✅ Ready" if model_exists else "⚠️ Not Found"
        st.metric("Model Status", status)
    
    with col2:
        w2v_exists = os.path.exists("models/saved_models/word2vec_model.pkl")
        status = "✅ Ready" if w2v_exists else "⚠️ Not Found"
        st.metric("Word2Vec", status)
    
    with col3:
        data_files = len([f for f in os.listdir("data/raw") if f.endswith('.csv')]) if os.path.exists("data/raw") else 0
        st.metric("Datasets", f"{data_files} files")
    
    with col4:
        model_files = len([f for f in os.listdir("models/saved_models") if f.endswith('.pkl')]) if os.path.exists("models/saved_models") else 0
        st.metric("Saved Models", f"{model_files} files")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>Built with ❤️ using Streamlit | Explainable AI for Fake News Detection</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
