"""
About Page - Information and documentation
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def show():
    """Display the about page"""
    
    st.title("ℹ️ About")
    st.markdown("Information about the Fake News Detection System")
    
    st.markdown("---")
    
    # Overview
    st.markdown("## 🔍 System Overview")
    
    st.markdown("""
    The **Fake News Detection System** is a comprehensive, explainable AI solution for identifying 
    fake news and misinformation. Unlike systems that rely on black-box pretrained language models, 
    this system is built from scratch with full transparency and interpretability.
    
    ### Key Principles:
    - **Transparency**: Every prediction can be explained
    - **Local Processing**: No external APIs or data sharing
    - **Customizable**: Train on your own datasets
    - **Comprehensive**: Multi-faceted analysis approach
    """)
    
    st.markdown("---")
    
    # Features
    st.markdown("## ✨ Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧠 Analysis Capabilities
        
        #### Linguistic Features
        - Lexical analysis (word choice, vocabulary)
        - Syntactic patterns (sentence structure)
        - Semantic meaning extraction
        - Stylistic inconsistencies
        
        #### Psychological Indicators
        - Sentiment polarity and shifts
        - Emotional manipulation detection
        - Exaggeration markers
        - Urgency and pressure tactics
        
        #### Structural Analysis
        - Headline-body consistency
        - Clickbait pattern detection
        - Source citation analysis
        - Formatting anomalies
        
        #### Coherence Scoring
        - Sentence-to-sentence flow
        - Logical consistency
        - Contextual relevance
        - Topic coherence
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Technical Features
        
        #### Machine Learning
        - Custom Word2Vec embeddings
        - Hybrid TF-IDF + Neural Network
        - Ensemble prediction methods
        - Regularization and optimization
        
        #### Explainability
        - Feature importance analysis
        - Contribution explanations
        - Confidence scoring
        - Visual reports
        
        #### User Interface
        - Interactive web application
        - Batch processing support
        - Model training interface
        - Export capabilities
        
        #### Performance
        - Fast inference (<2s per article)
        - Efficient batch processing
        - Low memory footprint
        - CPU-friendly architecture
        """)
    
    st.markdown("---")
    
    # Architecture
    st.markdown("## 🏗️ System Architecture")
    
    with st.expander("📐 View Architecture Diagram"):
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────┐
        │                    INPUT LAYER                          │
        │         (Article Text + Optional Headline)              │
        └─────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │                FEATURE EXTRACTION                        │
        ├─────────────────────────────────────────────────────────┤
        │  • Linguistic Features (lexical, syntactic, semantic)   │
        │  • Psychological Features (sentiment, emotions)         │
        │  • Structural Features (headline consistency)           │
        │  • Coherence Features (logical flow)                    │
        │  • Custom Word2Vec Embeddings                           │
        └─────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │              HYBRID MODEL LAYER                          │
        ├─────────────────────────────────────────────────────────┤
        │  • TF-IDF Vectorization                                 │
        │  • Handcrafted Feature Integration                      │
        │  • Neural Network Processing                            │
        │  • Ensemble Predictions                                 │
        └─────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │               PREDICTION LAYER                           │
        ├─────────────────────────────────────────────────────────┤
        │  • Binary Classification (Fake/Real)                    │
        │  • Confidence Scores                                    │
        │  • Probability Estimates                                │
        └─────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │             EXPLAINABILITY LAYER                         │
        ├─────────────────────────────────────────────────────────┤
        │  • Feature Importance Analysis                          │
        │  • Contribution Breakdown                               │
        │  • Visual Explanations                                  │
        └─────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────────┐
        │                 OUTPUT LAYER                             │
        │    (Prediction + Confidence + Explanation)               │
        └─────────────────────────────────────────────────────────┘
        ```
        """)
    
    st.markdown("---")
    
    # How to Use
    st.markdown("## 📖 How to Use")
    
    with st.expander("1️⃣ Getting Started"):
        st.markdown("""
        ### First-Time Setup
        
        1. **Install Dependencies**
           ```bash
           pip install -r requirements.txt
           python -m spacy download en_core_web_sm
           python setup_nltk.py
           ```
        
        2. **Train Your First Model**
           - Go to the **Model Training** page
           - Upload a dataset or use the quick demo
           - Configure training parameters
           - Start training
        
        3. **Test the System**
           - Go to **Article Analysis** page
           - Try the sample articles
           - Analyze your own articles
        """)
    
    with st.expander("2️⃣ Analyzing Articles"):
        st.markdown("""
        ### Single Article Analysis
        
        1. Navigate to **Article Analysis** page
        2. Enter the article headline (optional but recommended)
        3. Paste the article text
        4. Click "Analyze Article"
        5. Review results and explanations
        6. Export results if needed
        
        ### Interpretation Guide
        
        - **Fake Probability < 30%**: Likely credible
        - **30% - 70%**: Uncertain, manual review recommended
        - **> 70%**: Strong indicators of fake news
        
        - **Confidence > 80%**: High certainty in prediction
        - **Confidence 50-80%**: Moderate certainty
        - **Confidence < 50%**: Low certainty, review features
        """)
    
    with st.expander("3️⃣ Batch Processing"):
        st.markdown("""
        ### Processing Multiple Articles
        
        1. Prepare CSV file with required columns:
           - `text` (required): Article content
           - `headline` (optional): Article headline
           - `label` (optional): True label for evaluation
        
        2. Go to **Batch Analysis** page
        3. Upload your CSV file
        4. Configure processing options
        5. Click "Process Batch"
        6. Review results and download reports
        
        ### CSV Format Example
        ```csv
        headline,text,label
        "Breaking News","Article text here...",0
        "Another Story","More article text...",1
        ```
        """)
    
    with st.expander("4️⃣ Training Custom Models"):
        st.markdown("""
        ### Training Your Own Model
        
        1. Prepare training dataset (CSV format)
        2. Go to **Model Training** page
        3. Upload dataset or select existing one
        4. Configure hyperparameters:
           - Test set size (recommend 20%)
           - Training epochs (10-20 for small datasets)
           - Batch size (16-32 typical)
           - Word2Vec dimensions (100-200 typical)
        5. Start training
        6. Review performance metrics
        7. Use new model for predictions
        
        ### Dataset Requirements
        - Minimum 100 samples (1000+ recommended)
        - Balanced classes (similar fake/real counts)
        - Diverse topics and writing styles
        - Clean, properly labeled data
        """)
    
    st.markdown("---")
    
    # Technical Details
    st.markdown("## 🔧 Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Dependencies
        - **Python**: 3.8+
        - **Streamlit**: Web interface
        - **scikit-learn**: ML algorithms
        - **TensorFlow/Keras**: Neural networks
        - **spaCy**: NLP processing
        - **NLTK**: Text analysis
        - **pandas**: Data handling
        - **plotly**: Visualizations
        """)
    
    with col2:
        st.markdown("""
        ### Performance
        - **Inference Speed**: ~1-2s per article
        - **Batch Processing**: ~50-100 articles/minute
        - **Memory Usage**: ~500MB-1GB
        - **Model Size**: ~50-100MB
        - **Training Time**: ~10-30 min (1000 samples)
        """)
    
    st.markdown("---")
    
    # FAQ
    st.markdown("## ❓ Frequently Asked Questions")
    
    with st.expander("Q: How accurate is the system?"):
        st.markdown("""
        Accuracy depends on:
        - Quality and size of training data
        - Similarity between training and test data
        - Article length and complexity
        
        Typical performance:
        - Well-trained models: 85-95% accuracy
        - Demo models: 70-80% accuracy
        - Edge cases: May require manual review
        """)
    
    with st.expander("Q: What makes this different from other fake news detectors?"):
        st.markdown("""
        **Our Advantages:**
        - Fully explainable predictions
        - No reliance on external APIs
        - Complete privacy (local processing)
        - Customizable for specific domains
        - Transparent feature engineering
        
        **Trade-offs:**
        - Requires training data
        - Not as accurate as large pretrained models on general tasks
        - Domain-specific (may need retraining for different topics)
        """)
    
    with st.expander("Q: Can I use this in production?"):
        st.markdown("""
        **Yes, with considerations:**
        
        ✅ **Good For:**
        - Internal content moderation
        - Educational purposes
        - Research and analysis
        - Domain-specific applications
        
        ⚠️ **Important Notes:**
        - Train on representative data
        - Regularly retrain with new examples
        - Use as one component in decision pipeline
        - Have human review for critical decisions
        - Consider legal/ethical implications
        """)
    
    with st.expander("Q: How do I improve accuracy?"):
        st.markdown("""
        **Strategies:**
        
        1. **Better Data**
           - More training samples (10,000+)
           - Better labeling quality
           - Diverse examples
           - Domain-relevant articles
        
        2. **Hyperparameter Tuning**
           - Adjust epochs and batch size
           - Try different Word2Vec dimensions
           - Experiment with neural network architecture
        
        3. **Feature Engineering**
           - Add domain-specific features
           - Adjust feature weights
           - Combine with external sources
        
        4. **Ensemble Methods**
           - Train multiple models
           - Use voting or averaging
           - Combine with rule-based systems
        """)
    
    st.markdown("---")
    
    # Credits and License
    st.markdown("## 📜 Credits & License")
    
    st.markdown("""
    ### Built With
    - **Streamlit** - Web interface framework
    - **scikit-learn** - Machine learning library
    - **spaCy** - NLP processing
    - **TensorFlow** - Deep learning
    
    ### License
    This project is available for educational and research purposes.
    
    ### Contact
    For questions, issues, or contributions, please contact the development team.
    """)
    
    st.markdown("---")
    
    # Version Info
    st.markdown("## 🔖 Version Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Version**: 1.0.0")
    with col2:
        st.info("**Release Date**: February 2026")
    with col3:
        st.info("**Status**: Stable")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>Built with ❤️ for fighting misinformation</p>
        <p>Fake News Detection System | Streamlit Edition</p>
    </div>
    """, unsafe_allow_html=True)
