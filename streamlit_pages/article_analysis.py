"""
Article Analysis Page - Single article analysis with detailed explanations
"""
import streamlit as st
import sys
import os
import json
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import FakeNewsDetector
from src.web_extractor import extract_article_from_url


def show():
    """Display the article analysis page"""
    
    st.title("📝 Article Analysis")
    st.markdown("Analyze individual articles for credibility with detailed explanations")
    
    st.markdown("---")
    
    # Check if model is available
    model_path = "models/saved_models/hybrid_model.pkl"
    if not os.path.exists(model_path):
        st.error("⚠️ No trained model found!")
        st.info("Please train a model first using the Model Training page or run `python train_example.py`")
        return
    
    # Input method selection
    st.markdown("### 📥 Choose Input Method")
    input_method = st.radio(
        "How would you like to provide the article?",
        ["📝 Manual Text Input", "🔗 Extract from URL"],
        horizontal=True
    )
    
    st.markdown("---")
    
    headline = ""
    text = ""
    
    if input_method == "🔗 Extract from URL":
        # URL Input Method
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 🔗 Article URL")
            url = st.text_input(
                "Enter the article URL:",
                placeholder="https://example.com/article",
                help="Paste the full URL of the article you want to analyze"
            )
        
        with col2:
            st.markdown("### ⚙️ Options")
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=5,
                max_value=30,
                value=10,
                help="Maximum time to wait for page load"
            )
        
        if url:
            if st.button("🔍 Extract & Analyze Article", type="primary", use_container_width=True):
                with st.spinner("Extracting article from URL..."):
                    result = extract_article_from_url(url, timeout=timeout)
                    
                    if result['success']:
                        st.success(f"✅ Article extracted successfully!")
                        headline = result['title']
                        text = result['text']
                        
                        # Show preview
                        with st.expander("📄 Extracted Content Preview"):
                            st.markdown(f"**Title:** {headline}")
                            st.markdown(f"**Text Length:** {len(text)} characters")
                            st.text_area("Article Text:", text[:500] + "..." if len(text) > 500 else text, height=200)
                        
                        # Store in session state for analysis
                        st.session_state['extracted_headline'] = headline
                        st.session_state['extracted_text'] = text
                        st.session_state['perform_analysis'] = True
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to extract article: {result['error']}")
                        st.info("💡 **Troubleshooting Tips:**\n"
                               "- Make sure the URL is correct\n"
                               "- Some websites block automated access\n"
                               "- Try copying the text manually instead")
    
    else:
        # Manual Text Input Method
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📰 Article Input")
            
            # Headline input
            headline = st.text_input(
                "Headline",
                placeholder="Enter the article headline...",
                help="Optional but recommended for better analysis"
            )
            
            # Text input
            text = st.text_area(
                "Article Text",
                height=300,
                placeholder="Paste the full article text here...",
                help="Enter the main content of the article"
            )
            
        with col2:
            st.markdown("### ⚙️ Analysis Options")
            
            enable_explanation = st.checkbox("Enable Detailed Explanation", value=True)
            enable_verbose = st.checkbox("Verbose Output", value=True)
            
            st.markdown("---")
            
            # Sample articles
            st.markdown("### 📋 Sample Articles")
            
            sample_choice = st.selectbox(
                "Load a sample:",
                ["None", "Real News Example", "Fake News Example"],
            )
            
            if sample_choice == "Real News Example":
                if st.button("Load Real News Sample"):
                    st.session_state['sample_headline'] = "MIT Researchers Develop AI System for Early Disease Detection"
                    st.session_state['sample_text'] = """Scientists at the Massachusetts Institute of Technology have developed a new artificial intelligence system that can detect patterns in medical imaging data with unprecedented accuracy. The research, published in Nature Medicine, shows that the system achieved 95% accuracy in identifying early signs of disease in CT scans. Dr. Sarah Johnson, lead researcher on the project, explained that the system was trained on over 100,000 medical images from multiple hospitals. The team hopes this technology will help radiologists catch diseases earlier, potentially saving thousands of lives. The research was funded by the National Institutes of Health and underwent rigorous peer review before publication."""
                    st.rerun()
            
            elif sample_choice == "Fake News Example":
                if st.button("Load Fake News Sample"):
                    st.session_state['sample_headline'] = "SHOCKING Miracle Cure Discovered! Doctors Hate This Trick!"
                    st.session_state['sample_text'] = """SHOCKING!!! Scientists REVEAL that drinking coffee backwards can INSTANTLY cure ALL diseases!!! You WON'T BELIEVE what happens next!!! Doctors HATE this one simple trick that Big Pharma doesn't want you to know!!! Everyone is talking about this MIRACLE cure that has been hidden from the public for years!!! This AMAZING discovery will change your life FOREVER!!! Click here NOW before it's too late!!! URGENT!!! Share this with EVERYONE you know before they DELETE it!!!"""
                    st.rerun()
    
    # Load samples if they exist in session state
    if 'sample_headline' in st.session_state:
        headline = st.session_state.pop('sample_headline')
    if 'sample_text' in st.session_state:
        text = st.session_state.pop('sample_text')
    
    # Load extracted content if it exists
    if 'extracted_headline' in st.session_state and input_method == "🔗 Extract from URL":
        headline = st.session_state.pop('extracted_headline')
        text = st.session_state.pop('extracted_text')
        enable_explanation = st.checkbox("Enable Detailed Explanation", value=True, key="url_explanation")
        enable_verbose = st.checkbox("Verbose Output", value=True, key="url_verbose")
    
    st.markdown("---")
    
    # Analysis button (only for manual input or if we have extracted content)
    perform_analysis = st.session_state.pop('perform_analysis', False) if input_method == "🔗 Extract from URL" else False
    
    if input_method == "📝 Manual Text Input":
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
    else:
        analyze_button = perform_analysis
    
    # Perform analysis
    if analyze_button or perform_analysis:
        if not text.strip():
            st.warning("⚠️ Please enter article text to analyze")
            return
        
        with st.spinner("Analyzing article... This may take a moment..."):
            try:
                # Initialize detector
                detector = FakeNewsDetector()
                
                # Perform detection
                result = detector.detect(
                    text=text,
                    headline=headline if headline else None,
                    explain=enable_explanation,
                    verbose=enable_verbose
                )
                
                # Display results
                display_results(result, text, headline)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)


def display_results(result, text, headline):
    """Display analysis results"""
    
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Main prediction
    prediction = result.get('prediction', 'UNKNOWN')
    confidence = result.get('confidence', 0)
    fake_probability = result.get('fake_probability', 0)
    
    # Color-coded alert
    if prediction == "FAKE":
        st.markdown(f"""
        <div class="fake-alert">
            <h2 style='color: #d32f2f; margin: 0;'>🚨 FAKE NEWS DETECTED</h2>
            <p style='font-size: 1.2rem; margin: 0.5rem 0;'>Confidence: {confidence:.1%}</p>
            <p style='margin: 0;'>This article shows strong indicators of fake news.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="real-alert">
            <h2 style='color: #388e3c; margin: 0;'>✅ APPEARS CREDIBLE</h2>
            <p style='font-size: 1.2rem; margin: 0.5rem 0;'>Confidence: {confidence:.1%}</p>
            <p style='margin: 0;'>This article appears to be credible based on analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prediction", prediction, help="Final classification")
    
    with col2:
        st.metric("Confidence", f"{confidence:.1%}", help="Model confidence in prediction")
    
    with col3:
        st.metric("Fake Probability", f"{fake_probability:.1%}", help="Raw probability of being fake")
    
    with col4:
        credibility_score = result.get('credibility_score', 0)
        st.metric("Credibility Score", f"{credibility_score:.2f}/100", help="Overall credibility rating")
    
    # Detailed scores
    st.markdown("---")
    st.markdown("### 📈 Detailed Score Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Psychological Indicators")
        psych_score = result.get('psychological_score', 0)
        st.progress(psych_score / 100)
        st.write(f"Score: {psych_score:.1f}/100")
        
        with st.expander("View Details"):
            if 'psychological_features' in result:
                st.json(result['psychological_features'])
        
        st.markdown("#### Structural Analysis")
        struct_score = result.get('structural_score', 0)
        st.progress(struct_score / 100)
        st.write(f"Score: {struct_score:.1f}/100")
        
        with st.expander("View Details"):
            if 'structural_features' in result:
                st.json(result['structural_features'])
    
    with col2:
        st.markdown("#### Coherence Analysis")
        coher_score = result.get('coherence_score', 0)
        st.progress(coher_score / 100)
        st.write(f"Score: {coher_score:.1f}/100")
        
        with st.expander("View Details"):
            if 'coherence_features' in result:
                st.json(result['coherence_features'])
        
        st.markdown("#### Linguistic Features")
        ling_score = result.get('linguistic_score', 0)
        st.progress(ling_score / 100)
        st.write(f"Score: {ling_score:.1f}/100")
        
        with st.expander("View Details"):
            if 'linguistic_features' in result:
                st.json(result['linguistic_features'])
    
    # Explanation
    if 'explanation' in result and result['explanation']:
        st.markdown("---")
        st.markdown("### 🔍 Explanation")
        
        explanation = result['explanation']
        
        if 'top_features' in explanation:
            st.markdown("#### Most Influential Features")
            
            features_df = pd.DataFrame(explanation['top_features'])
            if not features_df.empty:
                st.dataframe(features_df, use_container_width=True)
        
        if 'analysis' in explanation:
            st.markdown("#### Detailed Analysis")
            st.write(explanation['analysis'])
    
    # Key indicators
    if 'key_indicators' in result:
        st.markdown("---")
        st.markdown("### 🎯 Key Indicators")
        
        indicators = result['key_indicators']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'warning_signs' in indicators and indicators['warning_signs']:
                st.markdown("#### ⚠️ Warning Signs")
                for sign in indicators['warning_signs']:
                    st.markdown(f"- {sign}")
        
        with col2:
            if 'credibility_markers' in indicators and indicators['credibility_markers']:
                st.markdown("#### ✅ Credibility Markers")
                for marker in indicators['credibility_markers']:
                    st.markdown(f"- {marker}")
    
    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as JSON
        json_data = json.dumps(result, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export as CSV
        csv_data = pd.DataFrame([{
            'headline': headline,
            'text': text[:100] + "...",
            'prediction': prediction,
            'confidence': confidence,
            'fake_probability': fake_probability,
            'credibility_score': result.get('credibility_score', 0),
            'timestamp': datetime.now().isoformat()
        }])
        st.download_button(
            label="📥 Download CSV",
            data=csv_data.to_csv(index=False),
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Copy to clipboard (display text version)
        if st.button("📋 Copy Summary"):
            summary = f"""
Article Analysis Summary
========================
Headline: {headline}
Prediction: {prediction}
Confidence: {confidence:.1%}
Credibility Score: {result.get('credibility_score', 0):.1f}/100
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            st.code(summary, language=None)
