"""
Batch Analysis Page - Process multiple articles from CSV files
"""
import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import FakeNewsDetector


def show():
    """Display the batch analysis page"""
    
    st.title("📊 Batch Analysis")
    st.markdown("Upload CSV files to analyze multiple articles at once")
    
    st.markdown("---")
    
    # Check if model is available
    model_path = "models/saved_models/hybrid_model.pkl"
    if not os.path.exists(model_path):
        st.error("⚠️ No trained model found!")
        st.info("Please train a model first using the Model Training page or run `python train_example.py`")
        return
    
    # Instructions
    with st.expander("📖 Instructions", expanded=False):
        st.markdown("""
        ### CSV Format Requirements
        
        Your CSV file should have the following columns:
        - **text** (required): The article content
        - **headline** (optional): The article headline
        - **label** (optional): True label for evaluation (0=real, 1=fake)
        
        ### Example CSV Format:
        ```csv
        headline,text,label
        "Breaking News","This is the article text...",0
        "Another Article","More article text...",1
        ```
        
        ### Features:
        - ✅ Process multiple articles in one batch
        - ✅ Get predictions and confidence scores
        - ✅ Compare against true labels (if provided)
        - ✅ Export results to CSV
        - ✅ View detailed statistics
        """)
    
    st.markdown("---")
    
    # File upload
    st.markdown("### 📁 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with 'text' and optionally 'headline' and 'label' columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} articles.")
            
            # Show preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Validate columns
            if 'text' not in df.columns:
                st.error("❌ CSV file must have a 'text' column!")
                return
            
            # Processing options
            st.markdown("---")
            st.markdown("### ⚙️ Processing Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_articles = st.number_input(
                    "Maximum articles to process",
                    min_value=1,
                    max_value=len(df),
                    value=min(100, len(df)),
                    help="Limit processing for large datasets"
                )
                
                enable_explanation = st.checkbox(
                    "Include detailed explanations",
                    value=False,
                    help="Warning: This will slow down processing significantly"
                )
            
            with col2:
                has_labels = 'label' in df.columns
                if has_labels:
                    st.info("✅ Labels found - will compute accuracy metrics")
                else:
                    st.info("ℹ️ No labels found - will only predict")
                
                batch_size = st.selectbox(
                    "Batch size",
                    [10, 25, 50, 100],
                    index=1,
                    help="Process articles in batches"
                )
            
            st.markdown("---")
            
            # Process button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                process_button = st.button("🚀 Process Batch", type="primary", use_container_width=True)
            
            # Process articles
            if process_button:
                process_batch(df, max_articles, enable_explanation, batch_size, has_labels)
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)
    else:
        st.info("👆 Upload a CSV file to get started")
        
        # Sample data download
        st.markdown("---")
        st.markdown("### 📥 Download Sample Template")
        
        sample_data = pd.DataFrame({
            'headline': [
                'Scientists Discover New Planet',
                'SHOCKING: You Won\'t Believe This!!!',
                'Local School Wins Competition'
            ],
            'text': [
                'Astronomers at the European Southern Observatory have discovered a new exoplanet in the habitable zone of a nearby star system.',
                'AMAZING miracle cure discovered!!! Doctors HATE this one simple trick! Click NOW before it\'s deleted!!!',
                'Students from Lincoln High School won first place in the regional science competition with their innovative project.'
            ],
            'label': [0, 1, 0]
        })
        
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_data.to_csv(index=False),
            file_name="sample_dataset.csv",
            mime="text/csv"
        )


def process_batch(df, max_articles, enable_explanation, batch_size, has_labels):
    """Process batch of articles"""
    
    # Limit articles
    df = df.head(max_articles)
    
    st.markdown("---")
    st.markdown("## 🔄 Processing Articles...")
    
    # Initialize detector
    with st.spinner("Loading model..."):
        detector = FakeNewsDetector()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    start_time = time.time()
    
    # Process in batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        for idx, row in batch.iterrows():
            try:
                # Get text and headline
                text = str(row['text'])
                headline = str(row.get('headline', '')) if 'headline' in row else None
                
                # Analyze
                result = detector.detect(
                    text=text,
                    headline=headline,
                    explain=enable_explanation,
                    verbose=False
                )
                
                # Store result
                result_dict = {
                    'index': idx,
                    'headline': headline,
                    'text_preview': text[:100] + "..." if len(text) > 100 else text,
                    'prediction': result.get('prediction', 'UNKNOWN'),
                    'fake_probability': result.get('fake_probability', 0),
                    'confidence': result.get('confidence', 0),
                    'credibility_score': result.get('credibility_score', 0),
                }
                
                # Add true label if available
                if has_labels:
                    result_dict['true_label'] = 'FAKE' if row['label'] == 1 else 'REAL'
                    result_dict['correct'] = result_dict['prediction'] == result_dict['true_label']
                
                results.append(result_dict)
                
            except Exception as e:
                st.warning(f"⚠️ Error processing article {idx}: {str(e)}")
            
            # Update progress
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Processed {idx + 1} / {len(df)} articles...")
    
    elapsed_time = time.time() - start_time
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Processing complete! Analyzed {len(results)} articles in {elapsed_time:.1f} seconds")
    
    # Display results
    display_batch_results(results, has_labels, elapsed_time)


def display_batch_results(results, has_labels, elapsed_time):
    """Display batch processing results"""
    
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = len(results_df)
        st.metric("Total Articles", total)
    
    with col2:
        fake_count = len(results_df[results_df['prediction'] == 'FAKE'])
        st.metric("Fake News", fake_count, f"{fake_count/total*100:.1f}%")
    
    with col3:
        real_count = len(results_df[results_df['prediction'] == 'REAL'])
        st.metric("Real News", real_count, f"{real_count/total*100:.1f}%")
    
    with col4:
        avg_confidence = results_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Accuracy metrics if labels available
    if has_labels:
        st.markdown("---")
        st.markdown("### 🎯 Accuracy Metrics")
        
        correct = results_df['correct'].sum()
        accuracy = correct / len(results_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.1%}", help="Overall accuracy")
        
        with col2:
            # True Positives (predicted fake, actually fake)
            tp = len(results_df[(results_df['prediction'] == 'FAKE') & (results_df['true_label'] == 'FAKE')])
            st.metric("True Positives", tp, help="Correctly identified fake news")
        
        with col3:
            # False Positives (predicted fake, actually real)
            fp = len(results_df[(results_df['prediction'] == 'FAKE') & (results_df['true_label'] == 'REAL')])
            st.metric("False Positives", fp, help="Real news marked as fake")
        
        with col4:
            # True Negatives (predicted real, actually real)
            tn = len(results_df[(results_df['prediction'] == 'REAL') & (results_df['true_label'] == 'REAL')])
            st.metric("True Negatives", tn, help="Correctly identified real news")
        
        # Confusion matrix
        st.markdown("#### Confusion Matrix")
        confusion = pd.crosstab(
            results_df['true_label'],
            results_df['prediction'],
            rownames=['Actual'],
            colnames=['Predicted'],
            margins=True
        )
        st.dataframe(confusion, use_container_width=True)
    
    # Results table
    st.markdown("---")
    st.markdown("### 📋 Detailed Results")
    
    # Display options
    col1, col2 = st.columns(2)
    
    with col1:
        show_only = st.selectbox(
            "Filter results:",
            ["All", "Fake Only", "Real Only", "High Confidence (>80%)", "Low Confidence (<50%)"]
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ["Index", "Fake Probability (High to Low)", "Confidence (High to Low)", "Credibility Score"]
        )
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if show_only == "Fake Only":
        filtered_df = filtered_df[filtered_df['prediction'] == 'FAKE']
    elif show_only == "Real Only":
        filtered_df = filtered_df[filtered_df['prediction'] == 'REAL']
    elif show_only == "High Confidence (>80%)":
        filtered_df = filtered_df[filtered_df['confidence'] > 0.8]
    elif show_only == "Low Confidence (<50%)":
        filtered_df = filtered_df[filtered_df['confidence'] < 0.5]
    
    # Apply sorting
    if sort_by == "Fake Probability (High to Low)":
        filtered_df = filtered_df.sort_values('fake_probability', ascending=False)
    elif sort_by == "Confidence (High to Low)":
        filtered_df = filtered_df.sort_values('confidence', ascending=False)
    elif sort_by == "Credibility Score":
        filtered_df = filtered_df.sort_values('credibility_score', ascending=False)
    
    # Format for display
    display_df = filtered_df.copy()
    display_df['fake_probability'] = display_df['fake_probability'].apply(lambda x: f"{x:.1%}")
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    display_df['credibility_score'] = display_df['credibility_score'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📈 Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Fake Probability Distribution")
        st.bar_chart(results_df['fake_probability'].value_counts().sort_index())
    
    with col2:
        st.markdown("#### Confidence Distribution")
        st.bar_chart(results_df['confidence'].value_counts().sort_index())
    
    # Export options
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export full results
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv_data,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export fake only
        fake_df = results_df[results_df['prediction'] == 'FAKE']
        st.download_button(
            label="📥 Download Fake Only (CSV)",
            data=fake_df.to_csv(index=False),
            file_name=f"fake_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Export summary
        summary = {
            'total_articles': len(results_df),
            'fake_count': len(results_df[results_df['prediction'] == 'FAKE']),
            'real_count': len(results_df[results_df['prediction'] == 'REAL']),
            'avg_confidence': results_df['confidence'].mean(),
            'processing_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if has_labels:
            summary['accuracy'] = results_df['correct'].sum() / len(results_df)
        
        import json
        st.download_button(
            label="📥 Download Summary (JSON)",
            data=json.dumps(summary, indent=2),
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
