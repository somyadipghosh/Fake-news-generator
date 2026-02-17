"""
Model Training Page - Train new models on custom datasets
"""
import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import FakeNewsTrainer
from config import RAW_DATA_DIR, SAVED_MODELS_DIR


def show():
    """Display the model training page"""
    
    st.title("🎯 Model Training")
    st.markdown("Train custom fake news detection models on your datasets")
    
    st.markdown("---")
    
    # Instructions
    with st.expander("📖 Training Instructions", expanded=False):
        st.markdown("""
        ### Dataset Requirements
        
        Your CSV file must have the following columns:
        - **text** (required): The article content
        - **label** (required): Binary label (0=real, 1=fake)
        - **headline** (optional): The article headline
        
        ### Training Process:
        1. **Data Preprocessing**: Clean and prepare text data
        2. **Feature Extraction**: Extract linguistic, psychological, structural features
        3. **Word2Vec Training**: Train custom word embeddings
        4. **Model Training**: Train hybrid model (TF-IDF + Neural Network)
        5. **Evaluation**: Compute accuracy, precision, recall, F1 score
        
        ### Recommendations:
        - Minimum 100 samples for basic training
        - 1000+ samples recommended for production
        - Balanced dataset (50-50 fake/real) works best
        - Include diverse writing styles and topics
        """)
    
    st.markdown("---")
    
    # Training mode selection
    st.markdown("### 🎲 Select Training Mode")
    
    training_mode = st.radio(
        "Choose your training approach:",
        ["Upload Custom Dataset", "Use Existing Dataset", "Quick Demo Training"],
        help="Select how you want to train the model"
    )
    
    st.markdown("---")
    
    if training_mode == "Upload Custom Dataset":
        upload_and_train()
    elif training_mode == "Use Existing Dataset":
        use_existing_dataset()
    elif training_mode == "Quick Demo Training":
        quick_demo_training()


def upload_and_train():
    """Handle custom dataset upload and training"""
    
    st.markdown("### 📁 Upload Training Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with 'text', 'label', and optionally 'headline' columns",
        key="upload_train"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} articles.")
            
            # Show preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Validate columns
            if 'text' not in df.columns or 'label' not in df.columns:
                st.error("❌ CSV file must have 'text' and 'label' columns!")
                return
            
            # Dataset statistics
            st.markdown("---")
            st.markdown("#### 📊 Dataset Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(df))
            
            with col2:
                fake_count = len(df[df['label'] == 1])
                st.metric("Fake News", fake_count, f"{fake_count/len(df)*100:.1f}%")
            
            with col3:
                real_count = len(df[df['label'] == 0])
                st.metric("Real News", real_count, f"{real_count/len(df)*100:.1f}%")
            
            with col4:
                balance_ratio = min(fake_count, real_count) / max(fake_count, real_count)
                st.metric("Balance Ratio", f"{balance_ratio:.2f}", help="1.0 is perfectly balanced")
            
            # Training configuration
            st.markdown("---")
            configure_and_train(df)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)
    else:
        st.info("👆 Upload a CSV file to get started")


def use_existing_dataset():
    """Use existing dataset from data/raw directory"""
    
    st.markdown("### 📂 Select Existing Dataset")
    
    # List available datasets
    if not os.path.exists(RAW_DATA_DIR):
        st.warning("⚠️ No data directory found!")
        return
    
    csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    
    if not csv_files:
        st.warning("⚠️ No CSV files found in data/raw directory!")
        st.info("Add your datasets to the data/raw folder or use the upload option.")
        return
    
    selected_file = st.selectbox(
        "Choose a dataset:",
        csv_files,
        help="Select from available datasets in data/raw/"
    )
    
    if selected_file:
        file_path = os.path.join(RAW_DATA_DIR, selected_file)
        
        try:
            df = pd.read_csv(file_path)
            
            st.success(f"✅ Dataset loaded: {selected_file}")
            
            # Show preview
            with st.expander("👀 View Data Preview"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Validate
            if 'text' not in df.columns or 'label' not in df.columns:
                st.error("❌ Dataset must have 'text' and 'label' columns!")
                return
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Fake News", len(df[df['label'] == 1]))
            with col3:
                st.metric("Real News", len(df[df['label'] == 0]))
            
            st.markdown("---")
            configure_and_train(df)
            
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")


def quick_demo_training():
    """Quick demo training with small sample dataset"""
    
    st.markdown("### ⚡ Quick Demo Training")
    st.info("This will train a model on a small sample dataset for demonstration purposes.")
    
    st.warning("⚠️ Demo models are not suitable for production use!")
    
    if st.button("🚀 Start Demo Training", type="primary"):
        # Generate sample data
        with st.spinner("Generating sample dataset..."):
            df = generate_sample_dataset()
        
        st.success(f"✅ Generated {len(df)} sample articles")
        
        with st.expander("👀 View Sample Data"):
            st.dataframe(df, use_container_width=True)
        
        # Train with default settings
        st.markdown("---")
        st.markdown("### 🔄 Training Progress")
        
        config = {
            'test_size': 0.2,
            'random_state': 42,
            'epochs': 5,
            'batch_size': 8,
            'word2vec_dim': 50
        }
        
        train_model(df, config, model_name="demo_model")


def configure_and_train(df):
    """Configure training parameters and start training"""
    
    st.markdown("### ⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Settings")
        
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for testing"
        )
        
        random_state = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            help="For reproducible results"
        )
        
        model_name = st.text_input(
            "Model Name",
            value=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Name for the saved model"
        )
    
    with col2:
        st.markdown("#### Advanced Settings")
        
        epochs = st.number_input(
            "Training Epochs",
            min_value=1,
            max_value=100,
            value=10,
            help="Number of training iterations"
        )
        
        batch_size = st.selectbox(
            "Batch Size",
            [8, 16, 32, 64],
            index=1,
            help="Number of samples per training batch"
        )
        
        word2vec_dim = st.selectbox(
            "Word2Vec Dimensions",
            [50, 100, 200, 300],
            index=1,
            help="Dimensionality of word embeddings"
        )
    
    st.markdown("---")
    
    # Start training
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        train_button = st.button("🚀 Start Training", type="primary", use_container_width=True)
    
    if train_button:
        config = {
            'test_size': test_size,
            'random_state': random_state,
            'epochs': epochs,
            'batch_size': batch_size,
            'word2vec_dim': word2vec_dim
        }
        
        train_model(df, config, model_name)


def train_model(df, config, model_name):
    """Train the model with given configuration"""
    
    st.markdown("---")
    st.markdown("## 🔄 Training in Progress...")
    
    start_time = time.time()
    
    try:
        # Initialize trainer
        with st.spinner("Initializing trainer..."):
            trainer = FakeNewsTrainer()
        
        st.info("✓ Trainer initialized")
        
        # Prepare data
        with st.spinner("Preparing data..."):
            texts = df['text'].tolist()
            headlines = df['headline'].tolist() if 'headline' in df else None
            labels = df['label'].tolist()
        
        st.info("✓ Data prepared")
        
        # Train model
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Training model... This may take several minutes...")
        
        # Note: The actual training happens here
        # We'll simulate progress since the trainer doesn't have built-in progress callbacks
        trainer.train(
            texts=texts,
            labels=labels,
            headlines=headlines,
            test_size=config['test_size'],
            random_state=config['random_state'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            word2vec_dim=config['word2vec_dim']
        )
        
        progress_bar.progress(100)
        status_text.text("Training complete!")
        
        st.success("✅ Model training completed successfully!")
        
        # Save model
        with st.spinner("Saving model..."):
            model_path = os.path.join(SAVED_MODELS_DIR, f"{model_name}.pkl")
            trainer.save_model(model_path)
        
        st.success(f"✅ Model saved to: {model_path}")
        
        # Get evaluation metrics
        st.markdown("---")
        st.markdown("### 📊 Training Results")
        
        metrics = trainer.get_metrics()
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1', 0):.2%}")
            
            # Additional metrics
            with st.expander("📈 Detailed Metrics"):
                st.json(metrics)
        
        elapsed_time = time.time() - start_time
        st.info(f"⏱️ Total training time: {elapsed_time:.1f} seconds")
        
        # Next steps
        st.markdown("---")
        st.markdown("### ✨ Next Steps")
        st.markdown("""
        Your model is now ready to use! You can:
        - Go to **Article Analysis** to test it on individual articles
        - Go to **Batch Analysis** to process multiple articles
        - Go to **Visualization Dashboard** to view performance metrics
        """)
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.exception(e)


def generate_sample_dataset():
    """Generate a small sample dataset for demo"""
    
    data = {
        'headline': [
            'Study Shows Benefits of Exercise',
            'SHOCKING: Miracle Cure Found!!!',
            'New Technology Improves Solar Panels',
            'You WON\'T BELIEVE This TRICK!!!',
            'Local School Wins Competition',
            'DOCTORS HATE This Simple Method!!!',
        ],
        'text': [
            'A comprehensive study published in the Journal of Medicine has found that regular exercise significantly reduces the risk of heart disease. Researchers followed 10,000 participants over five years.',
            'AMAZING discovery that doctors don\'t want you to know!!! This MIRACLE cure will change everything!!! Click NOW before it\'s DELETED!!!',
            'Engineers at Stanford University have developed a new coating for solar panels that increases efficiency by 15%. The breakthrough was published in Nature Energy.',
            'This ONE WEIRD TRICK will solve ALL your problems!!! Big companies are trying to HIDE this from you!!! Act FAST!!!',
            'Lincoln High School students won first place in the state science fair with their innovative water purification project. The team will compete nationally next month.',
            'Doctors are FURIOUS about this SIMPLE method!!! Lose 50 pounds in ONE DAY with NO EFFORT!!! SHARE before they DELETE THIS!!!',
        ],
        'label': [0, 1, 0, 1, 0, 1]
    }
    
    return pd.DataFrame(data)
