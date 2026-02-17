# 🔍 Fake News Detection System

<div align="center">

**An Advanced, Explainable AI System for Detecting Fake News and Misinformation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Documentation](#-documentation) • [Authors](#-authors)

</div>

---

## 📋 Overview

A fully local, transparent, and explainable fake news detection system built from scratch without relying on pretrained language models. This project emphasizes **interpretability**, **customizability**, and **privacy** by processing everything locally with complete transparency into the decision-making process.

### 🎯 Key Highlights

- ✅ **100% Local Processing** - No external APIs or data sharing
- ✅ **Fully Explainable** - Understand every prediction with detailed feature analysis
- ✅ **Custom Training** - Train on your own datasets for domain-specific detection
- ✅ **Web Interface** - User-friendly Streamlit application with interactive dashboards
- ✅ **URL Extraction** - Analyze articles directly from URLs
- ✅ **Batch Processing** - Handle multiple articles efficiently
- ✅ **Multi-layered Analysis** - Psychological, structural, linguistic, and coherence features

---

## ✨ Features

### 🧠 **Deep Linguistic Analysis**
- **Lexical Features**: Vocabulary richness, word complexity, readability scores
- **Syntactic Analysis**: Sentence structure, grammar patterns, POS tagging
- **Semantic Understanding**: Word embeddings, contextual meaning
- **Stylistic Markers**: Writing style inconsistencies, formality levels

### 🎭 **Psychological Indicators**
- **Sentiment Analysis**: VADER-based emotion detection
- **Polarity Shifts**: Emotional manipulation through sentiment swings
- **Exaggeration Markers**: Hyperbole, superlatives, and overstatement detection
- **Urgency Tactics**: Pressure words and time-sensitive language
- **Emotional Manipulation**: Fear, anger, and outrage detection

### 🏗️ **Structural Anomaly Detection**
- **Headline-Body Consistency**: Clickbait and mismatch detection
- **Source Citation Analysis**: Credibility indicators
- **Formatting Patterns**: Unusual capitalization, punctuation abuse
- **Content Structure**: Logical flow and organization

### 🔗 **Contextual Coherence**
- **Sentence Similarity**: Flow and topic consistency
- **Logical Coherence**: Argument structure and reasoning
- **Topic Modeling**: Subject matter consistency
- **Information Density**: Content quality metrics

### 🤖 **Hybrid AI Architecture**
- **TF-IDF Vectorization**: Traditional text representation
- **Custom Word2Vec**: Domain-specific word embeddings
- **Neural Networks**: Deep learning for pattern recognition
- **Ensemble Methods**: Combined predictions for accuracy

### 🔍 **Explainability & Transparency**
- **Feature Importance**: See what drives each prediction
- **Contribution Analysis**: Understand feature impact
- **Visual Reports**: Interactive charts and graphs
- **Confidence Scores**: Transparent uncertainty estimates

### 🌐 **Web Application (Streamlit)**
- **Interactive Interface**: User-friendly web application
- **URL Extraction**: Analyze articles directly from links
- **Batch Processing**: Upload CSV files for bulk analysis
- **Model Training**: Train custom models through the UI
- **Visualization Dashboard**: Performance metrics and analytics
- **Export Capabilities**: Download results in CSV/JSON

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB+ RAM recommended
- Windows/Linux/macOS

### Quick Install

```bash
# Clone or download the repository
cd bct_project

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Setup NLTK data
python setup_nltk.py
```

### Using the Installation Script (Windows)

```batch
# Double-click or run:
install.bat
```

---

## ⚡ Quick Start

### 1️⃣ **Launch the Web Application**

```bash
# Run with Python 3.13 (recommended)
C:\Python313\python.exe -m streamlit run streamlit_app.py

# Or use the batch file (Windows)
run_streamlit.bat
```

The app will open at `http://localhost:8501`

### 2️⃣ **Analyze Your First Article**

**Option A: Manual Input**
1. Go to **📝 Article Analysis** page
2. Enter headline and article text
3. Click **Analyze Article**

**Option B: From URL** ⭐ *NEW*
1. Select **🔗 Extract from URL** option
2. Paste any news article URL
3. Click **Extract & Analyze**

### 3️⃣ **Train Your First Model** (Optional)

```bash
# Quick training with sample data
python train_example.py

# Or use the UI: Go to 🎯 Model Training page
```

---

## 📖 Usage

### Command Line Interface

#### **Analyze a Single Article**

```python
from src.detector import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()

# Analyze article
result = detector.detect(
    text="Your article text here...",
    headline="Article headline",
    explain=True,
    verbose=True
)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Credibility Score: {result['credibility_score']}/100")
```

#### **Train a Custom Model**

```python
from src.training import FakeNewsTrainer

# Initialize trainer
trainer = FakeNewsTrainer()

# Train on your dataset
trainer.train(
    texts=article_texts,
    labels=labels,
    headlines=headlines,
    test_size=0.2,
    epochs=10,
    batch_size=32
)

# Save model
trainer.save_model("models/saved_models/my_model.pkl")
```

#### **Batch Processing**

```python
import pandas as pd
from src.detector import FakeNewsDetector

# Load articles
df = pd.read_csv("articles.csv")

# Initialize detector
detector = FakeNewsDetector()

# Process batch
results = []
for _, row in df.iterrows():
    result = detector.detect(row['text'], row['headline'])
    results.append(result)

# Save results
pd.DataFrame(results).to_csv("analysis_results.csv")
```

### Web Application

#### **Article Analysis Page**
- Analyze individual articles with detailed explanations
- Choose between manual input or URL extraction
- View real-time results with confidence scores
- Export analysis reports

#### **Batch Analysis Page**
- Upload CSV files with multiple articles
- Process up to thousands of articles
- View accuracy metrics and confusion matrix
- Download comprehensive reports

#### **Model Training Page**
- Upload custom datasets
- Configure hyperparameters
- Monitor training progress
- Evaluate model performance

#### **Visualization Dashboard**
- View model performance metrics
- Analyze feature importance
- Track processing history
- Export visualizations

---

## 📊 Output Format

### Analysis Results

```json
{
    "prediction": "FAKE",
    "confidence": 0.92,
    "fake_probability": 0.89,
    "credibility_score": 23.5,
    "psychological_score": 78.3,
    "structural_score": 34.2,
    "coherence_score": 41.7,
    "linguistic_score": 52.1,
    "explanation": {
        "top_features": [
            {"feature": "exaggeration_ratio", "importance": 0.23},
            {"feature": "sentiment_volatility", "importance": 0.19}
        ],
        "analysis": "High exaggeration markers detected..."
    },
    "key_indicators": {
        "warning_signs": [
            "Excessive use of capital letters",
            "Multiple exclamation marks",
            "Emotional manipulation detected"
        ],
        "credibility_markers": []
    }
}
```

### Performance Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Fake news detection precision
- **Recall**: Fake news detection recall
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

---

## 🏗️ Project Structure

```
bct_project/
├── streamlit_app.py              # Main Streamlit application
├── streamlit_pages/              # Streamlit page modules
│   ├── article_analysis.py      # Single article analysis
│   ├── batch_analysis.py        # Batch processing
│   ├── model_training.py        # Model training interface
│   ├── visualization.py         # Analytics dashboard
│   └── about.py                 # Documentation
├── src/                          # Core detection system
│   ├── detector.py              # Main detector class
│   ├── web_extractor.py         # URL content extraction
│   ├── features/                # Feature extraction modules
│   │   ├── linguistic_features.py
│   │   ├── psychological_features.py
│   │   ├── structural_features.py
│   │   └── coherence_features.py
│   ├── embeddings/              # Word embedding models
│   │   └── custom_word2vec.py
│   ├── models/                  # ML models
│   │   ├── hybrid_model.py      # Hybrid detection model
│   │   └── neural_network.py    # Neural network component
│   ├── training/                # Training utilities
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── explainability/          # Explanation generation
│       └── explainer.py
├── data/                         # Datasets
│   ├── raw/                     # Raw CSV files
│   └── processed/               # Processed data
├── models/                       # Trained models
│   └── saved_models/            # Model files (.pkl)
├── results/                      # Analysis results
├── .streamlit/                   # Streamlit configuration
│   └── config.toml
├── config.py                     # Global configuration
├── requirements.txt              # Python dependencies
├── setup_nltk.py                # NLTK data setup
├── train_example.py             # Training script
├── demo.py                      # Demo script
├── test_article.py              # Testing script
├── run_streamlit.bat            # Quick launch script (Windows)
└── install.bat                  # Installation script (Windows)
```

---

## 📚 Documentation

### Additional Resources

- **[STREAMLIT_README.md](STREAMLIT_README.md)** - Streamlit app user guide
- **[STREAMLIT_IMPLEMENTATION.md](STREAMLIT_IMPLEMENTATION.md)** - Technical implementation details
- **[TECHNICAL_DOCS.md](TECHNICAL_DOCS.md)** - System architecture and algorithms
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick command reference
- **[FIX_SUMMARY.md](FIX_SUMMARY.md)** - Bug fixes and updates

### Key Components

#### Feature Extractors
- `LinguisticFeatureExtractor` - Lexical, syntactic, semantic analysis
- `PsychologicalFeatureExtractor` - Sentiment and emotional manipulation
- `StructuralFeatureExtractor` - Document structure and formatting
- `CoherenceFeatureExtractor` - Logical flow and consistency

#### Models
- `HybridFakeNewsModel` - Main detection model combining TF-IDF and neural networks
- `CustomWord2Vec` - Domain-specific word embeddings
- `NeuralNetwork` - Deep learning component

#### Training
- `FakeNewsTrainer` - Model training pipeline
- `Evaluator` - Performance evaluation and metrics

---

## 🎓 How It Works

### Detection Pipeline

```
┌─────────────────────────┐
│   Input Article         │
│  (Text + Headline)      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Preprocessing         │
│  • Tokenization         │
│  • Cleaning             │
│  • Normalization        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Feature Extraction     │
│  • Linguistic (40+)     │
│  • Psychological (20+)  │
│  • Structural (15+)     │
│  • Coherence (10+)      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Hybrid Model          │
│  • TF-IDF Features      │
│  • Handcrafted Features │
│  • Word Embeddings      │
│  • Neural Network       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Prediction            │
│  • Classification       │
│  • Confidence Score     │
│  • Detailed Scores      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Explainability        │
│  • Feature Importance   │
│  • Contribution Analysis│
│  • Warning Indicators   │
└─────────────────────────┘
```

---

## 🔧 Configuration

### Model Parameters

Edit `config.py` to adjust:
- Model paths and directories
- Confidence thresholds
- Feature weights
- Neural network architecture
- Training hyperparameters

### Streamlit Settings

Edit `.streamlit/config.toml` for:
- Server port and host
- Upload size limits
- Theme customization
- Performance settings

---

## 🐛 Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'nltk'**
```bash
pip install nltk
python setup_nltk.py
```

**Streamlit uses wrong Python version**
```bash
# Use specific Python version
C:\Python313\python.exe -m streamlit run streamlit_app.py
```

**Model not found error**
```bash
# Train a model first
python train_example.py
```

**Port already in use**
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502
```

---

## 📈 Performance

### Benchmark Results

| Metric | Score |
|--------|-------|
| Accuracy | 85-95% |
| Precision | 87-93% |
| Recall | 84-92% |
| F1 Score | 86-92% |

*Results vary based on dataset quality and domain*

### Processing Speed

- **Single Article**: ~1-2 seconds
- **Batch (100 articles)**: ~2-3 minutes
- **Training (1000 samples)**: ~10-15 minutes

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- [ ] Additional feature extractors
- [ ] Support for more languages
- [ ] Improved URL extraction for diverse websites
- [ ] Pre-trained models for common domains
- [ ] Mobile-friendly interface
- [ ] API endpoint support

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

This project was developed by:

- **Somyadip Ghosh**
- **Shamonnoy Halder**
- **Rajarshi Bhowmik**
- **Rajat Mitra**
- **Purba Saha**

---

## 🙏 Acknowledgments

- **NLTK** - Natural Language Toolkit
- **spaCy** - Industrial-strength NLP
- **scikit-learn** - Machine learning library
- **Streamlit** - Web application framework
- **VADER** - Sentiment analysis tool
- **BeautifulSoup** - Web scraping library

---

## 📞 Contact

For questions, issues, or collaborations:
- Open an issue in the repository
- Contact the development team
- Check the documentation in the About page

---

## ⭐ Features Roadmap

### Upcoming Features
- [ ] Multi-language support
- [ ] Browser extension
- [ ] Real-time monitoring
- [ ] Advanced visualization
- [ ] Integration with fact-checking APIs
- [ ] Social media analysis
- [ ] Mobile application

---

<div align="center">

**Built with ❤️ for fighting misinformation**

*Fake News Detection System | 2026*

[⬆ Back to Top](#-fake-news-detection-system)

</div>
