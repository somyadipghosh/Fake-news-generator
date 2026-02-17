"""
Configuration file for the fake news detection system
"""
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODELS_DIR, SAVED_MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Word2Vec parameters
WORD2VEC_CONFIG = {
    'vector_size': 200,
    'window': 5,
    'min_count': 2,
    'workers': 4,
    'sg': 1,  # Skip-gram (1) or CBOW (0)
    'epochs': 20,
    'alpha': 0.025,
    'min_alpha': 0.0001
}

# Neural network parameters
NN_CONFIG = {
    'hidden_layers': [256, 128, 64],
    'dropout_rate': 0.3,
    'activation': 'relu',
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10
}

# Training parameters
TRAINING_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'cv_folds': 5,
    'random_state': 42,
    'balance_classes': True,
    'sampling_strategy': 'SMOTE'  # or 'undersample', 'oversample'
}

# Feature extraction parameters
FEATURE_CONFIG = {
    'max_features_tfidf': 5000,
    'ngram_range': (1, 3),
    'use_idf': True,
    'sublinear_tf': True,
    'min_df': 2,
    'max_df': 0.95
}

# Thresholds
THRESHOLDS = {
    'confidence_threshold': 0.5,
    'manipulation_high': 70,
    'manipulation_medium': 40,
    'structural_integrity_low': 50,
    'volatility_high': 60
}

# Explainability
EXPLAINABILITY_CONFIG = {
    'top_features': 15,
    'generate_plots': True,
    'save_explanations': True
}
