"""
Training pipeline with cross-validation and class imbalance handling
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from nltk import word_tokenize
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAINING_CONFIG, NN_CONFIG, WORD2VEC_CONFIG, SAVED_MODELS_DIR
from src.embeddings import CustomWord2Vec
from src.models import HybridFakeNewsModel


class FakeNewsTrainer:
    """
    Complete training pipeline for fake news detection
    """
    
    def __init__(self, data_path=None, data_df=None):
        """
        Initialize trainer
        
        Args:
            data_path: Path to training data CSV
            data_df: DataFrame with training data (alternative to data_path)
        """
        self.data_path = data_path
        self.data_df = data_df
        
        # Load data
        if data_df is not None:
            self.df = data_df
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or data_df must be provided")
        
        # Models
        self.word2vec_model = None
        self.hybrid_model = None
        
        # Training history
        self.cv_scores = []
        self.best_model = None
        
    def preprocess_data(self):
        """Preprocess the dataset"""
        print("Preprocessing data...")
        
        # Check required columns
        # Expected columns: 'text', 'headline' (optional), 'label'
        if 'text' not in self.df.columns:
            raise ValueError("Dataset must have 'text' column")
        
        if 'label' not in self.df.columns:
            raise ValueError("Dataset must have 'label' column")
        
        # Drop rows with missing values
        self.df = self.df.dropna(subset=['text', 'label'])
        
        # Convert labels to binary (0 = real, 1 = fake)
        if self.df['label'].dtype == 'object':
            label_map = {
                'fake': 1, 'real': 0,
                'Fake': 1, 'Real': 0,
                'FAKE': 1, 'REAL': 0,
                '1': 1, '0': 0,
                1: 1, 0: 0
            }
            self.df['label'] = self.df['label'].map(label_map)
        
        # Ensure headlines exist
        if 'headline' not in self.df.columns:
            self.df['headline'] = ''
        
        # Clean text
        self.df['text'] = self.df['text'].astype(str)
        self.df['headline'] = self.df['headline'].astype(str)
        
        print(f"Dataset size: {len(self.df)}")
        print(f"Class distribution:\n{self.df['label'].value_counts()}")
        
        return self.df
    
    def train_word2vec(self, save_path=None):
        """
        Train Word2Vec model on the dataset
        
        Args:
            save_path: Path to save the trained model
        """
        print("\n" + "="*50)
        print("Training Word2Vec Embeddings")
        print("="*50)
        
        # Tokenize all texts
        print("Tokenizing texts...")
        sentences = []
        for text in self.df['text']:
            tokens = word_tokenize(str(text).lower())
            sentences.append(tokens)
        
        # Initialize and train Word2Vec
        self.word2vec_model = CustomWord2Vec(
            vector_size=WORD2VEC_CONFIG['vector_size'],
            window=WORD2VEC_CONFIG['window'],
            min_count=WORD2VEC_CONFIG['min_count'],
            epochs=WORD2VEC_CONFIG['epochs'],
            learning_rate=WORD2VEC_CONFIG['alpha']
        )
        
        self.word2vec_model.train(sentences, verbose=True)
        
        # Save model
        if save_path:
            self.word2vec_model.save(save_path)
        
        print("Word2Vec training complete!")
        
        return self.word2vec_model
    
    def load_word2vec(self, model_path):
        """Load pretrained Word2Vec model"""
        self.word2vec_model = CustomWord2Vec()
        self.word2vec_model.load(model_path)
        return self.word2vec_model
    
    def balance_dataset(self, X, y, strategy='SMOTE'):
        """
        Balance the dataset using various strategies
        
        Args:
            X: Feature matrix
            y: Labels
            strategy: 'SMOTE', 'undersample', 'oversample', or 'SMOTEENN'
        
        Returns:
            Balanced X and y
        """
        print(f"\nBalancing dataset using {strategy}...")
        print(f"Original class distribution: {np.bincount(y)}")
        
        if strategy == 'SMOTE':
            smote = SMOTE(random_state=TRAINING_CONFIG['random_state'])
            X_balanced, y_balanced = smote.fit_resample(X, y)
        elif strategy == 'undersample':
            rus = RandomUnderSampler(random_state=TRAINING_CONFIG['random_state'])
            X_balanced, y_balanced = rus.fit_resample(X, y)
        elif strategy == 'SMOTEENN':
            smoteenn = SMOTEENN(random_state=TRAINING_CONFIG['random_state'])
            X_balanced, y_balanced = smoteenn.fit_resample(X, y)
        else:
            # No balancing
            X_balanced, y_balanced = X, y
        
        print(f"Balanced class distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def cross_validate(self, n_folds=5):
        """
        Perform cross-validation
        
        Args:
            n_folds: Number of CV folds
        
        Returns:
            Cross-validation scores
        """
        print("\n" + "="*50)
        print(f"Performing {n_folds}-Fold Cross-Validation")
        print("="*50)
        
        # Prepare data
        texts = self.df['text'].values
        headlines = self.df['headline'].values
        labels = self.df['label'].values
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_folds, 
                             shuffle=True, 
                             random_state=TRAINING_CONFIG['random_state'])
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")
            
            # Split data
            X_train_texts = texts[train_idx]
            X_val_texts = texts[val_idx]
            y_train = labels[train_idx]
            y_val = labels[val_idx]
            
            headlines_train = headlines[train_idx]
            headlines_val = headlines[val_idx]
            
            # Create model
            model = HybridFakeNewsModel(
                word2vec_model=self.word2vec_model,
                nn_config=NN_CONFIG
            )
            
            # Prepare features
            print("Preparing features...")
            X_train_features = model.prepare_features(
                list(X_train_texts), 
                list(headlines_train), 
                fit=True
            )
            X_val_features = model.prepare_features(
                list(X_val_texts), 
                list(headlines_val), 
                fit=False
            )
            
            # Balance training data if configured
            if TRAINING_CONFIG['balance_classes']:
                X_train_features, y_train = self.balance_dataset(
                    X_train_features, 
                    y_train, 
                    TRAINING_CONFIG['sampling_strategy']
                )
            
            # Train model
            print("Training model...")
            model.fit(
                X_train_features, 
                y_train,
                X_val=X_val_features,
                y_val=y_val,
                epochs=NN_CONFIG['epochs'],
                batch_size=NN_CONFIG['batch_size'],
                verbose=False
            )
            
            # Evaluate
            y_pred = model.predict(X_val_features)
            y_proba = model.predict_proba(X_val_features).ravel()
            
            # Compute metrics
            acc = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            
            fold_scores.append({
                'fold': fold + 1,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            })
            
            print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Summary
        print("\n" + "="*50)
        print("Cross-Validation Summary")
        print("="*50)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            scores = [s[metric] for s in fold_scores]
            print(f"{metric.capitalize():12s}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        self.cv_scores = fold_scores
        return fold_scores
    
    def train(self, save_model=True, perform_cv=False):
        """
        Complete training pipeline
        
        Args:
            save_model: Whether to save the trained model
            perform_cv: Whether to perform cross-validation
        
        Returns:
            Trained model
        """
        print("\n" + "="*70)
        print(" "*20 + "FAKE NEWS DETECTION TRAINING")
        print("="*70)
        
        # Preprocess data
        self.preprocess_data()
        
        # Train Word2Vec
        word2vec_path = os.path.join(SAVED_MODELS_DIR, 'word2vec_model.pkl')
        if not os.path.exists(word2vec_path):
            self.train_word2vec(save_path=word2vec_path)
        else:
            print("\nLoading existing Word2Vec model...")
            self.load_word2vec(word2vec_path)
        
        # Perform cross-validation if requested
        if perform_cv:
            self.cross_validate(n_folds=TRAINING_CONFIG['cv_folds'])
        
        # Train final model on all data
        print("\n" + "="*50)
        print("Training Final Model")
        print("="*50)
        
        # Split data
        texts = self.df['text'].values
        headlines = self.df['headline'].values
        labels = self.df['label'].values
        
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, labels,
            test_size=TRAINING_CONFIG['test_size'],
            random_state=TRAINING_CONFIG['random_state'],
            stratify=labels
        )
        
        train_idx = list(range(len(X_train_texts)))
        test_idx = list(range(len(X_train_texts), len(X_train_texts) + len(X_test_texts)))
        
        headlines_train = headlines[:len(X_train_texts)]
        headlines_test = headlines[len(X_train_texts):len(X_train_texts) + len(X_test_texts)]
        
        # Actually split headlines properly
        all_headlines = list(headlines)
        train_indices = np.random.choice(len(texts), size=len(X_train_texts), replace=False)
        test_indices = np.array([i for i in range(len(texts)) if i not in train_indices])
        
        headlines_train = [all_headlines[i] for i in train_indices]
        headlines_test = [all_headlines[i] for i in test_indices]
        
        # Create validation set
        X_train_texts, X_val_texts, y_train, y_val = train_test_split(
            X_train_texts, y_train,
            test_size=TRAINING_CONFIG['validation_size'],
            random_state=TRAINING_CONFIG['random_state'],
            stratify=y_train
        )
        
        headlines_train, headlines_val = train_test_split(
            headlines_train,
            test_size=TRAINING_CONFIG['validation_size'],
            random_state=TRAINING_CONFIG['random_state']
        )
        
        # Create and train model
        self.hybrid_model = HybridFakeNewsModel(
            word2vec_model=self.word2vec_model,
            nn_config=NN_CONFIG
        )
        
        # Prepare features
        print("Preparing features...")
        X_train_features = self.hybrid_model.prepare_features(
            list(X_train_texts), 
            list(headlines_train), 
            fit=True
        )
        X_val_features = self.hybrid_model.prepare_features(
            list(X_val_texts), 
            list(headlines_val), 
            fit=False
        )
        X_test_features = self.hybrid_model.prepare_features(
            list(X_test_texts), 
            list(headlines_test), 
            fit=False
        )
        
        # Balance training data
        if TRAINING_CONFIG['balance_classes']:
            X_train_features, y_train = self.balance_dataset(
                X_train_features, 
                y_train, 
                TRAINING_CONFIG['sampling_strategy']
            )
        
        # Train model
        print("\nTraining hybrid model...")
        self.hybrid_model.fit(
            X_train_features, 
            y_train,
            X_val=X_val_features,
            y_val=y_val,
            epochs=NN_CONFIG['epochs'],
            batch_size=NN_CONFIG['batch_size'],
            verbose=True
        )
        
        # Evaluate on test set
        print("\n" + "="*50)
        print("Final Evaluation on Test Set")
        print("="*50)
        
        y_pred = self.hybrid_model.predict(X_test_features)
        y_proba = self.hybrid_model.predict_proba(X_test_features).ravel()
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Real', 'Fake']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        
        # Save model
        if save_model:
            model_path = os.path.join(SAVED_MODELS_DIR, 'hybrid_model.pkl')
            self.hybrid_model.save(model_path)
            print(f"\nModel saved to {model_path}")
        
        self.best_model = self.hybrid_model
        
        return self.hybrid_model
    
    def save_training_history(self, filepath):
        """Save training history and metrics"""
        history = {
            'cv_scores': self.cv_scores,
            'nn_history': self.hybrid_model.neural_network.history if self.hybrid_model else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(history, f)
        
        print(f"Training history saved to {filepath}")
