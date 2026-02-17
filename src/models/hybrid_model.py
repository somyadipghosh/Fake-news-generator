"""
Hybrid model combining TF-IDF, handcrafted features, embeddings, and neural network
"""
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk import word_tokenize

from ..features import (LinguisticFeatureExtractor, 
                        PsychologicalFeatureExtractor,
                        StructuralFeatureExtractor,
                        CoherenceFeatureExtractor)
from ..embeddings import CustomWord2Vec, EmbeddingFeatureExtractor
from .neural_network import NeuralNetwork


class HybridFakeNewsModel:
    """
    Hybrid model combining multiple feature types and a custom neural network
    """
    
    def __init__(self, word2vec_model=None, nn_config=None):
        """
        Initialize hybrid model
        
        Args:
            word2vec_model: Trained CustomWord2Vec model (optional)
            nn_config: Configuration dictionary for neural network
        """
        # Feature extractors
        self.linguistic_extractor = LinguisticFeatureExtractor()
        self.psychological_extractor = PsychologicalFeatureExtractor()
        self.structural_extractor = StructuralFeatureExtractor()
        self.coherence_extractor = CoherenceFeatureExtractor()
        
        # TF-IDF vectorizer
        self.tfidf_vectorizer = None
        
        # Word2Vec model and embedding extractor
        self.word2vec_model = word2vec_model
        self.embedding_extractor = None
        if word2vec_model:
            self.embedding_extractor = EmbeddingFeatureExtractor(word2vec_model)
        
        # Scalers for normalization
        self.handcrafted_scaler = StandardScaler()
        self.tfidf_scaler = StandardScaler()
        self.embedding_scaler = StandardScaler()
        
        # Neural network
        self.nn_config = nn_config or {
            'hidden_layers': [256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'activation': 'relu'
        }
        self.neural_network = None
        
        # Feature dimensions
        self.handcrafted_dim = 0
        self.tfidf_dim = 0
        self.embedding_dim = 0
        self.total_dim = 0
        
        # Feature names
        self.handcrafted_feature_names = []
        
        # Fitted flag
        self.is_fitted = False
    
    def extract_handcrafted_features(self, texts, headlines=None):
        """
        Extract all handcrafted features
        
        Args:
            texts: List of article texts
            headlines: List of headlines (optional)
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        all_features = []
        
        for i, text in enumerate(texts):
            headline = headlines[i] if headlines and i < len(headlines) else None
            
            # Extract all feature types
            ling_features = self.linguistic_extractor.extract_all_features(text)
            psych_features = self.psychological_extractor.extract_all_features(text)
            struct_features = self.structural_extractor.extract_all_features(text, headline)
            coher_features = self.coherence_extractor.extract_all_features(text)
            
            # Combine all features
            combined = {**ling_features, **psych_features, 
                       **struct_features, **coher_features}
            
            all_features.append(combined)
        
        # Convert to matrix
        if not self.handcrafted_feature_names:
            self.handcrafted_feature_names = list(all_features[0].keys())
        
        # Use .get() with default value 0.0 to handle missing features
        feature_matrix = np.array([[f.get(name, 0.0) for name in self.handcrafted_feature_names] 
                                   for f in all_features])
        
        return feature_matrix
    
    def extract_tfidf_features(self, texts, fit=False):
        """
        Extract TF-IDF features
        
        Args:
            texts: List of texts
            fit: Whether to fit the vectorizer
        
        Returns:
            TF-IDF feature matrix
        """
        if fit or self.tfidf_vectorizer is None:
            from config import FEATURE_CONFIG
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=FEATURE_CONFIG['max_features_tfidf'],
                ngram_range=FEATURE_CONFIG['ngram_range'],
                use_idf=FEATURE_CONFIG['use_idf'],
                sublinear_tf=FEATURE_CONFIG['sublinear_tf'],
                min_df=FEATURE_CONFIG['min_df'],
                max_df=FEATURE_CONFIG['max_df'],
                stop_words='english'
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts).toarray()
        
        return tfidf_matrix
    
    def extract_embedding_features(self, texts):
        """
        Extract embedding-based features
        
        Args:
            texts: List of texts
        
        Returns:
            Embedding feature matrix
        """
        if not self.embedding_extractor:
            return np.array([])
        
        embedding_features = []
        
        for text in texts:
            # Get sentence embedding (mean of word embeddings)
            tokens = word_tokenize(text.lower())
            sentence_vec = self.word2vec_model.get_sentence_vector(tokens)
            
            # Additional embedding statistics
            emb_features = self.embedding_extractor.extract_features(text)
            
            # Combine sentence vector with statistics
            feature_vec = np.concatenate([
                sentence_vec,
                [emb_features['embedding_std'],
                 emb_features['embedding_variance'],
                 emb_features['embedding_magnitude_mean'],
                 emb_features['embedding_magnitude_std'],
                 emb_features['oov_ratio']]
            ])
            
            embedding_features.append(feature_vec)
        
        return np.array(embedding_features)
    
    def prepare_features(self, texts, headlines=None, fit=False):
        """
        Prepare all features for the model
        
        Args:
            texts: List of article texts
            headlines: List of headlines (optional)
            fit: Whether to fit preprocessors
        
        Returns:
            Combined feature matrix
        """
        # Extract handcrafted features
        handcrafted_features = self.extract_handcrafted_features(texts, headlines)
        
        # Extract TF-IDF features
        tfidf_features = self.extract_tfidf_features(texts, fit=fit)
        
        # Extract embedding features
        embedding_features = self.extract_embedding_features(texts)
        
        # Normalize features
        if fit:
            handcrafted_features = self.handcrafted_scaler.fit_transform(handcrafted_features)
            tfidf_features = self.tfidf_scaler.fit_transform(tfidf_features)
            if embedding_features.size > 0:
                embedding_features = self.embedding_scaler.fit_transform(embedding_features)
        else:
            handcrafted_features = self.handcrafted_scaler.transform(handcrafted_features)
            tfidf_features = self.tfidf_scaler.transform(tfidf_features)
            if embedding_features.size > 0:
                embedding_features = self.embedding_scaler.transform(embedding_features)
        
        # Combine all features
        if embedding_features.size > 0:
            combined_features = np.hstack([handcrafted_features, 
                                          tfidf_features, 
                                          embedding_features])
        else:
            combined_features = np.hstack([handcrafted_features, tfidf_features])
        
        # Store dimensions
        if fit:
            self.handcrafted_dim = handcrafted_features.shape[1]
            self.tfidf_dim = tfidf_features.shape[1]
            self.embedding_dim = embedding_features.shape[1] if embedding_features.size > 0 else 0
            self.total_dim = combined_features.shape[1]
        
        return combined_features
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            texts=None, headlines=None, epochs=50, batch_size=32, verbose=True):
        """
        Train the hybrid model
        
        Args:
            X_train: Training texts (list of strings) or prepared features (numpy array)
            y_train: Training labels
            X_val: Validation texts or features (optional)
            y_val: Validation labels (optional)
            texts: Alternative way to pass texts
            headlines: Headlines for texts
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Print training progress
        """
        # Prepare features if texts are provided
        if isinstance(X_train, list):
            X_train_features = self.prepare_features(X_train, headlines, fit=True)
        else:
            X_train_features = X_train
            self.total_dim = X_train_features.shape[1]
        
        # Initialize neural network
        self.neural_network = NeuralNetwork(
            input_size=self.total_dim,
            hidden_layers=self.nn_config['hidden_layers'],
            dropout_rate=self.nn_config['dropout_rate'],
            learning_rate=self.nn_config['learning_rate'],
            activation=self.nn_config['activation']
        )
        
        # Prepare validation features if provided
        if X_val is not None and isinstance(X_val, list):
            X_val_features = self.prepare_features(X_val, fit=False)
        elif X_val is not None:
            X_val_features = X_val
        else:
            X_val_features = None
        
        # Reshape labels
        y_train = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
        if y_val is not None:
            y_val = y_val.reshape(-1, 1) if len(y_val.shape) == 1 else y_val
        
        # Training loop
        n_samples = X_train_features.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train_features[indices]
            y_shuffled = y_train[indices]
            
            # Train on batches
            epoch_losses = []
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                loss = self.neural_network.train_batch(X_batch, y_batch)
                epoch_losses.append(loss)
            
            # Compute epoch metrics
            avg_loss = np.mean(epoch_losses)
            train_acc = self.score(X_train_features, y_train)
            
            self.neural_network.history['loss'].append(avg_loss)
            self.neural_network.history['accuracy'].append(train_acc)
            
            # Validation metrics
            if X_val_features is not None:
                val_acc = self.score(X_val_features, y_val)
                
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - "
                          f"Train Acc: {train_acc:.4f} - "
                          f"Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {avg_loss:.4f} - "
                          f"Train Acc: {train_acc:.4f}")
        
        self.is_fitted = True
    
    def predict_proba(self, texts, headlines=None):
        """
        Predict probabilities
        
        Args:
            texts: List of texts or prepared feature matrix
            headlines: List of headlines (optional)
        
        Returns:
            Predicted probabilities
        """
        if isinstance(texts, list):
            X = self.prepare_features(texts, headlines, fit=False)
        else:
            X = texts
        
        return self.neural_network.predict_proba(X)
    
    def predict(self, texts, headlines=None, threshold=0.5):
        """
        Predict class labels
        
        Args:
            texts: List of texts or prepared feature matrix
            headlines: List of headlines (optional)
            threshold: Classification threshold
        
        Returns:
            Predicted labels
        """
        probas = self.predict_proba(texts, headlines)
        return (probas >= threshold).astype(int).ravel()
    
    def score(self, texts, y_true, headlines=None):
        """
        Compute accuracy score
        
        Args:
            texts: List of texts or prepared feature matrix
            y_true: True labels
            headlines: List of headlines (optional)
        
        Returns:
            Accuracy score
        """
        y_pred = self.predict(texts, headlines)
        y_true = y_true.ravel() if len(y_true.shape) > 1 else y_true
        return np.mean(y_pred == y_true)
    
    def save(self, filepath):
        """Save complete model to file"""
        model_data = {
            'nn_config': self.nn_config,
            'handcrafted_dim': self.handcrafted_dim,
            'tfidf_dim': self.tfidf_dim,
            'embedding_dim': self.embedding_dim,
            'total_dim': self.total_dim,
            'handcrafted_feature_names': self.handcrafted_feature_names,
            'is_fitted': self.is_fitted,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'handcrafted_scaler': self.handcrafted_scaler,
            'tfidf_scaler': self.tfidf_scaler,
            'embedding_scaler': self.embedding_scaler,
        }
        
        # Save neural network separately
        if self.neural_network:
            nn_filepath = filepath.replace('.pkl', '_nn.pkl')
            self.neural_network.save(nn_filepath)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load complete model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.nn_config = model_data['nn_config']
        self.handcrafted_dim = model_data['handcrafted_dim']
        self.tfidf_dim = model_data['tfidf_dim']
        self.embedding_dim = model_data['embedding_dim']
        self.total_dim = model_data['total_dim']
        self.handcrafted_feature_names = model_data['handcrafted_feature_names']
        self.is_fitted = model_data['is_fitted']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.handcrafted_scaler = model_data['handcrafted_scaler']
        self.tfidf_scaler = model_data['tfidf_scaler']
        self.embedding_scaler = model_data['embedding_scaler']
        
        # Load neural network
        nn_filepath = filepath.replace('.pkl', '_nn.pkl')
        self.neural_network = NeuralNetwork(
            input_size=self.total_dim,
            hidden_layers=self.nn_config['hidden_layers'],
            dropout_rate=self.nn_config['dropout_rate'],
            learning_rate=self.nn_config['learning_rate'],
            activation=self.nn_config['activation']
        )
        self.neural_network.load(nn_filepath)
        
        print(f"Model loaded from {filepath}")
