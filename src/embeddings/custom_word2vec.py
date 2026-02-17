"""
Custom Word2Vec implementation from scratch
Trains word embeddings on the dataset without using pretrained models
"""
import numpy as np
import pickle
from collections import defaultdict, Counter
from nltk import word_tokenize
import warnings
warnings.filterwarnings('ignore')


class CustomWord2Vec:
    """
    Custom Word2Vec implementation using Skip-gram with negative sampling
    """
    
    def __init__(self, vector_size=200, window=5, min_count=2, 
                 epochs=20, learning_rate=0.025, negative_samples=5):
        """
        Initialize Word2Vec model
        
        Args:
            vector_size: Dimension of word embeddings
            window: Context window size
            min_count: Minimum word frequency
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            negative_samples: Number of negative samples
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self.vocab_size = 0
        
        # Embeddings
        self.W_in = None  # Input embeddings
        self.W_out = None  # Output embeddings
        
        # For negative sampling
        self.sampling_table = None
        
    def build_vocab(self, sentences):
        """
        Build vocabulary from sentences
        
        Args:
            sentences: List of tokenized sentences
        """
        print("Building vocabulary...")
        
        # Count word frequencies
        for sentence in sentences:
            self.word_freq.update(sentence)
        
        # Filter by min_count
        vocab = [word for word, count in self.word_freq.items() 
                if count >= self.min_count]
        
        # Create word-to-index mapping
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Create negative sampling table
        self._create_sampling_table()
    
    def _init_embeddings(self):
        """Initialize embedding matrices"""
        # Xavier initialization
        limit = np.sqrt(6. / (self.vocab_size + self.vector_size))
        self.W_in = np.random.uniform(-limit, limit, 
                                     (self.vocab_size, self.vector_size))
        self.W_out = np.random.uniform(-limit, limit, 
                                      (self.vocab_size, self.vector_size))
    
    def _create_sampling_table(self):
        """Create probability table for negative sampling"""
        # Use frequency to power of 0.75 (as in original word2vec)
        freq_array = np.array([self.word_freq[self.idx2word[i]] 
                               for i in range(self.vocab_size)])
        freq_array = np.power(freq_array, 0.75)
        self.sampling_table = freq_array / freq_array.sum()
    
    def _get_negative_samples(self, target_idx, num_samples):
        """Get negative samples (words not in context)"""
        negative_samples = []
        while len(negative_samples) < num_samples:
            sample = np.random.choice(self.vocab_size, p=self.sampling_table)
            if sample != target_idx:
                negative_samples.append(sample)
        return negative_samples
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1. / (1. + np.exp(-np.clip(x, -20, 20)))
    
    def _train_pair(self, center_idx, context_idx, lr):
        """
        Train on a single (center, context) pair using skip-gram
        
        Args:
            center_idx: Index of center word
            context_idx: Index of context word
            lr: Learning rate
        """
        # Get embeddings
        center_vec = self.W_in[center_idx]  # (vector_size,)
        context_vec = self.W_out[context_idx]  # (vector_size,)
        
        # Positive sample (actual context word)
        score = np.dot(center_vec, context_vec)
        pred = self._sigmoid(score)
        
        # Gradient for positive sample
        grad = (pred - 1) * lr
        
        # Update embeddings
        self.W_out[context_idx] -= grad * center_vec
        center_grad = grad * context_vec
        
        # Negative samples
        negative_indices = self._get_negative_samples(context_idx, self.negative_samples)
        
        for neg_idx in negative_indices:
            neg_vec = self.W_out[neg_idx]
            score = np.dot(center_vec, neg_vec)
            pred = self._sigmoid(score)
            
            # Gradient for negative sample
            grad = pred * lr
            
            # Update embeddings
            self.W_out[neg_idx] -= grad * center_vec
            center_grad += grad * neg_vec
        
        # Update center word embedding
        self.W_in[center_idx] -= center_grad
    
    def train(self, sentences, verbose=True):
        """
        Train Word2Vec model
        
        Args:
            sentences: List of tokenized sentences
            verbose: Print training progress
        """
        if self.vocab_size == 0:
            self.build_vocab(sentences)
        
        print(f"\nTraining Word2Vec for {self.epochs} epochs...")
        
        total_pairs = 0
        for sentence in sentences:
            for i, word in enumerate(sentence):
                if word in self.word2idx:
                    # Get context window
                    start = max(0, i - self.window)
                    end = min(len(sentence), i + self.window + 1)
                    
                    for j in range(start, end):
                        if j != i and sentence[j] in self.word2idx:
                            total_pairs += 1
        
        print(f"Total training pairs: {total_pairs}")
        
        for epoch in range(self.epochs):
            # Decrease learning rate
            lr = self.learning_rate * (1 - epoch / self.epochs)
            lr = max(lr, self.learning_rate * 0.0001)
            
            pairs_trained = 0
            
            for sentence in sentences:
                for i, word in enumerate(sentence):
                    if word not in self.word2idx:
                        continue
                    
                    center_idx = self.word2idx[word]
                    
                    # Get context window
                    start = max(0, i - self.window)
                    end = min(len(sentence), i + self.window + 1)
                    
                    for j in range(start, end):
                        if j != i and sentence[j] in self.word2idx:
                            context_idx = self.word2idx[sentence[j]]
                            self._train_pair(center_idx, context_idx, lr)
                            pairs_trained += 1
            
            if verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - LR: {lr:.6f} - Pairs: {pairs_trained}")
        
        print("Training complete!")
    
    def get_vector(self, word):
        """Get embedding vector for a word"""
        if word in self.word2idx:
            return self.W_in[self.word2idx[word]]
        else:
            # Return zero vector for unknown words
            return np.zeros(self.vector_size)
    
    def get_sentence_vector(self, sentence):
        """
        Get average embedding vector for a sentence
        
        Args:
            sentence: List of words or string
        
        Returns:
            Average embedding vector
        """
        if isinstance(sentence, str):
            sentence = word_tokenize(sentence.lower())
        
        vectors = []
        for word in sentence:
            if word in self.word2idx:
                vectors.append(self.W_in[self.word2idx[word]])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)
    
    def most_similar(self, word, topn=10):
        """
        Find most similar words to the given word
        
        Args:
            word: Query word
            topn: Number of similar words to return
        
        Returns:
            List of (word, similarity) tuples
        """
        if word not in self.word2idx:
            return []
        
        word_vec = self.get_vector(word)
        
        # Compute cosine similarity with all words
        similarities = []
        for idx in range(self.vocab_size):
            if idx != self.word2idx[word]:
                other_vec = self.W_in[idx]
                # Cosine similarity
                sim = np.dot(word_vec, other_vec) / (
                    np.linalg.norm(word_vec) * np.linalg.norm(other_vec) + 1e-10
                )
                similarities.append((self.idx2word[idx], sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]
    
    def save(self, filepath):
        """Save model to file"""
        model_data = {
            'vector_size': self.vector_size,
            'window': self.window,
            'min_count': self.min_count,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': dict(self.word_freq),
            'vocab_size': self.vocab_size,
            'W_in': self.W_in,
            'W_out': self.W_out
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vector_size = model_data['vector_size']
        self.window = model_data['window']
        self.min_count = model_data['min_count']
        self.word2idx = model_data['word2idx']
        self.idx2word = model_data['idx2word']
        self.word_freq = Counter(model_data['word_freq'])
        self.vocab_size = model_data['vocab_size']
        self.W_in = model_data['W_in']
        self.W_out = model_data['W_out']
        
        # Recreate sampling table
        self._create_sampling_table()
        
        print(f"Model loaded from {filepath}")
    
    def __contains__(self, word):
        """Check if word is in vocabulary"""
        return word in self.word2idx


class EmbeddingFeatureExtractor:
    """
    Extract embedding-based features from text using trained Word2Vec
    """
    
    def __init__(self, word2vec_model):
        """
        Initialize with a trained Word2Vec model
        
        Args:
            word2vec_model: Trained CustomWord2Vec instance
        """
        self.model = word2vec_model
    
    def extract_features(self, text):
        """
        Extract embedding-based features from text
        
        Returns:
            Dictionary of embedding features
        """
        if not text or not isinstance(text, str):
            return self._get_empty_features()
        
        tokens = word_tokenize(text.lower())
        
        # Get sentence vector
        sentence_vec = self.model.get_sentence_vector(tokens)
        
        # Statistics on word embeddings
        word_vectors = []
        oov_count = 0
        
        for token in tokens:
            if token in self.model:
                word_vectors.append(self.model.get_vector(token))
            else:
                oov_count += 1
        
        features = {}
        
        if word_vectors:
            word_vectors = np.array(word_vectors)
            
            # Mean embedding
            features['embedding_mean'] = np.mean(word_vectors, axis=0)
            
            # Embedding statistics
            features['embedding_std'] = np.std(word_vectors)
            features['embedding_variance'] = np.var(word_vectors)
            
            # Vector magnitude statistics
            magnitudes = np.linalg.norm(word_vectors, axis=1)
            features['embedding_magnitude_mean'] = np.mean(magnitudes)
            features['embedding_magnitude_std'] = np.std(magnitudes)
        else:
            features['embedding_mean'] = np.zeros(self.model.vector_size)
            features['embedding_std'] = 0
            features['embedding_variance'] = 0
            features['embedding_magnitude_mean'] = 0
            features['embedding_magnitude_std'] = 0
        
        # Out-of-vocabulary ratio
        features['oov_ratio'] = oov_count / len(tokens) if tokens else 0
        
        return features
    
    def _get_empty_features(self):
        """Return empty feature dictionary"""
        return {
            'embedding_mean': np.zeros(self.model.vector_size),
            'embedding_std': 0,
            'embedding_variance': 0,
            'embedding_magnitude_mean': 0,
            'embedding_magnitude_std': 0,
            'oov_ratio': 0
        }
