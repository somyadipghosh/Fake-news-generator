"""
Custom neural network implementation for text classification
"""
import numpy as np
import pickle


class NeuralNetwork:
    """
    Custom Multilayer Perceptron (MLP) for binary classification
    """
    
    def __init__(self, input_size, hidden_layers=[256, 128, 64], 
                 dropout_rate=0.3, learning_rate=0.001, activation='relu'):
        """
        Initialize neural network
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability during training
            learning_rate: Learning rate for optimization
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation_type = activation
        
        # Network architecture: input -> hidden layers -> output(1)
        self.layers = [input_size] + hidden_layers + [1]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # For momentum - initialize before _init_parameters
        self.velocity_w = []
        self.velocity_b = []
        self.beta = 0.9  # Momentum coefficient
        
        self._init_parameters()
        
        # Training history
        self.history = {'loss': [], 'accuracy': []}
    
    def _init_parameters(self):
        """Initialize weights using Xavier/He initialization"""
        for i in range(len(self.layers) - 1):
            n_in = self.layers[i]
            n_out = self.layers[i + 1]
            
            # He initialization for ReLU
            if self.activation_type == 'relu':
                limit = np.sqrt(2. / n_in)
            else:
                # Xavier initialization for other activations
                limit = np.sqrt(6. / (n_in + n_out))
            
            W = np.random.randn(n_in, n_out) * limit
            b = np.zeros((1, n_out))
            
            self.weights.append(W)
            self.biases.append(b)
            
            # Initialize momentum
            self.velocity_w.append(np.zeros_like(W))
            self.velocity_b.append(np.zeros_like(b))
    
    def _activation(self, z):
        """Apply activation function"""
        if self.activation_type == 'relu':
            return np.maximum(0, z)
        elif self.activation_type == 'tanh':
            return np.tanh(z)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -20, 20)))
        else:
            return z
    
    def _activation_derivative(self, z):
        """Compute derivative of activation function"""
        if self.activation_type == 'relu':
            return (z > 0).astype(float)
        elif self.activation_type == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation_type == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(z, -20, 20)))
            return sig * (1 - sig)
        else:
            return np.ones_like(z)
    
    def _sigmoid(self, z):
        """Sigmoid for output layer"""
        return 1 / (1 + np.exp(-np.clip(z, -20, 20)))
    
    def _dropout(self, A, training=True):
        """Apply dropout"""
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=A.shape)
            return A * mask / (1 - self.dropout_rate), mask
        return A, None
    
    def forward(self, X, training=False):
        """
        Forward propagation
        
        Args:
            X: Input data (n_samples, n_features)
            training: Whether in training mode (for dropout)
        
        Returns:
            Output predictions and cache for backprop
        """
        cache = {'A': [X], 'Z': [], 'masks': []}
        A = X
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self._activation(Z)
            
            # Apply dropout
            if training:
                A, mask = self._dropout(A, training=True)
                cache['masks'].append(mask)
            
            cache['Z'].append(Z)
            cache['A'].append(A)
        
        # Output layer (sigmoid activation)
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self._sigmoid(Z)
        
        cache['Z'].append(Z)
        cache['A'].append(A)
        
        return A, cache
    
    def backward(self, y, cache):
        """
        Backward propagation
        
        Args:
            y: True labels (n_samples, 1)
            cache: Cache from forward pass
        
        Returns:
            Gradients for weights and biases
        """
        m = y.shape[0]
        grads = {'dW': [], 'db': []}
        
        # Output layer gradient
        A_out = cache['A'][-1]
        dZ = A_out - y  # Derivative of binary cross-entropy with sigmoid
        
        # Backward through layers
        for i in range(len(self.weights) - 1, -1, -1):
            A_prev = cache['A'][i]
            
            # Compute gradients
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Store gradients (in reverse order)
            grads['dW'].insert(0, dW)
            grads['db'].insert(0, db)
            
            # Continue backprop if not at input layer
            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                
                # Apply dropout mask
                if len(cache['masks']) > 0 and i - 1 < len(cache['masks']):
                    mask = cache['masks'][i - 1]
                    if mask is not None:
                        dA_prev = dA_prev * mask / (1 - self.dropout_rate)
                
                # Apply activation derivative
                Z_prev = cache['Z'][i - 1]
                dZ = dA_prev * self._activation_derivative(Z_prev)
        
        return grads
    
    def update_parameters(self, grads):
        """Update parameters using momentum"""
        for i in range(len(self.weights)):
            # Momentum update
            self.velocity_w[i] = (self.beta * self.velocity_w[i] + 
                                 (1 - self.beta) * grads['dW'][i])
            self.velocity_b[i] = (self.beta * self.velocity_b[i] + 
                                 (1 - self.beta) * grads['db'][i])
            
            # Update parameters
            self.weights[i] -= self.learning_rate * self.velocity_w[i]
            self.biases[i] -= self.learning_rate * self.velocity_b[i]
    
    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        m = y_true.shape[0]
        epsilon = 1e-7  # To avoid log(0)
        
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return loss
    
    def train_batch(self, X, y):
        """
        Train on a single batch
        
        Args:
            X: Input features (batch_size, n_features)
            y: Labels (batch_size, 1)
        
        Returns:
            Loss for this batch
        """
        # Forward pass
        y_pred, cache = self.forward(X, training=True)
        
        # Compute loss
        loss = self.compute_loss(y, y_pred)
        
        # Backward pass
        grads = self.backward(y, cache)
        
        # Update parameters
        self.update_parameters(grads)
        
        return loss
    
    def predict_proba(self, X):
        """
        Predict probabilities
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Predicted probabilities (n_samples, 1)
        """
        y_pred, _ = self.forward(X, training=False)
        return y_pred
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels
        
        Args:
            X: Input features (n_samples, n_features)
            threshold: Classification threshold
        
        Returns:
            Predicted labels (n_samples,)
        """
        y_pred = self.predict_proba(X)
        return (y_pred >= threshold).astype(int).ravel()
    
    def save(self, filepath):
        """Save model to file"""
        model_data = {
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'activation_type': self.activation_type,
            'weights': self.weights,
            'biases': self.biases,
            'history': self.history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.input_size = model_data['input_size']
        self.hidden_layers = model_data['hidden_layers']
        self.dropout_rate = model_data['dropout_rate']
        self.learning_rate = model_data['learning_rate']
        self.activation_type = model_data['activation_type']
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.history = model_data.get('history', {'loss': [], 'accuracy': []})
        
        # Reinitialize momentum
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]
