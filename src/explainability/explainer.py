"""
Explainability module for understanding model predictions
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import EXPLAINABILITY_CONFIG


class ModelExplainer:
    """
    Provide explainability for fake news detection model
    """
    
    def __init__(self, model):
        """
        Initialize explainer
        
        Args:
            model: Trained HybridFakeNewsModel instance
        """
        self.model = model
    
    def compute_feature_importance(self, X, y, method='perturbation', n_samples=100):
        """
        Compute feature importance scores
        
        Args:
            X: Feature matrix or list of texts
            y: True labels
            method: 'perturbation' or 'gradient'
            n_samples: Number of samples to use
        
        Returns:
            Feature importance scores
        """
        print(f"Computing feature importance using {method} method...")
        
        # Sample data if too large
        if len(y) > n_samples:
            indices = np.random.choice(len(y), n_samples, replace=False)
            if isinstance(X, list):
                X_sample = [X[i] for i in indices]
            else:
                X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        
        if method == 'perturbation':
            return self._perturbation_importance(X_sample, y_sample)
        else:
            return self._gradient_importance(X_sample)
    
    def _perturbation_importance(self, X, y):
        """
        Compute feature importance by perturbation
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Prepare features
        if isinstance(X, list):
            X_features = self.model.prepare_features(X, fit=False)
        else:
            X_features = X
        
        # Baseline performance
        baseline_pred = self.model.predict(X_features)
        baseline_acc = np.mean(baseline_pred == y)
        
        # Feature importance scores
        importance_scores = np.zeros(X_features.shape[1])
        
        # Perturb each feature
        for feature_idx in range(X_features.shape[1]):
            # Create perturbed copy
            X_perturbed = X_features.copy()
            
            # Shuffle this feature
            X_perturbed[:, feature_idx] = np.random.permutation(X_perturbed[:, feature_idx])
            
            # Evaluate with perturbed feature
            perturbed_pred = self.model.predict(X_perturbed)
            perturbed_acc = np.mean(perturbed_pred == y)
            
            # Importance = drop in accuracy
            importance_scores[feature_idx] = baseline_acc - perturbed_acc
        
        # Map to feature names
        feature_names = self._get_feature_names()
        importance_dict = dict(zip(feature_names, importance_scores))
        
        return importance_dict
    
    def _gradient_importance(self, X):
        """
        Compute feature importance using gradients
        (Simplified version)
        """
        # Prepare features
        if isinstance(X, list):
            X_features = self.model.prepare_features(X, fit=False)
        else:
            X_features = X
        
        # Get predictions
        y_pred, cache = self.model.neural_network.forward(X_features, training=False)
        
        # Compute gradient of output with respect to input
        # Simplified: use absolute mean of input layer weights
        input_weights = self.model.neural_network.weights[0]
        importance_scores = np.mean(np.abs(input_weights), axis=1)
        
        # Normalize
        importance_scores = importance_scores / np.sum(importance_scores)
        
        # Map to feature names
        feature_names = self._get_feature_names()
        importance_dict = dict(zip(feature_names, importance_scores))
        
        return importance_dict
    
    def _get_feature_names(self):
        """Get all feature names"""
        feature_names = []
        
        # Handcrafted feature names
        if self.model.handcrafted_feature_names:
            feature_names.extend(self.model.handcrafted_feature_names)
        
        # TF-IDF feature names
        if self.model.tfidf_vectorizer:
            tfidf_names = [f"tfidf_{name}" for name in 
                          self.model.tfidf_vectorizer.get_feature_names_out()]
            feature_names.extend(tfidf_names[:self.model.tfidf_dim])
        
        # Embedding feature names
        if self.model.embedding_dim > 0:
            embedding_names = [f"embedding_{i}" for i in range(
                self.model.word2vec_model.vector_size)]
            embedding_names.extend(['emb_std', 'emb_var', 'emb_mag_mean', 
                                   'emb_mag_std', 'oov_ratio'])
            feature_names.extend(embedding_names[:self.model.embedding_dim])
        
        return feature_names
    
    def explain_prediction(self, text, headline=None, top_n=15):
        """
        Explain a single prediction
        
        Args:
            text: Article text
            headline: Article headline (optional)
            top_n: Number of top features to show
        
        Returns:
            Explanation dictionary
        """
        # Get prediction
        pred_proba = self.model.predict_proba([text], [headline] if headline else None)[0][0]
        pred_label = "FAKE" if pred_proba >= 0.5 else "REAL"
        
        # Extract features
        X_features = self.model.prepare_features([text], 
                                                [headline] if headline else None, 
                                                fit=False)
        
        # Get feature values
        feature_names = self._get_feature_names()
        feature_values = X_features[0]
        
        # Compute contribution (feature_value * weight approximation)
        # Use first layer weights as approximation
        weights = self.model.neural_network.weights[0]
        contributions = feature_values * np.mean(np.abs(weights), axis=1)
        
        # Get top contributing features
        top_indices = np.argsort(np.abs(contributions))[-top_n:][::-1]
        
        top_features = []
        for idx in top_indices:
            if idx < len(feature_names):
                top_features.append({
                    'feature': feature_names[idx],
                    'value': feature_values[idx],
                    'contribution': contributions[idx]
                })
        
        # Compute additional scores
        from src.features import (PsychologicalFeatureExtractor, 
                                 StructuralFeatureExtractor)
        
        psych_extractor = PsychologicalFeatureExtractor()
        struct_extractor = StructuralFeatureExtractor()
        
        psych_features = psych_extractor.extract_all_features(text)
        struct_features = struct_extractor.extract_all_features(text, headline)
        
        psychological_score = psych_extractor.compute_psychological_score(psych_features)
        sentiment_volatility = psych_extractor.compute_sentiment_volatility(psych_features)
        structural_integrity = struct_extractor.compute_structural_integrity_score(struct_features)
        
        # Generate analysis text
        analysis_text = f"This article was classified as {pred_label} with {pred_proba if pred_proba >= 0.5 else 1 - pred_proba:.1%} confidence. "
        
        if pred_label == "FAKE":
            analysis_text += f"Key concerns include: "
            concerns = []
            if psychological_score > 40:
                concerns.append(f"high psychological manipulation ({psychological_score:.0f}/100)")
            if structural_integrity < 50:
                concerns.append(f"low structural integrity ({structural_integrity:.0f}/100)")
            if sentiment_volatility > 60:
                concerns.append(f"high sentiment volatility ({sentiment_volatility:.0f}/100)")
            analysis_text += ", ".join(concerns) if concerns else "various indicators of fake news"
        else:
            analysis_text += f"The article shows good credibility with "
            strengths = []
            if structural_integrity > 70:
                strengths.append(f"strong structure ({structural_integrity:.0f}/100)")
            if psychological_score < 30:
                strengths.append(f"low manipulation ({psychological_score:.0f}/100)")
            analysis_text += ", ".join(strengths) if strengths else "generally reliable indicators"
        
        explanation = {
            'prediction': pred_label,
            'confidence': float(pred_proba if pred_proba >= 0.5 else 1 - pred_proba),
            'fake_probability': float(pred_proba),
            'real_probability': float(1 - pred_proba),
            'psychological_manipulation_score': psychological_score,
            'structural_integrity_score': structural_integrity,
            'sentiment_volatility_index': sentiment_volatility,
            'top_contributing_features': top_features,
            'top_features': top_features,  # Alias for UI compatibility
            'analysis': analysis_text,
            'text_preview': text[:200] + '...' if len(text) > 200 else text
        }
        
        return explanation
    
    def print_explanation(self, text, headline=None, top_n=15):
        """Print human-readable explanation"""
        explanation = self.explain_prediction(text, headline, top_n)
        
        print("="*70)
        print(" "*25 + "PREDICTION EXPLANATION")
        print("="*70)
        
        print(f"\n📰 Text Preview:")
        print(f"   {explanation['text_preview']}")
        
        if headline:
            print(f"\n📌 Headline: {headline}")
        
        print(f"\n🔍 PREDICTION: {explanation['prediction']}")
        print(f"   Confidence: {explanation['confidence']:.1%}")
        
        print(f"\n📊 Probability Scores:")
        print(f"   Real: {explanation['real_probability']:.1%}")
        print(f"   Fake: {explanation['fake_probability']:.1%}")
        
        print(f"\n🧠 Analysis Scores:")
        print(f"   Psychological Manipulation: {explanation['psychological_manipulation_score']:.1f}/100")
        print(f"   Structural Integrity:       {explanation['structural_integrity_score']:.1f}/100")
        print(f"   Sentiment Volatility:       {explanation['sentiment_volatility_index']:.1f}/100")
        
        print(f"\n✨ Top {len(explanation['top_contributing_features'])} Contributing Features:")
        for i, feat in enumerate(explanation['top_contributing_features'], 1):
            direction = "↑" if feat['contribution'] > 0 else "↓"
            print(f"   {i:2d}. {feat['feature']:40s} {direction} "
                  f"(value: {feat['value']:6.3f}, contrib: {feat['contribution']:6.3f})")
        
        print("="*70)
        
        return explanation
    
    def plot_feature_importance(self, X, y, top_n=20, save_path=None):
        """
        Plot feature importance
        
        Args:
            X: Feature matrix or texts
            y: Labels
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        # Compute importance
        importance_dict = self.compute_feature_importance(X, y)
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), 
                               key=lambda x: abs(x[1]), 
                               reverse=True)[:top_n]
        
        features, scores = zip(*sorted_features)
        
        # Plot
        plt.figure(figsize=(12, 8))
        colors = ['green' if s > 0 else 'red' for s in scores]
        plt.barh(range(len(features)), scores, color=colors, alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score (Impact on Accuracy)')
        plt.title(f'Top {top_n} Feature Importances')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_score_breakdown(self, text, headline=None, save_path=None):
        """
        Plot score breakdown for a single prediction
        
        Args:
            text: Article text
            headline: Article headline
            save_path: Path to save plot
        """
        explanation = self.explain_prediction(text, headline)
        
        # Scores to plot
        scores = {
            'Psychological\nManipulation': explanation['psychological_manipulation_score'],
            'Structural\nIntegrity': explanation['structural_integrity_score'],
            'Sentiment\nVolatility': explanation['sentiment_volatility_index'],
            'Confidence': explanation['confidence'] * 100
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = list(scores.keys())
        values = list(scores.values())
        colors = ['#ff6b6b', '#51cf66', '#ffd43b', '#4dabf7']
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylim(0, 100)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Prediction: {explanation["prediction"]} '
                    f'(Confidence: {explanation["confidence"]:.1%})', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Score breakdown saved to {save_path}")
        
        plt.show()
