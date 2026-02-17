"""
Main Fake News Detector System
Unified interface for detecting fake news
"""
import os
import pickle
import numpy as np

from config import SAVED_MODELS_DIR, THRESHOLDS
from src.models import HybridFakeNewsModel
from src.embeddings import CustomWord2Vec
from src.explainability import ModelExplainer
from src.features import (PsychologicalFeatureExtractor, 
                         StructuralFeatureExtractor,
                         CoherenceFeatureExtractor)


class FakeNewsDetector:
    """
    Main fake news detection system
    Provides a simple interface for detection with explainability
    """
    
    def __init__(self, model_path=None, word2vec_path=None):
        """
        Initialize the detector
        
        Args:
            model_path: Path to trained hybrid model
            word2vec_path: Path to trained Word2Vec model
        """
        self.model = None
        self.word2vec_model = None
        self.explainer = None
        
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(SAVED_MODELS_DIR, 'hybrid_model.pkl')
        if word2vec_path is None:
            word2vec_path = os.path.join(SAVED_MODELS_DIR, 'word2vec_model.pkl')
        
        # Load models
        self.load_models(model_path, word2vec_path)
    
    def load_models(self, model_path, word2vec_path):
        """Load trained models"""
        print("Loading models...")
        
        # Load Word2Vec
        if os.path.exists(word2vec_path):
            self.word2vec_model = CustomWord2Vec()
            self.word2vec_model.load(word2vec_path)
            print("✓ Word2Vec model loaded")
        else:
            print(f"⚠ Word2Vec model not found at {word2vec_path}")
        
        # Load hybrid model
        if os.path.exists(model_path):
            self.model = HybridFakeNewsModel(word2vec_model=self.word2vec_model)
            self.model.load(model_path)
            print("✓ Hybrid model loaded")
            
            # Initialize explainer
            self.explainer = ModelExplainer(self.model)
        else:
            print(f"⚠ Hybrid model not found at {model_path}")
        
        print("Models ready!\n")
    
    def detect(self, text, headline=None, explain=True, verbose=True):
        """
        Detect if news article is fake or real
        
        Args:
            text: Article text
            headline: Article headline (optional)
            explain: Whether to provide explanation
            verbose: Whether to print results
        
        Returns:
            Detection result dictionary
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Get prediction
        pred_proba = self.model.predict_proba([text], 
                                             [headline] if headline else None)[0][0]
        pred_label = "FAKE" if pred_proba >= THRESHOLDS['confidence_threshold'] else "REAL"
        confidence = pred_proba if pred_proba >= 0.5 else 1 - pred_proba
        
        # Compute additional scores and extract features
        from src.features import LinguisticFeatureExtractor
        
        psych_extractor = PsychologicalFeatureExtractor()
        struct_extractor = StructuralFeatureExtractor()
        coher_extractor = CoherenceFeatureExtractor()
        ling_extractor = LinguisticFeatureExtractor()
        
        psych_features = psych_extractor.extract_all_features(text)
        struct_features = struct_extractor.extract_all_features(text, headline)
        coher_features = coher_extractor.extract_all_features(text)
        ling_features = ling_extractor.extract_all_features(text)
        
        psychological_score = psych_extractor.compute_psychological_score(psych_features)
        sentiment_volatility = psych_extractor.compute_sentiment_volatility(psych_features)
        structural_integrity = struct_extractor.compute_structural_integrity_score(struct_features)
        coherence_score_val = coher_extractor.compute_coherence_score(coher_features)
        
        # Compute linguistic score (inverse of complexity)
        linguistic_score = 100 - min(100, ling_features.get('avg_word_length', 5) * 10)
        
        # Compute credibility score (weighted average)
        credibility_score = (
            structural_integrity * 0.35 +
            coherence_score_val * 0.30 +
            (100 - psychological_score) * 0.25 +
            linguistic_score * 0.10
        )
        
        # Identify key indicators
        warning_signs = []
        credibility_markers = []
        
        if psychological_score > 40:
            warning_signs.append(f"High psychological manipulation detected ({psychological_score:.0f}/100)")
        if psych_features.get('exaggeration_score', 0) > 0.3:
            warning_signs.append("Excessive exaggeration detected")
        if struct_features.get('clickbait_score', 0) > 0.5:
            warning_signs.append("Clickbait patterns detected")
        if sentiment_volatility > 60:
            warning_signs.append(f"High sentiment volatility ({sentiment_volatility:.0f}/100)")
        if struct_features.get('caps_ratio', 0) > 0.15:
            warning_signs.append("Excessive capitalization")
        
        if structural_integrity > 70:
            credibility_markers.append(f"Strong structural integrity ({structural_integrity:.0f}/100)")
        if coherence_score_val > 70:
            credibility_markers.append(f"High coherence score ({coherence_score_val:.0f}/100)")
        if psych_features.get('sentiment_polarity', 0) > -0.2 and psych_features.get('sentiment_polarity', 0) < 0.2:
            credibility_markers.append("Balanced sentiment")
        if struct_features.get('has_sources', False):
            credibility_markers.append("Contains source citations")
        
        # Build result with all required fields
        result = {
            'prediction': pred_label,
            'confidence': float(confidence),
            'fake_probability': float(pred_proba),
            'real_probability': float(1 - pred_proba),
            'credibility_score': float(credibility_score),
            'psychological_score': float(psychological_score),
            'structural_score': float(structural_integrity),
            'coherence_score': float(coherence_score_val),
            'linguistic_score': float(linguistic_score),
            'psychological_manipulation_score': psychological_score,
            'structural_integrity_score': structural_integrity,
            'sentiment_volatility_index': sentiment_volatility,
            'headline': headline,
            'text_length': len(text),
            'risk_assessment': self._assess_risk(pred_proba, psychological_score, 
                                                 structural_integrity, sentiment_volatility),
            'psychological_features': psych_features,
            'structural_features': struct_features,
            'coherence_features': coher_features,
            'linguistic_features': ling_features,
            'key_indicators': {
                'warning_signs': warning_signs,
                'credibility_markers': credibility_markers
            }
        }
        
        # Add explanation if requested
        if explain and self.explainer:
            explanation = self.explainer.explain_prediction(text, headline, top_n=10)
            result['explanation'] = explanation
        
        # Print if verbose
        if verbose:
            self.print_result(result)
        
        return result
    
    def _assess_risk(self, fake_prob, psych_score, struct_score, volatility):
        """
        Assess overall risk level
        
        Returns:
            Risk assessment string
        """
        # Calculate risk score
        risk = (fake_prob * 40 + 
               psych_score * 0.3 + 
               (100 - struct_score) * 0.2 + 
               volatility * 0.1)
        
        if risk >= 70:
            return "HIGH RISK"
        elif risk >= 40:
            return "MEDIUM RISK"
        else:
            return "LOW RISK"
    
    def print_result(self, result):
        """Print detection result in a formatted way"""
        print("\n" + "="*70)
        print(" "*25 + "FAKE NEWS DETECTION RESULT")
        print("="*70)
        
        if result.get('headline'):
            print(f"\n📌 Headline: {result['headline']}")
        
        print(f"\n📰 Article Length: {result['text_length']} characters")
        
        # Main prediction
        print(f"\n🔍 PREDICTION: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Risk Level: {result['risk_assessment']}")
        
        # Probabilities
        print(f"\n📊 Probability Scores:")
        print(f"   Real News: {result['real_probability']:.1%}")
        print(f"   Fake News: {result['fake_probability']:.1%}")
        
        # Analysis scores
        print(f"\n🧠 Detailed Analysis:")
        print(f"   Psychological Manipulation: {result['psychological_manipulation_score']:5.1f}/100 ", end='')
        self._print_score_bar(result['psychological_manipulation_score'], threshold=40)
        
        print(f"   Structural Integrity:       {result['structural_integrity_score']:5.1f}/100 ", end='')
        self._print_score_bar(result['structural_integrity_score'], threshold=50, reverse=True)
        
        print(f"   Sentiment Volatility:       {result['sentiment_volatility_index']:5.1f}/100 ", end='')
        self._print_score_bar(result['sentiment_volatility_index'], threshold=60)
        
        print(f"   Coherence Score:            {result['coherence_score']:5.1f}/100 ", end='')
        self._print_score_bar(result['coherence_score'], threshold=50, reverse=True)
        
        # Explanation
        if 'explanation' in result and result['explanation']:
            exp = result['explanation']
            if 'top_contributing_features' in exp:
                print(f"\n✨ Top Contributing Features:")
                for i, feat in enumerate(exp['top_contributing_features'][:5], 1):
                    direction = "↑" if feat['contribution'] > 0 else "↓"
                    print(f"   {i}. {feat['feature']:35s} {direction}")
        
        print("\n" + "="*70 + "\n")
    
    def _print_score_bar(self, score, threshold=50, reverse=False):
        """Print a visual score bar"""
        # Determine color based on score
        if reverse:
            # For scores where higher is better
            if score >= 70:
                indicator = "✓ Good"
            elif score >= threshold:
                indicator = "○ Fair"
            else:
                indicator = "✗ Poor"
        else:
            # For scores where lower is better
            if score >= 70:
                indicator = "✗ High"
            elif score >= threshold:
                indicator = "○ Medium"
            else:
                indicator = "✓ Low"
        
        # Create bar
        filled = int(score / 10)
        bar = "█" * filled + "░" * (10 - filled)
        print(f"[{bar}] {indicator}")
    
    def detect_batch(self, texts, headlines=None, explain=False):
        """
        Detect fake news in a batch of articles
        
        Args:
            texts: List of article texts
            headlines: List of headlines (optional)
            explain: Whether to provide explanations
        
        Returns:
            List of detection results
        """
        results = []
        
        print(f"\nProcessing {len(texts)} articles...")
        
        for i, text in enumerate(texts):
            headline = headlines[i] if headlines and i < len(headlines) else None
            result = self.detect(text, headline, explain=explain, verbose=False)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(texts)} articles")
        
        print("Batch processing complete!\n")
        
        # Print summary
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results):
        """Print summary of batch results"""
        print("="*70)
        print(" "*25 + "BATCH DETECTION SUMMARY")
        print("="*70)
        
        total = len(results)
        fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
        real_count = total - fake_count
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_psych = np.mean([r['psychological_manipulation_score'] for r in results])
        avg_struct = np.mean([r['structural_integrity_score'] for r in results])
        
        print(f"\nTotal Articles: {total}")
        print(f"  Real News: {real_count} ({real_count/total*100:.1f}%)")
        print(f"  Fake News: {fake_count} ({fake_count/total*100:.1f}%)")
        
        print(f"\nAverage Scores:")
        print(f"  Confidence:                 {avg_confidence:.1%}")
        print(f"  Psychological Manipulation: {avg_psych:.1f}/100")
        print(f"  Structural Integrity:       {avg_struct:.1f}/100")
        
        print("="*70 + "\n")
    
    def get_explainer(self):
        """Get the model explainer for advanced analysis"""
        return self.explainer


# Convenience function
def detect_fake_news(text, headline=None, model_path=None, explain=True):
    """
    Convenience function to detect fake news
    
    Args:
        text: Article text
        headline: Article headline (optional)
        model_path: Path to model (optional)
        explain: Whether to provide explanation
    
    Returns:
        Detection result
    """
    detector = FakeNewsDetector(model_path=model_path)
    return detector.detect(text, headline, explain=explain)
