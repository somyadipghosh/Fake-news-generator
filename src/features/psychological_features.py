"""
Psychological and emotional feature extraction
Analyzes sentiment, polarity shifts, exaggeration, and emotional manipulation
"""
import re
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sent_tokenize, word_tokenize
from collections import Counter


class PsychologicalFeatureExtractor:
    """Extract psychological and emotional manipulation indicators"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        # Emotion lexicons
        self.fear_words = {
            'afraid', 'fear', 'terror', 'scary', 'frightening', 'panic', 'horror',
            'dread', 'anxiety', 'worried', 'concern', 'threat', 'danger', 'risk',
            'disaster', 'catastrophe', 'crisis', 'deadly', 'fatal', 'dangerous'
        }
        
        self.anger_words = {
            'angry', 'rage', 'fury', 'outrage', 'furious', 'mad', 'hate', 'hatred',
            'disgust', 'disgusting', 'outrageous', 'infuriating', 'enraged', 'livid',
            'hostile', 'aggressive', 'violent', 'attack', 'fight', 'war'
        }
        
        self.exaggeration_words = {
            'always', 'never', 'everyone', 'nobody', 'everything', 'nothing',
            'completely', 'totally', 'absolutely', 'extremely', 'incredible',
            'unbelievable', 'amazing', 'shocking', 'stunning', 'extraordinary',
            'unprecedented', 'massive', 'huge', 'enormous', 'devastating'
        }
        
        self.urgency_words = {
            'urgent', 'immediately', 'now', 'hurry', 'quick', 'fast', 'breaking',
            'alert', 'warning', 'emergency', 'critical', 'crucial', 'must', 'need'
        }
        
        self.certainty_words = {
            'definitely', 'certainly', 'obviously', 'clearly', 'undoubtedly',
            'surely', 'absolutely', 'proven', 'fact', 'truth', 'confirmed'
        }
        
    def extract_all_features(self, text):
        """Extract all psychological features"""
        if not text or not isinstance(text, str):
            return self._get_empty_features()
        
        features = {}
        
        # Sentiment analysis
        features.update(self._analyze_sentiment(text))
        
        # Emotional manipulation
        features.update(self._analyze_emotional_manipulation(text))
        
        # Polarity shifts
        features.update(self._analyze_polarity_shifts(text))
        
        # Exaggeration and hyperbole
        features.update(self._analyze_exaggeration(text))
        
        # Urgency and pressure
        features.update(self._analyze_urgency(text))
        
        # Certainty and assertiveness
        features.update(self._analyze_certainty(text))
        
        return features
    
    def _analyze_sentiment(self, text):
        """Analyze overall sentiment using VADER"""
        features = {}
        
        # Overall sentiment
        sentiment_scores = self.vader.polarity_scores(text)
        features['sentiment_positive'] = sentiment_scores['pos']
        features['sentiment_negative'] = sentiment_scores['neg']
        features['sentiment_neutral'] = sentiment_scores['neu']
        features['sentiment_compound'] = sentiment_scores['compound']
        
        # Sentence-level sentiment statistics
        sentences = sent_tokenize(text)
        if sentences:
            sent_scores = [self.vader.polarity_scores(s)['compound'] for s in sentences]
            features['sentiment_mean'] = np.mean(sent_scores)
            features['sentiment_std'] = np.std(sent_scores)
            features['sentiment_range'] = max(sent_scores) - min(sent_scores)
        else:
            features['sentiment_mean'] = 0
            features['sentiment_std'] = 0
            features['sentiment_range'] = 0
        
        return features
    
    def _analyze_emotional_manipulation(self, text):
        """Detect emotional manipulation through fear and anger"""
        features = {}
        
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        word_count = len(words) if words else 1
        
        # Fear amplification
        fear_count = sum(1 for word in words if word in self.fear_words)
        features['fear_word_ratio'] = fear_count / word_count
        features['fear_word_count'] = fear_count
        
        # Anger amplification
        anger_count = sum(1 for word in words if word in self.anger_words)
        features['anger_word_ratio'] = anger_count / word_count
        features['anger_word_count'] = anger_count
        
        # Combined emotional manipulation score
        features['emotional_manipulation_score'] = (fear_count + anger_count) / word_count * 100
        
        return features
    
    def _analyze_polarity_shifts(self, text):
        """Analyze sentiment polarity shifts between sentences"""
        features = {}
        
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            features['polarity_shifts'] = 0
            features['polarity_shift_ratio'] = 0
            features['avg_polarity_change'] = 0
            features['max_polarity_change'] = 0
            return features
        
        # Calculate sentiment for each sentence
        sentiments = [self.vader.polarity_scores(s)['compound'] for s in sentences]
        
        # Count polarity shifts (sign changes)
        shifts = 0
        changes = []
        for i in range(len(sentiments) - 1):
            change = abs(sentiments[i+1] - sentiments[i])
            changes.append(change)
            if sentiments[i] * sentiments[i+1] < 0:  # Sign change
                shifts += 1
        
        features['polarity_shifts'] = shifts
        features['polarity_shift_ratio'] = shifts / (len(sentences) - 1)
        features['avg_polarity_change'] = np.mean(changes) if changes else 0
        features['max_polarity_change'] = max(changes) if changes else 0
        
        return features
    
    def _analyze_exaggeration(self, text):
        """Detect exaggeration and hyperbole markers"""
        features = {}
        
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        word_count = len(words) if words else 1
        
        # Exaggeration word count
        exag_count = sum(1 for word in words if word in self.exaggeration_words)
        features['exaggeration_word_ratio'] = exag_count / word_count
        features['exaggeration_word_count'] = exag_count
        
        # Intensifiers (very, really, so, extremely, etc.)
        intensifiers = ['very', 'really', 'so', 'extremely', 'incredibly', 'absolutely',
                       'totally', 'completely', 'utterly', 'highly', 'seriously']
        intensifier_count = sum(words.count(w) for w in intensifiers)
        features['intensifier_ratio'] = intensifier_count / word_count
        
        # Superlatives (best, worst, most, least, etc.)
        superlative_patterns = [r'\w+est\b', r'\bmost\s+\w+', r'\bleast\s+\w+']
        superlative_count = sum(len(re.findall(pattern, text_lower)) for pattern in superlative_patterns)
        features['superlative_ratio'] = superlative_count / word_count
        
        return features
    
    def _analyze_urgency(self, text):
        """Analyze urgency and time pressure indicators"""
        features = {}
        
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        word_count = len(words) if words else 1
        
        # Urgency word count
        urgency_count = sum(1 for word in words if word in self.urgency_words)
        features['urgency_word_ratio'] = urgency_count / word_count
        features['urgency_word_count'] = urgency_count
        
        # Multiple exclamation marks (urgency indicator)
        features['urgency_punctuation_score'] = (
            text.count('!') + text.count('!!!') * 2 + 
            len(re.findall(r'!{2,}', text)) * 3
        ) / word_count
        
        return features
    
    def _analyze_certainty(self, text):
        """Analyze certainty and assertiveness markers"""
        features = {}
        
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        word_count = len(words) if words else 1
        
        # Certainty word count
        certainty_count = sum(1 for word in words if word in self.certainty_words)
        features['certainty_word_ratio'] = certainty_count / word_count
        features['certainty_word_count'] = certainty_count
        
        # Hedging words (opposite of certainty)
        hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'may',
                        'seem', 'appear', 'suggest', 'indicate', 'probably']
        hedging_count = sum(words.count(w) for w in hedging_words)
        features['hedging_ratio'] = hedging_count / word_count
        
        # Assertiveness score (high certainty, low hedging)
        features['assertiveness_score'] = (certainty_count - hedging_count) / word_count
        
        return features
    
    def compute_psychological_score(self, features):
        """
        Compute overall psychological manipulation score (0-100)
        Higher score = more manipulation
        """
        score = 0
        
        # Weight different factors
        score += features.get('emotional_manipulation_score', 0) * 0.3
        score += features.get('exaggeration_word_ratio', 0) * 100 * 0.2
        score += features.get('urgency_word_ratio', 0) * 100 * 0.15
        score += features.get('polarity_shift_ratio', 0) * 100 * 0.15
        score += features.get('assertiveness_score', 0) * 50 * 0.1
        score += abs(features.get('sentiment_compound', 0)) * 50 * 0.1
        
        return min(100, max(0, score))
    
    def compute_sentiment_volatility(self, features):
        """
        Compute sentiment volatility index (0-100)
        Higher score = more volatile sentiment
        """
        volatility = 0
        
        volatility += features.get('sentiment_std', 0) * 50
        volatility += features.get('polarity_shift_ratio', 0) * 30
        volatility += features.get('sentiment_range', 0) * 20
        
        return min(100, max(0, volatility))
    
    def _get_empty_features(self):
        """Return empty feature dictionary"""
        return {
            'sentiment_positive': 0, 'sentiment_negative': 0, 'sentiment_neutral': 0,
            'sentiment_compound': 0, 'sentiment_mean': 0, 'sentiment_std': 0,
            'sentiment_range': 0, 'fear_word_ratio': 0, 'fear_word_count': 0,
            'anger_word_ratio': 0, 'anger_word_count': 0,
            'emotional_manipulation_score': 0, 'polarity_shifts': 0,
            'polarity_shift_ratio': 0, 'avg_polarity_change': 0,
            'max_polarity_change': 0, 'exaggeration_word_ratio': 0,
            'exaggeration_word_count': 0, 'intensifier_ratio': 0,
            'superlative_ratio': 0, 'urgency_word_ratio': 0,
            'urgency_word_count': 0, 'urgency_punctuation_score': 0,
            'certainty_word_ratio': 0, 'certainty_word_count': 0,
            'hedging_ratio': 0, 'assertiveness_score': 0
        }
    
    def get_feature_names(self):
        """Return list of all feature names"""
        return list(self._get_empty_features().keys())
