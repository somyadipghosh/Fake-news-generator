"""
Contextual coherence analysis
Analyzes sentence similarity and logical flow within articles
"""
import numpy as np
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean


class CoherenceFeatureExtractor:
    """Extract contextual coherence features"""
    
    def __init__(self):
        pass
    
    def extract_all_features(self, text):
        """Extract all coherence features"""
        if not text or not isinstance(text, str):
            return self._get_empty_features()
        
        features = {}
        
        # Sentence-level coherence
        features.update(self._analyze_sentence_coherence(text))
        
        # Topic consistency
        features.update(self._analyze_topic_consistency(text))
        
        # Logical flow
        features.update(self._analyze_logical_flow(text))
        
        return features
    
    def _analyze_sentence_coherence(self, text):
        """Analyze coherence between consecutive sentences"""
        features = {}
        
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return {
                'avg_sentence_similarity': 0,
                'min_sentence_similarity': 0,
                'max_sentence_similarity': 0,
                'sentence_similarity_std': 0,
                'low_coherence_transitions': 0
            }
        
        try:
            # Create TF-IDF vectors for sentences
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            sentence_vectors = vectorizer.fit_transform(sentences)
            
            # Calculate similarity between consecutive sentences
            similarities = []
            for i in range(len(sentences) - 1):
                sim = cosine_similarity(sentence_vectors[i:i+1], 
                                       sentence_vectors[i+1:i+2])[0][0]
                similarities.append(sim)
            
            if similarities:
                features['avg_sentence_similarity'] = np.mean(similarities)
                features['min_sentence_similarity'] = np.min(similarities)
                features['max_sentence_similarity'] = np.max(similarities)
                features['sentence_similarity_std'] = np.std(similarities)
                
                # Count low coherence transitions (similarity < 0.1)
                features['low_coherence_transitions'] = sum(1 for s in similarities if s < 0.1)
                features['low_coherence_ratio'] = features['low_coherence_transitions'] / len(similarities)
            else:
                features['avg_sentence_similarity'] = 0
                features['min_sentence_similarity'] = 0
                features['max_sentence_similarity'] = 0
                features['sentence_similarity_std'] = 0
                features['low_coherence_transitions'] = 0
                features['low_coherence_ratio'] = 0
                
        except Exception as e:
            features = {
                'avg_sentence_similarity': 0,
                'min_sentence_similarity': 0,
                'max_sentence_similarity': 0,
                'sentence_similarity_std': 0,
                'low_coherence_transitions': 0,
                'low_coherence_ratio': 0
            }
        
        return features
    
    def _analyze_topic_consistency(self, text):
        """Analyze topic consistency across the document"""
        features = {}
        
        sentences = sent_tokenize(text)
        
        if len(sentences) < 3:
            return {
                'topic_consistency_score': 0,
                'topic_drift': 0,
                'intro_conclusion_similarity': 0
            }
        
        try:
            # Split document into sections
            intro = ' '.join(sentences[:len(sentences)//3])
            middle = ' '.join(sentences[len(sentences)//3:2*len(sentences)//3])
            conclusion = ' '.join(sentences[2*len(sentences)//3:])
            
            sections = [intro, middle, conclusion]
            
            # Create TF-IDF vectors for sections
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            section_vectors = vectorizer.fit_transform(sections)
            
            # Calculate pairwise similarities
            intro_middle_sim = cosine_similarity(section_vectors[0:1], section_vectors[1:2])[0][0]
            middle_conclusion_sim = cosine_similarity(section_vectors[1:2], section_vectors[2:3])[0][0]
            intro_conclusion_sim = cosine_similarity(section_vectors[0:1], section_vectors[2:3])[0][0]
            
            # Topic consistency score
            features['topic_consistency_score'] = np.mean([intro_middle_sim, 
                                                           middle_conclusion_sim, 
                                                           intro_conclusion_sim])
            
            # Topic drift (decrease in similarity from intro to conclusion)
            features['topic_drift'] = max(0, intro_middle_sim - middle_conclusion_sim)
            
            # Intro-conclusion similarity (circular structure indicator)
            features['intro_conclusion_similarity'] = intro_conclusion_sim
            
        except Exception as e:
            features = {
                'topic_consistency_score': 0,
                'topic_drift': 0,
                'intro_conclusion_similarity': 0
            }
        
        return features
    
    def _analyze_logical_flow(self, text):
        """Analyze logical flow and discourse markers"""
        features = {}
        
        text_lower = text.lower()
        
        # Discourse markers for logical connections
        causal_markers = ['because', 'therefore', 'thus', 'hence', 'consequently', 
                         'as a result', 'so', 'since', 'for this reason']
        contrast_markers = ['however', 'but', 'although', 'though', 'yet', 'nevertheless',
                           'on the other hand', 'in contrast', 'despite', 'whereas']
        addition_markers = ['additionally', 'furthermore', 'moreover', 'also', 'besides',
                           'in addition', 'as well', 'similarly', 'likewise']
        temporal_markers = ['then', 'next', 'after', 'before', 'finally', 'subsequently',
                           'meanwhile', 'first', 'second', 'lastly']
        
        # Count discourse markers
        features['causal_marker_count'] = sum(text_lower.count(marker) for marker in causal_markers)
        features['contrast_marker_count'] = sum(text_lower.count(marker) for marker in contrast_markers)
        features['addition_marker_count'] = sum(text_lower.count(marker) for marker in addition_markers)
        features['temporal_marker_count'] = sum(text_lower.count(marker) for marker in temporal_markers)
        
        total_markers = (features['causal_marker_count'] + 
                        features['contrast_marker_count'] + 
                        features['addition_marker_count'] + 
                        features['temporal_marker_count'])
        
        sentences = sent_tokenize(text)
        features['discourse_marker_density'] = total_markers / len(sentences) if sentences else 0
        
        # Logical flow score (more markers = better flow)
        features['logical_flow_score'] = min(1.0, features['discourse_marker_density'] / 2.0)
        
        # Check for abrupt topic changes (lack of markers)
        features['abrupt_transition_risk'] = 1.0 - features['logical_flow_score']
        
        return features
    
    def compute_coherence_score(self, features):
        """
        Compute overall coherence score (0-100)
        Higher score = better coherence
        """
        score = 0
        
        # Sentence coherence (0-40 points)
        score += features.get('avg_sentence_similarity', 0) * 40
        
        # Topic consistency (0-30 points)
        score += features.get('topic_consistency_score', 0) * 30
        
        # Logical flow (0-20 points)
        score += features.get('logical_flow_score', 0) * 20
        
        # Penalty for low coherence transitions (0-10 points)
        low_coherence_penalty = features.get('low_coherence_ratio', 0) * 10
        score -= low_coherence_penalty
        
        return max(0, min(100, score))
    
    def _get_empty_features(self):
        """Return empty feature dictionary"""
        return {
            'avg_sentence_similarity': 0,
            'min_sentence_similarity': 0,
            'max_sentence_similarity': 0,
            'sentence_similarity_std': 0,
            'low_coherence_transitions': 0,
            'low_coherence_ratio': 0,
            'topic_consistency_score': 0,
            'topic_drift': 0,
            'intro_conclusion_similarity': 0,
            'causal_marker_count': 0,
            'contrast_marker_count': 0,
            'addition_marker_count': 0,
            'temporal_marker_count': 0,
            'discourse_marker_density': 0,
            'logical_flow_score': 0,
            'abrupt_transition_risk': 0
        }
    
    def get_feature_names(self):
        """Return list of all feature names"""
        return list(self._get_empty_features().keys())
