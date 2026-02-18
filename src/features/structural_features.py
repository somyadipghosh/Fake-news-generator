"""
Structural anomaly detection
Analyzes headline-body consistency, clickbait patterns, and structural integrity
"""
import re
import numpy as np
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class StructuralFeatureExtractor:
    """Extract structural anomaly features"""
    
    def __init__(self):
        self.clickbait_patterns = [
            r'\d+\s+(reasons?|ways?|things?|facts?|tricks?)',
            r'you\s+won\'?t\s+believe',
            r'this\s+is\s+why',
            r'what\s+happened\s+next',
            r'shocking',
            r'amazing',
            r'incredible',
            r'unbelievable',
            r'must\s+see',
            r'won\'?t\s+believe',
            r'will\s+shock\s+you',
            r'number\s+\d+\s+will',
            r'doctors\s+hate',
            r'hate\s+him',
            r'one\s+weird\s+trick',
        ]
        
    def extract_all_features(self, text, headline=None):
        """Extract all structural features"""
        if not text or not isinstance(text, str):
            return self._get_empty_features()
        
        features = {}
        
        # Basic structural features
        features.update(self._analyze_structure(text))
        
        # Clickbait detection
        if headline:
            features.update(self._detect_clickbait(headline))
            features.update(self._analyze_headline_body_consistency(headline, text))
        else:
            features.update(self._get_empty_headline_features())
        
        # Capitalization anomalies
        features.update(self._analyze_capitalization_anomalies(text))
        
        # Punctuation manipulation
        features.update(self._analyze_punctuation_manipulation(text))
        
        # Repetition patterns
        features.update(self._analyze_repetition(text))
        
        # Quote and citation patterns
        features.update(self._analyze_citations(text))
        
        return features
    
    def _analyze_structure(self, text):
        """Analyze basic document structure"""
        features = {}
        
        sentences = sent_tokenize(text)
        
        if not sentences:
            return {
                'paragraph_count': 0,
                'avg_paragraph_length': 0,
                'sentence_length_variance': 0,
                'structure_score': 0
            }
        
        # Paragraph detection (simple: split by double newlines)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        features['paragraph_count'] = len(paragraphs)
        features['avg_paragraph_length'] = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0
        
        # Sentence length variance
        sent_lengths = [len(word_tokenize(s)) for s in sentences]
        features['sentence_length_variance'] = np.var(sent_lengths) if sent_lengths else 0
        features['min_sentence_length'] = min(sent_lengths) if sent_lengths else 0
        features['max_sentence_length'] = max(sent_lengths) if sent_lengths else 0
        
        # Very short sentences (potential for fragmentation)
        features['very_short_sentence_ratio'] = sum(1 for l in sent_lengths if l < 5) / len(sent_lengths) if sent_lengths else 0
        
        # Very long sentences (potential for run-ons)
        features['very_long_sentence_ratio'] = sum(1 for l in sent_lengths if l > 40) / len(sent_lengths) if sent_lengths else 0
        
        return features
    
    def _detect_clickbait(self, headline):
        """Detect clickbait patterns in headline"""
        features = {}
        
        headline_lower = headline.lower()
        
        # Count clickbait pattern matches
        clickbait_matches = sum(1 for pattern in self.clickbait_patterns 
                               if re.search(pattern, headline_lower))
        features['clickbait_pattern_count'] = clickbait_matches
        features['is_clickbait'] = 1 if clickbait_matches > 0 else 0
        
        # Additional clickbait indicators
        features['headline_has_number'] = 1 if re.search(r'\d+', headline) else 0
        features['headline_has_question'] = 1 if '?' in headline else 0
        features['headline_word_count'] = len(headline.split())
        
        # Excessive punctuation in headline
        features['headline_exclamation'] = headline.count('!')
        features['headline_has_multiple_exclamation'] = 1 if '!!' in headline else 0
        
        # All caps words in headline
        words = headline.split()
        features['headline_all_caps_words'] = sum(1 for w in words if w.isupper() and len(w) > 1)
        
        # Sensationalism indicators
        sensational_words = ['shocking', 'amazing', 'incredible', 'unbelievable', 
                            'stunning', 'breaking', 'exclusive', 'revealed']
        features['headline_sensational_words'] = sum(1 for word in sensational_words 
                                                     if word in headline_lower)
        
        return features
    
    def _analyze_headline_body_consistency(self, headline, text):
        """Analyze consistency between headline and article body"""
        features = {}
        
        try:
            # Use TF-IDF to compute semantic similarity
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            
            # Get first few sentences as article intro
            sentences = sent_tokenize(text)
            intro = ' '.join(sentences[:3]) if len(sentences) >= 3 else text[:500]
            
            # Compute similarity
            tfidf_matrix = vectorizer.fit_transform([headline, intro])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            features['headline_body_similarity'] = similarity
            features['headline_body_mismatch'] = 1 - similarity
            
            # Check if headline keywords appear in body
            headline_words = set(word_tokenize(headline.lower()))
            headline_words = {w for w in headline_words if w.isalnum() and len(w) > 3}
            
            body_words = set(word_tokenize(text.lower()))
            
            if headline_words:
                overlap = len(headline_words & body_words) / len(headline_words)
                features['headline_keyword_overlap'] = overlap
            else:
                features['headline_keyword_overlap'] = 0
                
        except Exception as e:
            features['headline_body_similarity'] = 0.5
            features['headline_body_mismatch'] = 0.5
            features['headline_keyword_overlap'] = 0.5
        
        return features
    
    def _analyze_capitalization_anomalies(self, text):
        """Detect excessive or unusual capitalization"""
        features = {}
        
        if not text:
            return self._get_empty_capitalization_features()
        
        # All caps words
        words = text.split()
        all_caps_words = [w for w in words if w.isupper() and len(w) > 1]
        features['all_caps_word_count'] = len(all_caps_words)
        features['all_caps_word_ratio'] = len(all_caps_words) / len(words) if words else 0
        
        # Consecutive all caps words
        consecutive_caps = 0
        for i in range(len(words) - 1):
            if words[i].isupper() and words[i+1].isupper():
                consecutive_caps += 1
        features['consecutive_caps_count'] = consecutive_caps
        
        # Mixed case anomalies (LiKe ThIs)
        mixed_case_words = [w for w in words if w.isalpha() and 
                           not w.isupper() and not w.islower() and not w.istitle()]
        features['mixed_case_word_ratio'] = len(mixed_case_words) / len(words) if words else 0
        
        # Capital letters in middle of sentences
        sentences = sent_tokenize(text)
        mid_sentence_caps = 0
        for sent in sentences:
            sent_words = sent.split()[1:-1]  # Exclude first and last word
            mid_sentence_caps += sum(1 for w in sent_words if w and w[0].isupper() 
                                    and not any(c in w for c in ['/', '.', '-', '@']))
        features['mid_sentence_caps_ratio'] = mid_sentence_caps / len(words) if words else 0
        
        return features
    
    def _analyze_punctuation_manipulation(self, text):
        """Detect punctuation manipulation and abuse"""
        features = {}
        
        # Multiple exclamation marks
        features['multiple_exclamation_count'] = len(re.findall(r'!{2,}', text))
        features['max_consecutive_exclamation'] = len(max(re.findall(r'!+', text), key=len, default=''))
        
        # Multiple question marks
        features['multiple_question_count'] = len(re.findall(r'\?{2,}', text))
        
        # Mixed punctuation (e.g., "?!", "!?!")
        features['mixed_punctuation_count'] = len(re.findall(r'[!?]{2,}', text))
        
        # Ellipsis abuse
        features['ellipsis_count'] = text.count('...') + text.count('…')
        features['multiple_ellipsis'] = len(re.findall(r'\.{4,}', text))
        
        # Quotation mark anomaly
        single_quotes = text.count("'")
        double_quotes = text.count('"')
        features['quote_imbalance'] = abs(single_quotes % 2) + abs(double_quotes % 2)
        
        # Dash and hyphen usage
        features['dash_count'] = text.count('—') + text.count('–')
        features['hyphen_count'] = text.count('-')
        
        # Parentheses usage
        features['parentheses_count'] = text.count('(') + text.count(')')
        features['parentheses_imbalance'] = abs(text.count('(') - text.count(')'))
        
        return features
    
    def _analyze_repetition(self, text):
        """Detect repetition patterns and bias"""
        features = {}
        
        words = word_tokenize(text.lower())
        
        if not words:
            return self._get_empty_repetition_features()
        
        # Word frequency analysis
        word_counts = {}
        for word in words:
            if word.isalnum():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        if word_counts:
            # Most repeated word frequency
            max_repetition = max(word_counts.values())
            features['max_word_repetition'] = max_repetition
            features['max_word_repetition_ratio'] = max_repetition / len(words)
            
            # Count of highly repeated words (>5 times)
            highly_repeated = sum(1 for count in word_counts.values() if count > 5)
            features['highly_repeated_word_count'] = highly_repeated
        else:
            features['max_word_repetition'] = 0
            features['max_word_repetition_ratio'] = 0
            features['highly_repeated_word_count'] = 0
        
        # Phrase repetition (bigrams)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        if bigrams:
            bigram_counts = {}
            for bigram in bigrams:
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            max_bigram_rep = max(bigram_counts.values())
            features['max_phrase_repetition'] = max_bigram_rep
        else:
            features['max_phrase_repetition'] = 0
        
        # Sentence repetition
        sentences = sent_tokenize(text.lower())
        if len(sentences) > 1:
            unique_sentences = len(set(sentences))
            features['sentence_repetition_ratio'] = 1 - (unique_sentences / len(sentences))
        else:
            features['sentence_repetition_ratio'] = 0
        
        return features
    
    def _analyze_citations(self, text):
        """Analyze quotes, citations, and attribution patterns"""
        features = {}
        
        # Quote detection
        features['quote_count'] = text.count('"') // 2  # Approximate quote pairs
        
        # Attribution patterns (said, according to, etc.)
        attribution_patterns = [
            r'\bsaid\b', r'\baccording to\b', r'\breported\b', 
            r'\bstated\b', r'\bclaimed\b', r'\btold\b'
        ]
        attribution_count = sum(len(re.findall(pattern, text.lower())) 
                               for pattern in attribution_patterns)
        features['attribution_count'] = attribution_count
        
        # Source detection (basic patterns)
        source_patterns = [
            r'source:', r'sources?:', r'\([\w\s]+\)', r'http[s]?://',
            r'www\.', r'\.com', r'\.org', r'\.gov'
        ]
        source_count = sum(len(re.findall(pattern, text.lower())) 
                          for pattern in source_patterns)
        features['source_reference_count'] = source_count
        
        # Quote to word ratio
        words = word_tokenize(text)
        features['quote_density'] = features['quote_count'] / len(words) if words else 0
        
        return features
    
    def compute_structural_integrity_score(self, features):
        """
        Compute overall structural integrity score (0-100)
        Higher score = better structural integrity
        """
        score = 100
        
        # Deduct for clickbait indicators (max 20 points)
        clickbait_deduction = min(20, features.get('clickbait_pattern_count', 0) * 8)
        score -= clickbait_deduction
        score -= min(10, features.get('headline_sensational_words', 0) * 3)
        
        # Deduct for headline-body mismatch (max 25 points)
        score -= min(25, features.get('headline_body_mismatch', 0) * 25)
        
        # Deduct for capitalization anomalies (max 20 points)
        caps_deduction = features.get('all_caps_word_ratio', 0) * 40
        score -= min(20, caps_deduction)
        score -= min(5, features.get('consecutive_caps_count', 0) * 2)
        
        # Deduct for punctuation manipulation (max 15 points)
        exclamation_deduction = features.get('multiple_exclamation_count', 0) * 1.5
        score -= min(15, exclamation_deduction)
        score -= min(8, features.get('mixed_punctuation_count', 0) * 3)
        
        # Deduct for excessive repetition (max 15 points)
        repetition_deduction = features.get('max_word_repetition_ratio', 0) * 20
        score -= min(15, repetition_deduction)
        
        # Deduct for lack of citations (5 points)
        if features.get('source_reference_count', 0) == 0:
            score -= 5
        
        return max(0, min(100, score))
    
    def _get_empty_features(self):
        """Return empty feature dictionary"""
        return {
            **self._get_empty_structure_features(),
            **self._get_empty_headline_features(),
            **self._get_empty_capitalization_features(),
            **self._get_empty_punctuation_features(),
            **self._get_empty_repetition_features(),
            **self._get_empty_citation_features()
        }
    
    def _get_empty_structure_features(self):
        return {
            'paragraph_count': 0, 'avg_paragraph_length': 0,
            'sentence_length_variance': 0, 'min_sentence_length': 0,
            'max_sentence_length': 0, 'very_short_sentence_ratio': 0,
            'very_long_sentence_ratio': 0
        }
    
    def _get_empty_headline_features(self):
        return {
            'clickbait_pattern_count': 0, 'is_clickbait': 0,
            'headline_has_number': 0, 'headline_has_question': 0,
            'headline_word_count': 0, 'headline_exclamation': 0,
            'headline_has_multiple_exclamation': 0, 'headline_all_caps_words': 0,
            'headline_sensational_words': 0, 'headline_body_similarity': 0.5,
            'headline_body_mismatch': 0.5, 'headline_keyword_overlap': 0.5
        }
    
    def _get_empty_capitalization_features(self):
        return {
            'all_caps_word_count': 0, 'all_caps_word_ratio': 0,
            'consecutive_caps_count': 0, 'mixed_case_word_ratio': 0,
            'mid_sentence_caps_ratio': 0
        }
    
    def _get_empty_punctuation_features(self):
        return {
            'multiple_exclamation_count': 0, 'max_consecutive_exclamation': 0,
            'multiple_question_count': 0, 'mixed_punctuation_count': 0,
            'ellipsis_count': 0, 'multiple_ellipsis': 0,
            'quote_imbalance': 0, 'dash_count': 0, 'hyphen_count': 0,
            'parentheses_count': 0, 'parentheses_imbalance': 0
        }
    
    def _get_empty_repetition_features(self):
        return {
            'max_word_repetition': 0, 'max_word_repetition_ratio': 0,
            'highly_repeated_word_count': 0, 'max_phrase_repetition': 0,
            'sentence_repetition_ratio': 0
        }
    
    def _get_empty_citation_features(self):
        return {
            'quote_count': 0, 'attribution_count': 0,
            'source_reference_count': 0, 'quote_density': 0
        }
    
    def get_feature_names(self):
        """Return list of all feature names"""
        return list(self._get_empty_features().keys())
