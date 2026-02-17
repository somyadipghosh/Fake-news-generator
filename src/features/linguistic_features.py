"""
Linguistic feature extraction module
Extracts lexical, syntactic, semantic, and stylistic features from text
"""
import re
import numpy as np
import nltk
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import textstat
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except:
    nlp = None


class LinguisticFeatureExtractor:
    """Extract comprehensive linguistic features from text"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def extract_all_features(self, text):
        """
        Extract all linguistic features
        Returns a dictionary of features
        """
        if not text or not isinstance(text, str):
            return self._get_empty_features()
        
        features = {}
        
        # Lexical features
        features.update(self._extract_lexical_features(text))
        
        # Syntactic features
        features.update(self._extract_syntactic_features(text))
        
        # Semantic features
        features.update(self._extract_semantic_features(text))
        
        # Stylistic features
        features.update(self._extract_stylistic_features(text))
        
        # Readability features
        features.update(self._extract_readability_features(text))
        
        return features
    
    def _extract_lexical_features(self, text):
        """Extract lexical features"""
        features = {}
        
        # Basic tokenization
        tokens = word_tokenize(text.lower())
        words = [t for t in tokens if t.isalnum()]
        
        # Character and word counts
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['unique_word_count'] = len(set(words))
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['lexical_diversity'] = len(set(words)) / len(words) if words else 0
        
        # Stop word ratio
        stop_word_count = sum(1 for w in words if w in self.stop_words)
        features['stop_word_ratio'] = stop_word_count / len(words) if words else 0
        
        # Sentence statistics
        sentences = sent_tokenize(text)
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = len(words) / len(sentences) if sentences else 0
        
        # Long word ratio (>6 chars)
        long_words = [w for w in words if len(w) > 6]
        features['long_word_ratio'] = len(long_words) / len(words) if words else 0
        
        # Short word ratio (<=3 chars)
        short_words = [w for w in words if len(w) <= 3]
        features['short_word_ratio'] = len(short_words) / len(words) if words else 0
        
        return features
    
    def _extract_syntactic_features(self, text):
        """Extract syntactic (POS) features"""
        features = {}
        
        tokens = word_tokenize(text)
        if not tokens:
            return self._get_empty_syntactic_features()
        
        # POS tagging
        pos_tags = pos_tag(tokens)
        pos_counts = Counter([tag for _, tag in pos_tags])
        total_tags = len(pos_tags)
        
        # Noun ratios
        noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
        features['noun_ratio'] = sum(pos_counts[tag] for tag in noun_tags) / total_tags if total_tags else 0
        
        # Verb ratios
        verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        features['verb_ratio'] = sum(pos_counts[tag] for tag in verb_tags) / total_tags if total_tags else 0
        
        # Adjective ratios
        adj_tags = ['JJ', 'JJR', 'JJS']
        features['adjective_ratio'] = sum(pos_counts[tag] for tag in adj_tags) / total_tags if total_tags else 0
        
        # Adverb ratios
        adv_tags = ['RB', 'RBR', 'RBS']
        features['adverb_ratio'] = sum(pos_counts[tag] for tag in adv_tags) / total_tags if total_tags else 0
        
        # Pronoun ratios
        pronoun_tags = ['PRP', 'PRP$', 'WP', 'WP$']
        features['pronoun_ratio'] = sum(pos_counts[tag] for tag in pronoun_tags) / total_tags if total_tags else 0
        
        # Determiner ratio
        features['determiner_ratio'] = pos_counts['DT'] / total_tags if total_tags else 0
        
        # Modal verb ratio
        features['modal_ratio'] = pos_counts['MD'] / total_tags if total_tags else 0
        
        # Punctuation features
        punct_count = sum(1 for token in tokens if not token.isalnum())
        features['punctuation_ratio'] = punct_count / len(tokens) if tokens else 0
        
        return features
    
    def _extract_semantic_features(self, text):
        """Extract semantic features using spacy if available"""
        features = {}
        
        if nlp is None:
            return self._get_empty_semantic_features()
        
        try:
            doc = nlp(text[:1000000])  # Limit text length for spacy
            
            # Named entity statistics
            entities = [ent for ent in doc.ents]
            features['entity_count'] = len(entities)
            features['entity_density'] = len(entities) / len(doc) if len(doc) > 0 else 0
            
            # Entity type diversity
            entity_types = set([ent.label_ for ent in entities])
            features['entity_type_diversity'] = len(entity_types)
            
            # Person, organization, location counts
            features['person_count'] = sum(1 for ent in entities if ent.label_ == 'PERSON')
            features['org_count'] = sum(1 for ent in entities if ent.label_ == 'ORG')
            features['location_count'] = sum(1 for ent in entities if ent.label_ in ['GPE', 'LOC'])
            
            # Dependency features
            features['avg_dependency_depth'] = np.mean([len(list(token.ancestors)) for token in doc]) if len(doc) > 0 else 0
            
        except Exception as e:
            features = self._get_empty_semantic_features()
        
        return features
    
    def _extract_stylistic_features(self, text):
        """Extract stylistic features"""
        features = {}
        
        # Capitalization features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['title_case_count'] = sum(1 for word in text.split() if word and word[0].isupper())
        
        # All caps words
        words = text.split()
        features['all_caps_word_ratio'] = sum(1 for w in words if w.isupper() and len(w) > 1) / len(words) if words else 0
        
        # Punctuation patterns
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['ellipsis_count'] = text.count('...') + text.count('…')
        features['quote_count'] = text.count('"') + text.count("'")
        
        # Multiple punctuation (!!!, ???, etc.)
        features['multiple_exclamation'] = len(re.findall(r'!{2,}', text))
        features['multiple_question'] = len(re.findall(r'\?{2,}', text))
        
        # Digits and numbers
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['number_count'] = len(re.findall(r'\d+', text))
        
        # Special characters
        features['special_char_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0
        
        # Hyphenation
        features['hyphen_count'] = text.count('-')
        
        return features
    
    def _extract_readability_features(self, text):
        """Extract readability scores"""
        features = {}
        
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            features['gunning_fog'] = textstat.gunning_fog(text)
            features['smog_index'] = textstat.smog_index(text)
            features['coleman_liau_index'] = textstat.coleman_liau_index(text)
            features['automated_readability_index'] = textstat.automated_readability_index(text)
            features['dale_chall_readability'] = textstat.dale_chall_readability_score(text)
            features['difficult_words'] = textstat.difficult_words(text)
            features['linsear_write_formula'] = textstat.linsear_write_formula(text)
        except:
            features = {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0,
                'smog_index': 0,
                'coleman_liau_index': 0,
                'automated_readability_index': 0,
                'dale_chall_readability': 0,
                'difficult_words': 0,
                'linsear_write_formula': 0
            }
        
        return features
    
    def _get_empty_features(self):
        """Return empty feature dictionary"""
        return {**self._get_empty_lexical_features(),
                **self._get_empty_syntactic_features(),
                **self._get_empty_semantic_features(),
                **self._get_empty_stylistic_features(),
                **self._get_empty_readability_features()}
    
    def _get_empty_lexical_features(self):
        return {
            'char_count': 0, 'word_count': 0, 'unique_word_count': 0,
            'avg_word_length': 0, 'lexical_diversity': 0, 'stop_word_ratio': 0,
            'sentence_count': 0, 'avg_sentence_length': 0, 'long_word_ratio': 0,
            'short_word_ratio': 0
        }
    
    def _get_empty_syntactic_features(self):
        return {
            'noun_ratio': 0, 'verb_ratio': 0, 'adjective_ratio': 0,
            'adverb_ratio': 0, 'pronoun_ratio': 0, 'determiner_ratio': 0,
            'modal_ratio': 0, 'punctuation_ratio': 0
        }
    
    def _get_empty_semantic_features(self):
        return {
            'entity_count': 0, 'entity_density': 0, 'entity_type_diversity': 0,
            'person_count': 0, 'org_count': 0, 'location_count': 0,
            'avg_dependency_depth': 0
        }
    
    def _get_empty_stylistic_features(self):
        return {
            'uppercase_ratio': 0, 'title_case_count': 0, 'all_caps_word_ratio': 0,
            'exclamation_count': 0, 'question_count': 0, 'ellipsis_count': 0,
            'quote_count': 0, 'multiple_exclamation': 0, 'multiple_question': 0,
            'digit_ratio': 0, 'number_count': 0, 'special_char_ratio': 0,
            'hyphen_count': 0
        }
    
    def _get_empty_readability_features(self):
        return {
            'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0,
            'gunning_fog': 0, 'smog_index': 0, 'coleman_liau_index': 0,
            'automated_readability_index': 0, 'dale_chall_readability': 0,
            'difficult_words': 0, 'linsear_write_formula': 0
        }
    
    def get_feature_names(self):
        """Return list of all feature names"""
        empty_features = self._get_empty_features()
        return list(empty_features.keys())
