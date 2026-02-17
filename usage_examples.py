"""
Quick start guide and usage examples
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def example_1_basic_detection():
    """Example 1: Basic fake news detection"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Fake News Detection")
    print("="*70)
    
    from src.detector import detect_fake_news
    
    article = """
    A new study from researchers at MIT has shown that artificial 
    intelligence can help diagnose diseases more accurately than 
    traditional methods. The peer-reviewed research, published in 
    Nature Medicine, analyzed data from over 50,000 patients.
    """
    
    headline = "MIT Study Shows AI Improves Disease Diagnosis"
    
    result = detect_fake_news(article, headline)
    
    print(f"\nResult: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")


def example_2_batch_detection():
    """Example 2: Batch detection"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Detection")
    print("="*70)
    
    from src.detector import FakeNewsDetector
    
    detector = FakeNewsDetector()
    
    articles = [
        "Scientists discover new treatment for common disease...",
        "SHOCKING! You won't believe this MIRACLE cure!!!",
        "Local school wins regional competition..."
    ]
    
    headlines = [
        "New Medical Treatment Discovered",
        "DOCTORS HATE THIS TRICK!!!",
        "School Wins Competition"
    ]
    
    results = detector.detect_batch(articles, headlines, explain=False)
    
    print(f"\nProcessed {len(results)} articles")


def example_3_detailed_analysis():
    """Example 3: Detailed analysis with explanation"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Detailed Analysis")
    print("="*70)
    
    from src.detector import FakeNewsDetector
    
    detector = FakeNewsDetector()
    
    article = """
    URGENT!!! Everyone needs to know this SHOCKING truth that 
    the government is hiding! Scientists have discovered that 
    drinking water backwards can cure ALL diseases instantly! 
    You WON'T BELIEVE what happens next! Share before they 
    delete this! BREAKING NEWS!!!
    """
    
    # Get detailed explanation
    explainer = detector.get_explainer()
    explanation = explainer.print_explanation(article)


def example_4_training_custom_model():
    """Example 4: Training a custom model"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Training Custom Model")
    print("="*70)
    
    from src.training import FakeNewsTrainer
    import pandas as pd
    
    # Prepare your data
    data = pd.DataFrame({
        'text': ['article text 1...', 'article text 2...'],
        'headline': ['headline 1', 'headline 2'],
        'label': [0, 1]  # 0 = real, 1 = fake
    })
    
    # Train
    trainer = FakeNewsTrainer(data_df=data)
    model = trainer.train(save_model=True, perform_cv=False)
    
    print("\nModel trained!")


def example_5_feature_extraction():
    """Example 5: Extract features manually"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Manual Feature Extraction")
    print("="*70)
    
    from src.features import (LinguisticFeatureExtractor, 
                              PsychologicalFeatureExtractor,
                              StructuralFeatureExtractor,
                              CoherenceFeatureExtractor)
    
    text = "Sample article text here..."
    headline = "Sample headline"
    
    # Extract different feature types
    ling_ext = LinguisticFeatureExtractor()
    psych_ext = PsychologicalFeatureExtractor()
    struct_ext = StructuralFeatureExtractor()
    coher_ext = CoherenceFeatureExtractor()
    
    ling_features = ling_ext.extract_all_features(text)
    psych_features = psych_ext.extract_all_features(text)
    struct_features = struct_ext.extract_all_features(text, headline)
    coher_features = coher_ext.extract_all_features(text)
    
    print(f"\nExtracted {len(ling_features)} linguistic features")
    print(f"Extracted {len(psych_features)} psychological features")
    print(f"Extracted {len(struct_features)} structural features")
    print(f"Extracted {len(coher_features)} coherence features")
    
    # Compute scores
    psych_score = psych_ext.compute_psychological_score(psych_features)
    struct_score = struct_ext.compute_structural_integrity_score(struct_features)
    coher_score = coher_ext.compute_coherence_score(coher_features)
    
    print(f"\nPsychological Manipulation Score: {psych_score:.1f}/100")
    print(f"Structural Integrity Score: {struct_score:.1f}/100")
    print(f"Coherence Score: {coher_score:.1f}/100")


def example_6_custom_word2vec():
    """Example 6: Train custom Word2Vec embeddings"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Word2Vec Training")
    print("="*70)
    
    from src.embeddings import CustomWord2Vec
    from nltk import word_tokenize
    
    # Prepare corpus
    texts = [
        "This is a sample sentence.",
        "Another example text here.",
        "Training word embeddings from scratch."
    ]
    
    sentences = [word_tokenize(text.lower()) for text in texts]
    
    # Train Word2Vec
    w2v = CustomWord2Vec(vector_size=100, epochs=10)
    w2v.train(sentences)
    
    # Use embeddings
    word_vector = w2v.get_vector("sample")
    sentence_vector = w2v.get_sentence_vector("This is a test")
    
    print(f"\nWord vector shape: {word_vector.shape}")
    print(f"Sentence vector shape: {sentence_vector.shape}")
    
    # Find similar words
    similar = w2v.most_similar("sample", topn=3)
    print(f"\nWords similar to 'sample': {similar}")


def example_7_model_evaluation():
    """Example 7: Evaluate model performance"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Model Evaluation")
    print("="*70)
    
    from src.training import ModelEvaluator
    from src.detector import FakeNewsDetector
    import numpy as np
    
    # Load model
    detector = FakeNewsDetector()
    
    # Prepare test data
    test_texts = ["article 1...", "article 2..."]
    test_labels = np.array([0, 1])
    
    # Evaluate
    evaluator = ModelEvaluator(detector.model)
    metrics = evaluator.print_evaluation(test_texts, test_labels)
    
    # Plot visualizations
    # evaluator.plot_confusion_matrix(test_texts, test_labels, save_path='confusion_matrix.png')
    # evaluator.plot_roc_curve(test_texts, test_labels, save_path='roc_curve.png')


def print_usage_guide():
    """Print comprehensive usage guide"""
    print("\n" + "="*70)
    print(" "*20 + "FAKE NEWS DETECTION SYSTEM")
    print(" "*25 + "USAGE GUIDE")
    print("="*70)
    
    print("\n📚 QUICK START:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Download NLTK data: python setup_nltk.py")
    print("   3. Download spaCy model: python -m spacy download en_core_web_sm")
    print("   4. Train a model: python train_example.py")
    print("   5. Run demo: python demo.py")
    
    print("\n🔧 BASIC USAGE:")
    print("   from src.detector import detect_fake_news")
    print("   result = detect_fake_news(article_text, headline)")
    print("   print(result['prediction'], result['confidence'])")
    
    print("\n📊 AVAILABLE FEATURES:")
    print("   ✓ Deep linguistic analysis (lexical, syntactic, semantic)")
    print("   ✓ Psychological manipulation detection")
    print("   ✓ Structural anomaly detection")
    print("   ✓ Coherence and logical flow analysis")
    print("   ✓ Custom Word2Vec embeddings")
    print("   ✓ Hybrid neural network classifier")
    print("   ✓ Full explainability and feature importance")
    
    print("\n📁 PROJECT STRUCTURE:")
    print("   config.py              - Configuration settings")
    print("   src/features/          - Feature extraction modules")
    print("   src/embeddings/        - Custom Word2Vec implementation")
    print("   src/models/            - Neural network and hybrid model")
    print("   src/training/          - Training and evaluation")
    print("   src/explainability/    - Model explanation")
    print("   src/detector.py        - Main detection interface")
    
    print("\n🎯 OUTPUT SCORES:")
    print("   • Prediction: FAKE or REAL")
    print("   • Confidence: 0-100%")
    print("   • Psychological Manipulation Score: 0-100")
    print("   • Structural Integrity Score: 0-100")
    print("   • Sentiment Volatility Index: 0-100")
    print("   • Coherence Score: 0-100")
    
    print("\n💡 EXAMPLES AVAILABLE:")
    print("   Run this script with example number:")
    print("   python usage_examples.py 1  # Basic detection")
    print("   python usage_examples.py 2  # Batch detection")
    print("   python usage_examples.py 3  # Detailed analysis")
    print("   python usage_examples.py 4  # Training custom model")
    print("   python usage_examples.py 5  # Feature extraction")
    print("   python usage_examples.py 6  # Word2Vec training")
    print("   python usage_examples.py 7  # Model evaluation")
    
    print("\n📖 DOCUMENTATION:")
    print("   See README.md for detailed documentation")
    print("   Check source code for API reference")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main function to run examples"""
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        examples = {
            '1': example_1_basic_detection,
            '2': example_2_batch_detection,
            '3': example_3_detailed_analysis,
            '4': example_4_training_custom_model,
            '5': example_5_feature_extraction,
            '6': example_6_custom_word2vec,
            '7': example_7_model_evaluation
        }
        
        if example_num in examples:
            try:
                examples[example_num]()
            except Exception as e:
                print(f"\n❌ Error running example: {e}")
                print("Make sure you have trained a model first (run train_example.py)")
        else:
            print(f"Invalid example number: {example_num}")
            print_usage_guide()
    else:
        print_usage_guide()


if __name__ == "__main__":
    main()
