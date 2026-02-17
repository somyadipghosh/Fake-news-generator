"""Debug script to check model predictions"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detector import FakeNewsDetector
import pandas as pd

print("Loading dataset...")
df = pd.read_csv('data/raw/sample_news.csv')

print("Loading detector...")
detector = FakeNewsDetector()

print("\n" + "="*70)
print("Testing Model Predictions")
print("="*70)

# Test a few clearly fake articles
fake_examples = df[df['label'] == 1].head(3)
print("\n### FAKE ARTICLES (should have high probability) ###\n")
for idx, row in fake_examples.iterrows():
    result = detector.detect(row['text'], row['headline'], explain=False, verbose=False)
    print(f"Headline: {row['headline']}")
    print(f"  True Label: FAKE (1)")
    print(f"  Predicted: {result['prediction']}")
    print(f"  Fake Probability: {result['fake_probability']:.4f}")
    print(f"  Real Probability: {result['real_probability']:.4f}")
    print()

# Test a few clearly real articles
real_examples = df[df['label'] == 0].head(3)
print("\n### REAL ARTICLES (should have low fake probability) ###\n")
for idx, row in real_examples.iterrows():
    result = detector.detect(row['text'], row['headline'], explain=False, verbose=False)
    print(f"Headline: {row['headline']}")
    print(f"  True Label: REAL (0)")
    print(f"  Predicted: {result['prediction']}")
    print(f"  Fake Probability: {result['fake_probability']:.4f}")
    print(f"  Real Probability: {result['real_probability']:.4f}")
    print()

# Test the user's article from test_article.py
print("\n### USER'S TEST ARTICLE ###\n")
article = """In a shocking discovery, researchers at the International Tropical Nutrition Institute revealed that eating mangoes after 8 PM can permanently damage memory cells. According to the lead scientist, late-night mango consumption disrupts brain waves and leads to short-term amnesia. Several participants reportedly forgot their own names after consuming mangoes past sunset."""
headline = "Scientists Confirm Eating Mangoes After 8 PM Causes Memory Loss"

result = detector.detect(article, headline, explain=False, verbose=False)
print(f"Headline: {headline}")
print(f"  Predicted: {result['prediction']}")
print(f"  Fake Probability: {result['fake_probability']:.4f}")
print(f"  Real Probability: {result['real_probability']:.4f}")
print()
