"""Test structural integrity score with NO headline"""
from src.features.structural_features import StructuralFeatureExtractor

# Sample article WITHOUT headline
text = """
This is a simple article with no unusual features.
It contains multiple sentences that flow naturally.
There are no excessive capital letters or punctuation marks.
The content is written in a professional manner.
Sources indicate that this is credible information.
"""

headline = None

# Extract features
extractor = StructuralFeatureExtractor()
features = extractor.extract_all_features(text, headline)

print("=" * 60)
print("STRUCTURAL FEATURES (NO HEADLINE)")
print("=" * 60)
for key, value in sorted(features.items()):
    print(f"{key:40s}: {value}")

print("\n" + "=" * 60)
print("COMPUTED SCORE FROM METHOD")
print("=" * 60)
computed_score = extractor.compute_structural_integrity_score(features)
print(f"compute_structural_integrity_score result: {computed_score:.2f}")

print("\n" + "=" * 60)
print("Now testing with EMPTY TEXT")
print("=" * 60)
features2 = extractor.extract_all_features("", "Test Headline")
score2 = extractor.compute_structural_integrity_score(features2)
print(f"Score with empty text: {score2:.2f}")

print("\n" + "=" * 60)
print("Now testing with article that has MANY issues")
print("=" * 60)

bad_text = """
YOU WON'T BELIEVE THIS!!!!! SHOCKING NEWS!!!!!

BREAKING: Something AMAZING happened!!!! This is INCREDIBLE!!!!!
WOW!!! WOW!!! WOW!!! WOW!!! WOW!!! WOW!!! WOW!!! WOW!!!

This is absolutely UNBELIEVABLE!!!!! This This This This This This
The most SHOCKING thing ever!!!! MUST SEE!!!!

DOCTORS HATE HIM!!!! YOU WON'T BELIEVE WHAT HAPPENED NEXT!!!!
"""

bad_headline = "10 SHOCKING Reasons Why You WON'T BELIEVE What Happened Next!!!"

features3 = extractor.extract_all_features(bad_text, bad_headline)
score3 = extractor.compute_structural_integrity_score(features3)

print(f"Score with problematic text: {score3:.2f}")
print("\nKey problematic features:")
print(f"  clickbait_pattern_count: {features3.get('clickbait_pattern_count', 0)}")
print(f"  headline_sensational_words: {features3.get('headline_sensational_words', 0)}")
print(f"  headline_body_mismatch: {features3.get('headline_body_mismatch', 0):.3f}")
print(f"  all_caps_word_ratio: {features3.get('all_caps_word_ratio', 0):.3f}")
print(f"  multiple_exclamation_count: {features3.get('multiple_exclamation_count', 0)}")
print(f"  max_word_repetition_ratio: {features3.get('max_word_repetition_ratio', 0):.3f}")
