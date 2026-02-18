"""Test structural integrity score calculation"""
from src.features.structural_features import StructuralFeatureExtractor

# Sample article
text = """
Breaking news today shows that something incredible happened. 
This is an amazing story that will shock you. Scientists made a discovery 
that changes everything we know about the world. According to sources,
this finding is unprecedented in history.

The research team, led by Dr. Smith, stated that their findings 
are groundbreaking. Multiple experts have confirmed these results.
"""

headline = "SHOCKING Discovery That Will Change Everything!"

# Extract features
extractor = StructuralFeatureExtractor()
features = extractor.extract_all_features(text, headline)

print("=" * 60)
print("STRUCTURAL FEATURES")
print("=" * 60)
for key, value in sorted(features.items()):
    print(f"{key:40s}: {value}")

print("\n" + "=" * 60)
print("SCORE CALCULATION")
print("=" * 60)

# Calculate score step by step
score = 100
print(f"Starting score:                           {score}")

deduct = features.get('clickbait_pattern_count', 0) * 10
score -= deduct
print(f"After clickbait deduction ({deduct}):            {score}")

deduct = features.get('headline_sensational_words', 0) * 5
score -= deduct
print(f"After sensational words ({deduct}):              {score}")

deduct = features.get('headline_body_mismatch', 0) * 30
score -= deduct
print(f"After headline mismatch ({deduct:.1f}):         {score:.1f}")

deduct = features.get('all_caps_word_ratio', 0) * 50
score -= deduct
print(f"After caps ratio ({deduct:.2f}):                 {score:.2f}")

deduct = features.get('consecutive_caps_count', 0) * 5
score -= deduct
print(f"After consecutive caps ({deduct}):              {score:.2f}")

deduct = features.get('multiple_exclamation_count', 0) * 3
score -= deduct
print(f"After exclamation ({deduct}):                   {score:.2f}")

deduct = features.get('mixed_punctuation_count', 0) * 5
score -= deduct
print(f"After mixed punctuation ({deduct}):             {score:.2f}")

deduct = features.get('max_word_repetition_ratio', 0) * 30
score -= deduct
print(f"After repetition ({deduct:.2f}):                 {score:.2f}")

if features.get('source_reference_count', 0) == 0:
    deduct = 10
else:
    deduct = 0
score -= deduct
print(f"After source check ({deduct}):                  {score:.2f}")

score = max(0, min(100, score))
print(f"Final score (clamped):                     {score:.2f}")

print("\n" + "=" * 60)
print("COMPUTED SCORE FROM METHOD")
print("=" * 60)
computed_score = extractor.compute_structural_integrity_score(features)
print(f"compute_structural_integrity_score result: {computed_score:.2f}")
