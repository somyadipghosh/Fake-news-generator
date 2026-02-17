"""Comprehensive test of various news types"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detector import FakeNewsDetector

print("="*70)
print("  COMPREHENSIVE NEWS DETECTION TEST")
print("="*70)

detector = FakeNewsDetector()

test_cases = [
    {
        "headline": "Study Shows Regular Exercise Improves Heart Health",
        "text": "A new peer-reviewed study published in the Journal of the American Medical Association found that regular cardiovascular exercise can significantly reduce the risk of heart disease. Researchers followed 5,000 participants over 10 years.",
        "expected": "REAL"
    },
    {
        "headline": "SHOCKING: Drinking Lemon Water Cures All Diseases!!!",
        "text": "AMAZING discovery! Drinking lemon water INSTANTLY cures cancer, diabetes, and all diseases! Doctors don't want you to know this SECRET! Share before they delete it! URGENT!!!",
        "expected": "FAKE"
    },
    {
        "headline": "Scientists Confirm Eating Mangoes After 8 PM Causes Memory Loss",
        "text": "In a shocking discovery, researchers at the International Tropical Nutrition Institute revealed that eating mangoes after 8 PM can permanently damage memory cells. According to the lead scientist, late-night mango consumption disrupts brain waves and leads to short-term amnesia.",
        "expected": "FAKE"
    },
    {
        "headline": "New Solar Panel Technology Increases Efficiency by 15%",
        "text": "Engineers at MIT have developed a new coating for solar panels that increases their conversion efficiency by 15%. The breakthrough was published in Nature Energy and has been tested over 12 months in various weather conditions.",
        "expected": "REAL"
    },
    {
        "headline": "Clapping Your Hands Cures Heart Disease Instantly",
        "text": "Medical researchers have discovered that clapping your hands vigorously for five minutes daily can completely cure heart disease. The vibrations created by clapping apparently clear arterial blockages. One patient avoided bypass surgery using only this method.",
        "expected": "FAKE"
    }
]

print("\nTesting {} articles...\n".format(len(test_cases)))

correct = 0
for i, case in enumerate(test_cases, 1):
    result = detector.detect(case["text"], case["headline"], explain=False, verbose=False)
    
    status = "✓" if result['prediction'] == case['expected'] else "✗"
    correct += (result['prediction'] == case['expected'])
    
    print(f"{status} Test {i}: {case['headline'][:60]}...")
    print(f"   Expected: {case['expected']:5s} | Got: {result['prediction']:5s} | Confidence: {result['confidence']:.1%}")
    print()

print("="*70)
print(f"Results: {correct}/{len(test_cases)} correct ({correct/len(test_cases)*100:.1f}%)")
print("="*70)
