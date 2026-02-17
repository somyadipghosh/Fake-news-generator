"""Quick test of user's article"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detector import FakeNewsDetector

# User's article
article = """NASA successfully launched the James Webb Space Telescope on December 25, 2021, aboard an Ariane 5 rocket from French Guiana. The telescope is designed to study the early universe, distant galaxies, and exoplanets using advanced infrared imaging technology

"""

headline = "NASA’s James Webb Space Telescope Launch"

print("Analyzing your article...\n")

detector = FakeNewsDetector()
result = detector.detect(
    text=article,
    headline=headline,
    explain=True,
    verbose=True
)

print("\n" + "="*70)
print("Analysis Complete!")
print("="*70)
