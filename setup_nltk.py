"""
Download required NLTK data
"""
import nltk

print("Downloading NLTK data...")

# Download required NLTK resources
resources = [
    'punkt',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words',
    'wordnet',
    'stopwords',
    'vader_lexicon',
    'brown',
    'opinion_lexicon'
]

for resource in resources:
    try:
        nltk.download(resource, quiet=True)
        print(f"✓ {resource}")
    except Exception as e:
        print(f"✗ {resource}: {e}")

print("\nNLTK setup complete!")
