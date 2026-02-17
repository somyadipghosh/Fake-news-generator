"""
Test script to verify all imports work correctly
"""
import sys
print("Python:", sys.executable)
print("Python version:", sys.version)
print()

try:
    import nltk
    print("✓ nltk imported successfully")
    print(f"  Version: {nltk.__version__}")
    print(f"  Path: {nltk.__file__}")
except ImportError as e:
    print(f"✗ Failed to import nltk: {e}")

try:
    from nltk import word_tokenize
    print("✓ word_tokenize imported successfully")
except ImportError as e:
    print(f"✗ Failed to import word_tokenize: {e}")

try:
    import spacy
    print("✓ spacy imported successfully")
    print(f"  Version: {spacy.__version__}")
except ImportError as e:
    print(f"✗ Failed to import spacy: {e}")

try:
    import numpy
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import numpy: {e}")

try:
    import pandas
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pandas: {e}")

try:
    import streamlit
    print("✓ streamlit imported successfully")
    print(f"  Version: {streamlit.__version__}")
except ImportError as e:
    print(f"✗ Failed to import streamlit: {e}")

try:
    import sklearn
    print("✓ sklearn imported successfully")
except ImportError as e:
    print(f"✗ Failed to import sklearn: {e}")

print("\n" + "="*50)
print("Testing project imports...")
print("="*50)

try:
    from src.detector import FakeNewsDetector
    print("✓ FakeNewsDetector imported successfully")
except Exception as e:
    print(f"✗ Failed to import FakeNewsDetector: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests complete!")
