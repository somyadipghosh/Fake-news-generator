"""
Demo script showing how to use the Fake News Detection System
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detector import FakeNewsDetector

# Sample articles for demonstration
SAMPLE_REAL_NEWS = """
Scientists at the Massachusetts Institute of Technology have developed a new 
artificial intelligence system that can detect patterns in medical imaging 
data with unprecedented accuracy. The research, published in Nature Medicine, 
shows that the system achieved 95% accuracy in identifying early signs of 
disease in CT scans. Dr. Sarah Johnson, lead researcher on the project, 
explained that the system was trained on over 100,000 medical images from 
multiple hospitals. The team hopes this technology will help radiologists 
catch diseases earlier, potentially saving thousands of lives. The research 
was funded by the National Institutes of Health and underwent rigorous peer 
review before publication.
"""

SAMPLE_FAKE_NEWS = """
SHOCKING!!! Scientists REVEAL that drinking coffee backwards can INSTANTLY 
cure ALL diseases!!! You WON'T BELIEVE what happens next!!! Doctors HATE this 
one simple trick that Big Pharma doesn't want you to know!!! Everyone is talking 
about this MIRACLE cure that has been hidden from the public for years!!! This 
AMAZING discovery will change your life FOREVER!!! Click here NOW before it's 
too late!!! URGENT!!! Share this with EVERYONE you know before they DELETE it!!!
"""

SAMPLE_REAL_HEADLINE = "MIT Researchers Develop AI System for Early Disease Detection"
SAMPLE_FAKE_HEADLINE = "SHOCKING Miracle Cure Discovered! Doctors Hate This Trick!"


def main():
    print("\n" + "="*70)
    print(" "*15 + "FAKE NEWS DETECTION SYSTEM - DEMO")
    print("="*70)
    
    print("\nThis demo will demonstrate the fake news detection system.")
    print("Note: You need to train a model first using train_example.py")
    print("\nChecking for trained models...")
    
    # Initialize detector
    try:
        detector = FakeNewsDetector()
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        print("\n💡 Tip: Run train_example.py first to train a model")
        print("   Or place your trained models in the models/saved_models/ directory")
        return
    
    # Example 1: Detect real news
    print("\n" + "="*70)
    print("EXAMPLE 1: Analyzing a REAL news article")
    print("="*70)
    
    result1 = detector.detect(
        text=SAMPLE_REAL_NEWS,
        headline=SAMPLE_REAL_HEADLINE,
        explain=True,
        verbose=True
    )
    
    input("\nPress Enter to continue to the next example...")
    
    # Example 2: Detect fake news
    print("\n" + "="*70)
    print("EXAMPLE 2: Analyzing a FAKE news article")
    print("="*70)
    
    result2 = detector.detect(
        text=SAMPLE_FAKE_NEWS,
        headline=SAMPLE_FAKE_HEADLINE,
        explain=True,
        verbose=True
    )
    
    # Example 3: Interactive mode
    print("\n" + "="*70)
    print("EXAMPLE 3: Interactive Mode")
    print("="*70)
    print("\nEnter your own article to analyze (or press Ctrl+C to exit)")
    
    try:
        while True:
            print("\n" + "-"*70)
            headline = input("\nEnter headline (or press Enter to skip): ").strip()
            print("\nEnter article text (end with an empty line):")
            
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            
            text = '\n'.join(lines)
            
            if not text:
                print("No text entered. Try again.")
                continue
            
            # Analyze
            detector.detect(
                text=text,
                headline=headline if headline else None,
                explain=True,
                verbose=True
            )
            
            continue_prompt = input("\nAnalyze another article? (y/n): ").strip().lower()
            if continue_prompt != 'y':
                break
    
    except KeyboardInterrupt:
        print("\n\nExiting interactive mode...")
    
    print("\n" + "="*70)
    print(" "*25 + "DEMO COMPLETE")
    print("="*70)
    print("\nThank you for using the Fake News Detection System!")
    print("\nFor more information, check out:")
    print("  - README.md for usage instructions")
    print("  - train_example.py for training your own model")
    print("  - src/detector.py for the API reference")
    print()


if __name__ == "__main__":
    main()
