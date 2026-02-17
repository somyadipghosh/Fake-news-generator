"""
Example script for training the fake news detection model
"""
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training import FakeNewsTrainer
from config import RAW_DATA_DIR


def create_sample_dataset(save_path):
    """
    Create a sample dataset for demonstration purposes
    In practice, you should use a real dataset like:
    - LIAR dataset
    - Fake News Challenge dataset
    - ISOT Fake News Dataset
    """
    print("Creating sample dataset for demonstration...")
    
    # Sample data (replace with real dataset)
    real_news = [
        {
            'headline': 'Study Shows Benefits of Exercise for Heart Health',
            'text': 'A comprehensive study published in the Journal of Medicine has found that regular cardiovascular exercise significantly reduces the risk of heart disease. Researchers from Harvard Medical School followed 10,000 participants over five years and observed that those who exercised for at least 30 minutes daily had a 25% lower risk of cardiac events. Dr. John Smith, lead researcher, emphasized the importance of maintaining a consistent exercise routine.',
            'label': 0
        },
        {
            'headline': 'New Technology Improves Solar Panel Efficiency',
            'text': 'Engineers at Stanford University have developed a new coating for solar panels that increases their efficiency by 15%. The breakthrough, published in Nature Energy, could make solar power more economically viable for households and businesses. The team tested the technology over twelve months and found consistent improvements in energy capture across different weather conditions.',
            'label': 0
        },
        {
            'headline': 'Local School Wins State Science Competition',
            'text': 'Lincoln High School students won first place in the state science fair with their project on water purification. The team of four students developed an affordable filtration system using natural materials. Their teacher, Ms. Johnson, praised their hard work and dedication. The students will represent the state at the national competition in Washington next month.',
            'label': 0
        }
    ]
    
    fake_news = [
        {
            'headline': 'SHOCKING: Scientists Discover Eating Chocolate Cures Cancer!!!',
            'text': 'AMAZING discovery that doctors don\'t want you to know!!! Eating chocolate INSTANTLY cures ALL types of cancer according to secret research! Everyone is talking about this MIRACLE cure that Big Pharma has been hiding for years! You WON\'T BELIEVE what happens next! SHARE before they DELETE this! URGENT!!!',
            'label': 1
        },
        {
            'headline': 'BREAKING: Aliens Control Government Officials!',
            'text': 'Shocking evidence reveals that ALIENS have been controlling world leaders for decades! Anonymous sources confirm that reptilian beings are secretly running the government! This EXPLOSIVE information is being censored everywhere! Wake up people! The TRUTH is finally out! Share this before it disappears!!!',
            'label': 1
        },
        {
            'headline': 'You Won\'t Believe This One Weird Trick!',
            'text': 'Doctors HATE him! This man discovered ONE WEIRD TRICK that makes you lose 50 pounds in 2 days! No exercise, no dieting! Just this SIMPLE trick! Click NOW before Big Pharma removes this! AMAZING results that will SHOCK you! Everyone is doing it! Don\'t miss out!!!',
            'label': 1
        }
    ]
    
    # Create balanced dataset
    samples = []
    
    # Add multiple copies to make dataset larger
    for _ in range(100):
        samples.extend(real_news)
        samples.extend(fake_news)
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Save to file
    df.to_csv(save_path, index=False)
    print(f"Sample dataset created: {save_path}")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    return df


def main():
    print("\n" + "="*70)
    print(" "*15 + "FAKE NEWS DETECTION - TRAINING EXAMPLE")
    print("="*70)
    
    # Prepare dataset
    data_path = os.path.join(RAW_DATA_DIR, 'sample_news.csv')
    
    print("\n📊 STEP 1: Preparing Dataset")
    print("-"*70)
    
    if not os.path.exists(data_path):
        print("\n⚠️  No dataset found. Creating sample dataset...")
        print("\n⚠️  NOTE: This is a SMALL SAMPLE dataset for demonstration only!")
        print("    For real use, please use a proper dataset like:")
        print("    - LIAR dataset (https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)")
        print("    - ISOT Fake News Dataset")
        print("    - Fake News Challenge Dataset")
        print()
        
        df = create_sample_dataset(data_path)
    else:
        print(f"Loading existing dataset from {data_path}")
        df = pd.read_csv(data_path)
    
    # Initialize trainer
    print("\n🔧 STEP 2: Initializing Trainer")
    print("-"*70)
    
    trainer = FakeNewsTrainer(data_path=data_path)
    
    # Train model
    print("\n🚀 STEP 3: Training Model")
    print("-"*70)
    print("\nThis may take some time depending on your dataset size...")
    print("Training includes:")
    print("  1. Data preprocessing")
    print("  2. Word2Vec embedding training")
    print("  3. Feature extraction")
    print("  4. Neural network training")
    print("  5. Model evaluation")
    print()
    
    # Ask user about cross-validation
    perform_cv = input("Perform cross-validation? (y/n, default: n): ").strip().lower() == 'y'
    
    # Train
    model = trainer.train(save_model=True, perform_cv=perform_cv)
    
    # Summary
    print("\n" + "="*70)
    print(" "*25 + "TRAINING COMPLETE!")
    print("="*70)
    
    print("\n✓ Model trained and saved successfully!")
    print("\n📁 Model files saved in: models/saved_models/")
    print("   - hybrid_model.pkl (main model)")
    print("   - hybrid_model_nn.pkl (neural network)")
    print("   - word2vec_model.pkl (word embeddings)")
    
    print("\n🎯 Next Steps:")
    print("   1. Run demo.py to test the model")
    print("   2. Use src.detector.FakeNewsDetector in your code")
    print("   3. Check results/ folder for evaluation plots")
    
    print("\n💡 Example Usage:")
    print("   from src.detector import FakeNewsDetector")
    print("   detector = FakeNewsDetector()")
    print("   result = detector.detect(article_text, headline)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        print("You can resume training by running this script again.\n")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("\nPlease check:")
        print("  1. All dependencies are installed (pip install -r requirements.txt)")
        print("  2. NLTK data is downloaded (python setup_nltk.py)")
        print("  3. Your dataset format is correct (text, headline, label columns)")
        print()
        raise
