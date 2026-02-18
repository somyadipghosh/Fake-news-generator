"""
Quick training script without cross-validation for faster demo
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import *
from src.training.trainer import FakeNewsTrainer

def main():
    print("\n" + "="*70)
    print("         FAKE NEWS DETECTION - QUICK TRAINING")
    print("="*70)
    
    # Check for augmented dataset first, then improved, then sample
    data_paths = [
        Path(DATA_DIR) / 'raw' / 'augmented_news.csv',
        Path(DATA_DIR) / 'raw' / 'improved_news.csv',
        Path(DATA_DIR) / 'raw' / 'sample_news.csv'
    ]
    
    data_path = None
    for path in data_paths:
        if path.exists():
            data_path = path
            break
    
    if not data_path:
        print("\n❌ No dataset found.")
        print("\n💡 Run this command first to create a dataset:")
        print("   python augment_dataset.py")
        return
    
    print(f"\n📊 Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset size: {len(df)}")
    print(f"Real news: {len(df[df['label']==0])}, Fake news: {len(df[df['label']==1])}")
    
    # Initialize trainer
    print("\n🔧 Initializing trainer...")
    trainer = FakeNewsTrainer(data_df=df)
    
    # Train without cross-validation for speed
    print("\n🚀 Training model (without cross-validation for speed)...")
    print("This will take 2-5 minutes...\n")
    
    model = trainer.train(
        save_model=True,
        perform_cv=False  # Skip cross-validation for speed
    )
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved to: {Path(MODELS_DIR) / 'saved_models'}")
    print("\nYou can now run: python demo.py")

if __name__ == "__main__":
    main()
