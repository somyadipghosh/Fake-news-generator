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
    
    # Check for existing dataset
    data_path = DATA_DIR / 'raw' / 'sample_news.csv'
    
    if not data_path.exists():
        print("\n❌ No dataset found. Please run train_example.py first to create sample data.")
        return
    
    print(f"\n📊 Loading dataset from {data_path}")
    df = pd.read_data(data_path)
    print(f"Dataset size: {len(df)}")
    
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
    print(f"Model saved to: {MODELS_DIR / 'saved_models'}")
    print("\nYou can now run: python demo.py")

if __name__ == "__main__":
    main()
