"""Retrain the model with improved diverse dataset"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.training import FakeNewsTrainer
from config import SAVED_MODELS_DIR

print("="*70)
print("  RETRAINING WITH IMPROVED DATASET")
print("="*70)

# Load the improved dataset
print("\nLoading improved dataset...")
df = pd.read_csv('data/raw/improved_news.csv')
print(f"Total samples: {len(df)}")
print(f"Unique samples: {df.drop_duplicates().shape[0]}")
print(f"Class distribution:")
print(df['label'].value_counts())

# Initialize trainer
print("\nInitializing trainer...")
trainer = FakeNewsTrainer(data_df=df)

# Train the full pipeline
print("\nStarting training pipeline...")
trainer.train(save_model=True, perform_cv=False)

print("\n" + "="*70)
print("  RETRAINING COMPLETE!")
print("="*70)
print("\nNow test your article again with: python test_article.py")
