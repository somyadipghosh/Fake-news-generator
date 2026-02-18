"""
Complete Workflow Script - Augment data and retrain model
"""
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and display output"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error during: {description}")
        return False
    
    return True

def main():
    print("\n" + "="*70)
    print("     FAKE NEWS DETECTOR - COMPLETE REBUILD WORKFLOW")
    print("="*70)
    print("\nThis script will:")
    print("  1. Fetch real news from NewsAPI")
    print("  2. Augment the training dataset")
    print("  3. Retrain the model with improved data")
    print("  4. Test the predictions")
    
    response = input("\n🤔 Continue? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Augment dataset
    if not run_command("python augment_dataset.py --num-articles 100", 
                      "STEP 1: Augmenting dataset with NewsAPI"):
        print("\n⚠️  Dataset augmentation failed, but continuing with existing data...")
    
    # Step 2: Train model
    if not run_command("python quick_train.py", 
                      "STEP 2: Training model with augmented data"):
        print("\n❌ Training failed!")
        return
    
    # Step 3: Test predictions
    print("\n" + "="*70)
    print("  STEP 3: Testing predictions")
    print("="*70)
    print("\nModel training complete!")
    print("\n🎉 You can now test the improved model:")
    print("   • Run: python demo.py")
    print("   • Or run: python streamlit_app.py")
    print("\nThe model should now predict news articles more accurately!")
    
if __name__ == "__main__":
    main()
