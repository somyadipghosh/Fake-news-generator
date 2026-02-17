"""
Model evaluation and metrics
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve,
                            confusion_matrix, classification_report)


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    def __init__(self, model):
        """
        Initialize evaluator
        
        Args:
            model: Trained model instance
        """
        self.model = model
    
    def evaluate(self, X, y, texts=None, headlines=None):
        """
        Comprehensive evaluation
        
        Args:
            X: Test features or texts
            y: True labels
            texts: Original texts (optional)
            headlines: Headlines (optional)
        
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        if texts is not None:
            y_pred = self.model.predict(texts, headlines)
            y_proba = self.model.predict_proba(texts, headlines).ravel()
        else:
            y_pred = self.model.predict(X)
            y_proba = self.model.predict_proba(X).ravel()
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Per-class metrics
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def print_evaluation(self, X, y, texts=None, headlines=None):
        """Print evaluation metrics"""
        metrics = self.evaluate(X, y, texts, headlines)
        
        print("="*50)
        print("Model Evaluation Results")
        print("="*50)
        
        print(f"\nAccuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1 Score:    {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("              Real   Fake")
        print(f"Actual Real   {metrics['true_negatives']:4d}   {metrics['false_positives']:4d}")
        print(f"       Fake   {metrics['false_negatives']:4d}   {metrics['true_positives']:4d}")
        
        return metrics
    
    def plot_confusion_matrix(self, X, y, texts=None, headlines=None, 
                            save_path=None):
        """Plot confusion matrix"""
        metrics = self.evaluate(X, y, texts, headlines)
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, X, y, texts=None, headlines=None, save_path=None):
        """Plot ROC curve"""
        # Get probabilities
        if texts is not None:
            y_proba = self.model.predict_proba(texts, headlines).ravel()
        else:
            y_proba = self.model.predict_proba(X).ravel()
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if not hasattr(self.model, 'neural_network'):
            print("No training history available")
            return
        
        history = self.model.neural_network.history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1.plot(history['loss'], linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(alpha=0.3)
        
        # Accuracy
        ax2.plot(history['accuracy'], linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()
