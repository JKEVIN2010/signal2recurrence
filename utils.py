"""
Utility functions for signal analysis and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from typing import Optional, Tuple


def plot_training_history(history, save_path: Optional[str] = None):
    """
    Plot training and validation loss.
    
    Parameters
    ----------
    history : keras.callbacks.History
        Training history object
    save_path : Optional[str]
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    label_names: Optional[dict] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Visualize high-dimensional embeddings in 2D.
    
    Parameters
    ----------
    embeddings : np.ndarray
        High-dimensional embeddings
    labels : np.ndarray
        Class labels
    method : str
        Dimensionality reduction method: 'tsne' or 'pca'
    label_names : Optional[dict]
        Mapping from labels to names
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save the plot
    """
    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names[label] if label_names else f"Class {label}"
        
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=label_name,
            alpha=0.6,
            s=50
        )
    
    ax.set_xlabel('Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Component 2', fontsize=12, fontweight='bold')
    ax.set_title(f'{method.upper()} Visualization of Embeddings', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : Optional[list]
        List of class names
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names if class_names else np.unique(y_true),
        yticklabels=class_names if class_names else np.unique(y_true),
        cbar=True,
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
):
    """
    Plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return roc_auc


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    class_names: Optional[list] = None,
    verbose: bool = True
) -> dict:
    """
    Comprehensive classifier evaluation.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : Optional[np.ndarray]
        Predicted probabilities (for ROC AUC)
    class_names : Optional[list]
        List of class names
    verbose : bool
        Print results
        
    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Classification report
    if verbose:
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    if verbose:
        print("\nConfusion Matrix:")
        print(cm)
    
    # ROC AUC (if probabilities provided)
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        metrics['roc_auc'] = roc_auc
        
        if verbose:
            print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    return metrics


def plot_cv_scores(
    cv_scores: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot cross-validation scores.
    
    Parameters
    ----------
    cv_scores : np.ndarray
        Array of CV scores
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save the plot
    """
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    folds = np.arange(1, len(cv_scores) + 1)
    
    ax.bar(folds, cv_scores, color='steelblue', edgecolor='black', linewidth=1.5)
    ax.axhline(mean_score, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_score:.4f} ± {std_score:.4f}')
    
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Validation Scores', fontsize=14, fontweight='bold')
    ax.set_xticks(folds)
    ax.set_xticklabels([f'Fold {i}' for i in folds])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
