"""
Example: Character-Level Speech Analysis for Dementia Detection

This example demonstrates the application of the signal2recurrence methodology
to linguistic biomarkers for cognitive health assessment.
"""

import pandas as pd
import numpy as np
from signal2recurrence import SignalPipeline
from signal2recurrence.utils import (
    visualize_embeddings,
    plot_confusion_matrix,
    plot_roc_curve,
    evaluate_classifier,
    plot_cv_scores
)
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score


def main():
    # Load data
    # Expected format: CSV with 'utts' (speech transcripts) and 'labels' (0=Dementia, 1=Healthy)
    print("Loading data...")
    df = pd.read_csv('cookie.csv')
    
    # Extract signals and labels
    signals = df['utts']
    labels = df['labels']
    
    print(f"Loaded {len(signals)} samples")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Initialize pipeline
    print("\n" + "="*50)
    print("Initializing Signal2Recurrence Pipeline")
    print("="*50)
    
    pipeline = SignalPipeline(
        embedding_dim=32,
        max_sequence_length=None,  # Auto-detect
        recurrence_epsilon=None,    # Auto-calculate
        image_size=(128, 128),
        siamese_embedding_dim=128
    )
    
    # Step 1-2: Preprocess and generate recurrence plots
    print("\nProcessing signals...")
    recurrence_images = pipeline.fit_transform(
        signals=signals,
        labels=labels,
        tokenization='character',
        save_plots=True,
        output_dir='recurrence_plots_characters',
        verbose=True
    )
    
    # Step 3: Train Siamese network
    print("\n" + "="*50)
    print("Training Deep Metric Learning Model")
    print("="*50)
    
    pipeline.train_siamese_network(
        epochs=20,
        batch_size=16,
        validation_split=0.2,
        learning_rate=0.001,
        verbose=1
    )
    
    # Step 4: Extract embeddings
    print("\n" + "="*50)
    print("Extracting Learned Embeddings")
    print("="*50)
    
    embeddings_data = pipeline.get_embeddings(split=True, test_size=0.2)
    
    X_train = embeddings_data['train']
    X_test = embeddings_data['test']
    y_train = embeddings_data['train_labels']
    y_test = embeddings_data['test_labels']
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Visualize embeddings
    print("\nVisualizing embeddings...")
    all_embeddings = pipeline.get_embeddings(split=False)
    visualize_embeddings(
        all_embeddings,
        labels,
        method='tsne',
        label_names={0: 'Dementia', 1: 'Healthy'},
        save_path='embeddings_tsne.png'
    )
    
    # Step 5: Train XGBoost classifier
    print("\n" + "="*50)
    print("Training XGBoost Classifier")
    print("="*50)
    
    # Grid search for best hyperparameters
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9]
    }
    
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    print("Running grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV ROC AUC Score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    y_pred_proba = best_clf.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("Test Set Evaluation")
    print("="*50)
    
    metrics = evaluate_classifier(
        y_test,
        y_pred,
        y_pred_proba,
        class_names=['Dementia', 'Healthy'],
        verbose=True
    )
    
    # Plot results
    print("\nGenerating visualizations...")
    
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names=['Dementia', 'Healthy'],
        save_path='confusion_matrix.png'
    )
    
    plot_roc_curve(
        y_test,
        y_pred_proba,
        save_path='roc_curve.png'
    )
    
    # Cross-validation on full dataset
    print("\n" + "="*50)
    print("Stratified K-Fold Cross-Validation")
    print("="*50)
    
    all_embeddings = pipeline.get_embeddings(split=False)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        best_clf,
        all_embeddings,
        labels,
        cv=skf,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    print(f"Cross-Validation ROC AUC Scores: {cv_scores}")
    print(f"Mean ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    plot_cv_scores(cv_scores, save_path='cv_scores.png')
    
    # Save model
    print("\nSaving trained model...")
    pipeline.save_model('siamese_model_weights.h5')
    
    print("\n" + "="*50)
    print("Analysis Complete!")
    print("="*50)
    print("\nResults Summary:")
    print(f"  - Test ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  - CV Mean ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\nOutputs saved:")
    print(f"  - Recurrence plots: recurrence_plots_characters/")
    print(f"  - Model weights: siamese_model_weights.h5")
    print(f"  - Visualizations: embeddings_tsne.png, confusion_matrix.png, roc_curve.png")


if __name__ == "__main__":
    main()
