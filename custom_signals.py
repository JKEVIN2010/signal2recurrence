"""
Example: Custom Signal Analysis Template

This template demonstrates how to apply the signal2recurrence methodology
to any sequential signal data.
"""

import numpy as np
import pandas as pd
from signal2recurrence import SignalPipeline
from signal2recurrence.utils import visualize_embeddings, evaluate_classifier
from sklearn.ensemble import RandomForestClassifier


def load_your_data():
    """
    Load your custom signal data.
    
    Returns
    -------
    signals : list
        List of sequential signals. Each signal can be:
        - A string (will be tokenized by character or word)
        - A list of tokens/values
        - A numpy array
    labels : np.ndarray
        Array of labels for each signal
    """
    # REPLACE THIS with your data loading logic
    # Example: Load from CSV, database, or generate synthetic data
    
    # Synthetic example for demonstration
    np.random.seed(42)
    n_samples = 200
    
    signals = []
    labels = []
    
    for i in range(n_samples):
        # Generate synthetic sequential data
        # Class 0: Pattern A
        # Class 1: Pattern B
        if i < n_samples // 2:
            # Pattern A: Low values with occasional spikes
            signal = list(np.random.randint(0, 5, size=50))
            signal[10] = 20
            signal[30] = 20
            label = 0
        else:
            # Pattern B: High values with occasional dips
            signal = list(np.random.randint(15, 20, size=50))
            signal[10] = 5
            signal[30] = 5
            label = 1
        
        signals.append(signal)
        labels.append(label)
    
    return signals, np.array(labels)


def main():
    print("="*60)
    print("Custom Signal Analysis with Signal2Recurrence")
    print("="*60)
    
    # Load your data
    print("\n1. Loading data...")
    signals, labels = load_your_data()
    
    print(f"   Loaded {len(signals)} signals")
    print(f"   Signal length example: {len(signals[0])}")
    print(f"   Class distribution: {np.bincount(labels)}")
    
    # Initialize pipeline
    print("\n2. Initializing pipeline...")
    pipeline = SignalPipeline(
        embedding_dim=32,           # Dimension for initial embeddings
        max_sequence_length=None,   # Auto-detect from data
        recurrence_epsilon=None,    # Auto-calculate threshold
        image_size=(128, 128),      # Size of recurrence plots
        siamese_embedding_dim=128   # Dimension of learned embeddings
    )
    
    # Process signals
    # For custom tokenization, signals should already be in list format
    print("\n3. Generating recurrence plots...")
    pipeline.fit_transform(
        signals=signals,
        labels=labels,
        tokenization='custom',  # Use 'custom' for pre-tokenized signals
        save_plots=True,
        output_dir='custom_recurrence_plots',
        verbose=True
    )
    
    # Train deep metric learning model
    print("\n4. Training Siamese network...")
    pipeline.train_siamese_network(
        epochs=20,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Extract embeddings
    print("\n5. Extracting embeddings...")
    embeddings_data = pipeline.get_embeddings(split=True, test_size=0.2)
    
    X_train = embeddings_data['train']
    X_test = embeddings_data['test']
    y_train = embeddings_data['train_labels']
    y_test = embeddings_data['test_labels']
    
    # Visualize embeddings
    print("\n6. Visualizing embeddings...")
    all_embeddings = pipeline.get_embeddings(split=False)
    visualize_embeddings(
        all_embeddings,
        labels,
        method='tsne',
        label_names={0: 'Class 0', 1: 'Class 1'},
        save_path='custom_embeddings_tsne.png'
    )
    
    # Train classifier
    print("\n7. Training classifier...")
    # You can use any classifier here
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("\n8. Evaluating on test set...")
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_classifier(
        y_test,
        y_pred,
        y_pred_proba,
        class_names=['Class 0', 'Class 1'],
        verbose=True
    )
    
    # Save model
    print("\n9. Saving model...")
    pipeline.save_model('custom_model_weights.h5')
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nTest ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nGenerated files:")
    print("  - Recurrence plots: custom_recurrence_plots/")
    print("  - Model weights: custom_model_weights.h5")
    print("  - Visualization: custom_embeddings_tsne.png")


if __name__ == "__main__":
    main()
