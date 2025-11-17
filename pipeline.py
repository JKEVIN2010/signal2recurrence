"""
End-to-end pipeline for signal analysis.

Combines preprocessing, recurrence plot generation, and deep metric learning.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict
from .preprocessing import SignalPreprocessor
from .recurrence import RecurrencePlotGenerator
from .siamese import SiameseNetwork


class SignalPipeline:
    """
    Complete pipeline for signal-to-embedding transformation.
    
    This class orchestrates the entire workflow:
    1. Signal preprocessing and embedding
    2. Recurrence plot generation
    3. Deep metric learning
    4. Feature extraction
    
    Parameters
    ----------
    embedding_dim : int, default=32
        Dimension for signal embeddings
    max_sequence_length : Optional[int], default=None
        Maximum sequence length (auto-detected if None)
    recurrence_epsilon : Optional[float], default=None
        Threshold for recurrence plots (auto-calculated if None)
    image_size : Tuple[int, int], default=(128, 128)
        Size of recurrence plot images
    siamese_embedding_dim : int, default=128
        Dimension of learned embeddings from Siamese network
    """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        max_sequence_length: Optional[int] = None,
        recurrence_epsilon: Optional[float] = None,
        image_size: Tuple[int, int] = (128, 128),
        siamese_embedding_dim: int = 128
    ):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.recurrence_epsilon = recurrence_epsilon
        self.image_size = image_size
        self.siamese_embedding_dim = siamese_embedding_dim
        
        # Initialize components
        self.preprocessor = None
        self.recurrence_generator = None
        self.siamese_network = None
        
        # Data storage
        self.signal_embeddings = None
        self.recurrence_images = None
        self.learned_embeddings = None
        self.labels = None
    
    def fit_transform(
        self,
        signals: Union[List, pd.Series],
        labels: Optional[Union[List, pd.Series, np.ndarray]] = None,
        tokenization: str = 'character',
        save_plots: bool = True,
        output_dir: str = 'recurrence_plots',
        verbose: bool = True
    ) -> np.ndarray:
        """
        Fit the pipeline and transform signals to recurrence plots.
        
        Parameters
        ----------
        signals : Union[List, pd.Series]
            Sequential signals (text strings or sequences)
        labels : Optional[Union[List, pd.Series, np.ndarray]]
            Class labels for each signal
        tokenization : str
            Tokenization strategy
        save_plots : bool
            Whether to save recurrence plots to disk
        output_dir : str
            Directory for saving plots
        verbose : bool
            Print progress information
            
        Returns
        -------
        recurrence_images : np.ndarray
            Array of recurrence plot images
        """
        # Convert to appropriate format
        if isinstance(signals, pd.Series):
            signals = signals.tolist()
        if labels is not None and isinstance(labels, pd.Series):
            labels = labels.values
        
        self.labels = labels
        
        # Step 1: Preprocess signals
        if verbose:
            print("Step 1: Preprocessing signals...")
        
        self.preprocessor = SignalPreprocessor(
            tokenization=tokenization,
            embedding_type='learned',
            embedding_dim=self.embedding_dim,
            max_length=self.max_sequence_length
        )
        
        self.signal_embeddings = self.preprocessor.fit_transform(signals)
        
        if verbose:
            print(f"  Created embeddings: {self.signal_embeddings.shape}")
        
        # Step 2: Generate recurrence plots
        if verbose:
            print("Step 2: Generating recurrence plots...")
        
        self.recurrence_generator = RecurrencePlotGenerator(
            epsilon=self.recurrence_epsilon,
            image_size=self.image_size
        )
        
        # Save plots if requested
        if save_plots:
            label_names = None
            if labels is not None:
                unique_labels = np.unique(labels)
                label_names = {label: f"class_{label}" for label in unique_labels}
            
            self.recurrence_generator.generate_plots(
                embeddings=self.signal_embeddings,
                labels=labels,
                output_dir=output_dir,
                label_names=label_names,
                verbose=verbose
            )
        
        # Load recurrence plot images
        self.recurrence_images = self._load_recurrence_images(output_dir)
        
        if verbose:
            print(f"  Generated {len(self.recurrence_images)} recurrence plots")
        
        return self.recurrence_images
    
    def train_siamese_network(
        self,
        epochs: int = 20,
        batch_size: int = 16,
        validation_split: float = 0.2,
        learning_rate: float = 0.001,
        verbose: int = 1
    ):
        """
        Train the Siamese network for deep metric learning.
        
        Parameters
        ----------
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        validation_split : float
            Fraction of data for validation
        learning_rate : float
            Learning rate
        verbose : int
            Verbosity level
        """
        if self.recurrence_images is None or self.labels is None:
            raise ValueError("Must run fit_transform first")
        
        print("Step 3: Training Siamese network...")
        
        # Initialize Siamese network
        input_shape = self.recurrence_images.shape[1:]
        self.siamese_network = SiameseNetwork(
            input_shape=input_shape,
            embedding_dim=self.siamese_embedding_dim,
            learning_rate=learning_rate
        )
        
        # Train
        self.siamese_network.train(
            images=self.recurrence_images,
            labels=self.labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        print("  Training complete!")
    
    def get_embeddings(self, split: bool = False, test_size: float = 0.2) -> Union[np.ndarray, Dict]:
        """
        Extract learned embeddings from the Siamese network.
        
        Parameters
        ----------
        split : bool
            Whether to split into train/test sets
        test_size : float
            Fraction of data for testing (if split=True)
            
        Returns
        -------
        embeddings : Union[np.ndarray, Dict]
            If split=False: array of embeddings
            If split=True: dict with 'train', 'test', 'train_labels', 'test_labels'
        """
        if self.siamese_network is None:
            raise ValueError("Must train Siamese network first")
        
        self.learned_embeddings = self.siamese_network.get_embeddings(self.recurrence_images)
        
        if not split:
            return self.learned_embeddings
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.learned_embeddings,
            self.labels,
            test_size=test_size,
            random_state=42,
            stratify=self.labels
        )
        
        return {
            'train': X_train,
            'test': X_test,
            'train_labels': y_train,
            'test_labels': y_test
        }
    
    def _load_recurrence_images(self, image_dir: str) -> np.ndarray:
        """Load recurrence plot images from directory."""
        if self.siamese_network is None:
            # Create temporary network just to load images
            temp_network = SiameseNetwork(input_shape=(self.image_size[0], self.image_size[1], 1))
            images, _ = temp_network.load_images_from_directory(image_dir)
            return images
        else:
            images, _ = self.siamese_network.load_images_from_directory(image_dir)
            return images
    
    def save_model(self, filepath: str):
        """Save the trained Siamese network weights."""
        if self.siamese_network is not None:
            self.siamese_network.save_weights(filepath)
    
    def load_model(self, filepath: str):
        """Load Siamese network weights."""
        if self.siamese_network is None:
            # Must initialize first
            if self.recurrence_images is not None:
                input_shape = self.recurrence_images.shape[1:]
                self.siamese_network = SiameseNetwork(
                    input_shape=input_shape,
                    embedding_dim=self.siamese_embedding_dim
                )
        self.siamese_network.load_weights(filepath)
    
    def get_config(self) -> dict:
        """Return pipeline configuration."""
        config = {
            'embedding_dim': self.embedding_dim,
            'max_sequence_length': self.max_sequence_length,
            'recurrence_epsilon': self.recurrence_epsilon,
            'image_size': self.image_size,
            'siamese_embedding_dim': self.siamese_embedding_dim
        }
        
        if self.preprocessor is not None:
            config['preprocessor'] = self.preprocessor.get_config()
        
        return config
