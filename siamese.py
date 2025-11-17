"""
Siamese network implementation for deep metric learning.

Trains a neural network to learn discriminative embeddings through contrastive loss.
"""

import numpy as np
from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os


class SiameseNetwork:
    """
    Siamese Network for learning embeddings via contrastive loss.
    
    The network learns to map similar inputs close together and dissimilar
    inputs far apart in the embedding space.
    
    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        Shape of input images (height, width, channels)
    base_filters : int, default=32
        Number of filters in first convolutional layer
    embedding_dim : int, default=128
        Dimension of the learned embedding space
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    margin : float, default=1.0
        Margin for contrastive loss
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (128, 128, 1),
        base_filters: int = 32,
        embedding_dim: int = 128,
        learning_rate: float = 0.001,
        margin: float = 1.0
    ):
        self.input_shape = input_shape
        self.base_filters = base_filters
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        
        self.base_network = None
        self.siamese_model = None
        self.history = None
        
    def create_base_network(self) -> Model:
        """
        Create the base CNN network for feature extraction.
        
        Returns
        -------
        model : Model
            Base network model
        """
        input_layer = Input(shape=self.input_shape)
        
        # Convolutional layers
        x = Conv2D(self.base_filters, (3, 3), activation='relu', padding='same')(input_layer)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(self.base_filters * 2, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(self.base_filters * 4, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Dense layers
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.embedding_dim, activation='relu')(x)
        
        model = Model(inputs=input_layer, outputs=x)
        return model
    
    @staticmethod
    def euclidean_distance(vectors: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Compute Euclidean distance between two vectors.
        
        Parameters
        ----------
        vectors : Tuple[tf.Tensor, tf.Tensor]
            Pair of vectors
            
        Returns
        -------
        distance : tf.Tensor
            Euclidean distance
        """
        x, y = vectors
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    def contrastive_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Contrastive loss function.
        
        Parameters
        ----------
        y_true : tf.Tensor
            True labels (1 for similar, 0 for dissimilar)
        y_pred : tf.Tensor
            Predicted distances
            
        Returns
        -------
        loss : tf.Tensor
            Contrastive loss value
        """
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    def build(self):
        """Build the complete Siamese network architecture."""
        # Create base network
        self.base_network = self.create_base_network()
        
        # Define inputs for the pair
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        
        # Process both inputs through base network
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)
        
        # Compute distance
        distance = Lambda(self.euclidean_distance)([processed_a, processed_b])
        
        # Create model
        self.siamese_model = Model(inputs=[input_a, input_b], outputs=distance)
        
        # Compile
        self.siamese_model.compile(
            loss=self.contrastive_loss,
            optimizer=Adam(learning_rate=self.learning_rate)
        )
    
    def create_pairs(
        self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create pairs of images with labels (1=similar, 0=dissimilar).
        
        Parameters
        ----------
        images : np.ndarray
            Array of images, shape (n_samples, height, width, channels)
        labels : np.ndarray
            Array of labels
            
        Returns
        -------
        pairs : np.ndarray
            Array of image pairs, shape (n_pairs, 2, height, width, channels)
        pair_labels : np.ndarray
            Array of pair labels (1 or 0)
        """
        pairs = []
        pair_labels = []
        
        num_classes = len(np.unique(labels))
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        
        # Find minimum samples per class
        min_samples = min(len(idx) for idx in class_indices) - 1
        
        if min_samples < 1:
            raise ValueError("Not enough samples per class to create pairs")
        
        for idx in range(min_samples):
            for class_idx in range(num_classes):
                # Positive pair (same class)
                anchor_idx = class_indices[class_idx][idx]
                positive_idx = class_indices[class_idx][idx + 1]
                pairs.append([images[anchor_idx], images[positive_idx]])
                pair_labels.append(1)
                
                # Negative pair (different classes)
                negative_class_idx = (class_idx + 1) % num_classes
                negative_idx = class_indices[negative_class_idx][idx]
                pairs.append([images[anchor_idx], images[negative_idx]])
                pair_labels.append(0)
        
        return np.array(pairs), np.array(pair_labels)
    
    def train(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 20,
        batch_size: int = 16,
        verbose: int = 1
    ):
        """
        Train the Siamese network.
        
        Parameters
        ----------
        images : np.ndarray
            Training images
        labels : np.ndarray
            Training labels
        validation_split : float
            Fraction of data to use for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        verbose : int
            Verbosity level
        """
        if self.siamese_model is None:
            self.build()
        
        # Create pairs
        pairs, pair_labels = self.create_pairs(images, labels)
        
        # Split into training and validation
        n_val = int(len(pairs) * validation_split)
        pairs_train = pairs[:-n_val]
        pairs_val = pairs[-n_val:]
        labels_train = pair_labels[:-n_val]
        labels_val = pair_labels[-n_val:]
        
        # Unpack pairs
        x_train_a = pairs_train[:, 0]
        x_train_b = pairs_train[:, 1]
        x_val_a = pairs_val[:, 0]
        x_val_b = pairs_val[:, 1]
        
        # Train
        self.history = self.siamese_model.fit(
            [x_train_a, x_train_b],
            labels_train,
            validation_data=([x_val_a, x_val_b], labels_val),
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose
        )
    
    def get_embeddings(self, images: np.ndarray) -> np.ndarray:
        """
        Extract embeddings using the trained base network.
        
        Parameters
        ----------
        images : np.ndarray
            Input images
            
        Returns
        -------
        embeddings : np.ndarray
            Learned embeddings, shape (n_samples, embedding_dim)
        """
        if self.base_network is None:
            raise ValueError("Model not built or trained yet")
        
        return self.base_network.predict(images, verbose=0)
    
    def load_images_from_directory(
        self,
        image_dir: str,
        target_size: Optional[Tuple[int, int]] = None,
        color_mode: str = 'grayscale'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images and labels from a directory.
        
        Expects filenames in format: sample_XXXXX_labelname.png
        
        Parameters
        ----------
        image_dir : str
            Directory containing images
        target_size : Optional[Tuple[int, int]]
            Target size for images. If None, uses input_shape
        color_mode : str
            Color mode: 'grayscale' or 'rgb'
            
        Returns
        -------
        images : np.ndarray
            Array of images
        labels : np.ndarray
            Array of labels
        """
        if target_size is None:
            target_size = (self.input_shape[0], self.input_shape[1])
        
        images = []
        labels = []
        
        for filename in sorted(os.listdir(image_dir)):
            if filename.endswith('.png'):
                # Load image
                img_path = os.path.join(image_dir, filename)
                img = load_img(img_path, target_size=target_size, color_mode=color_mode)
                img_array = img_to_array(img) / 255.0  # Normalize
                images.append(img_array)
                
                # Extract label from filename
                # Assumes format: sample_XXXXX_labelname.png
                parts = filename.split('_')
                if len(parts) >= 3:
                    label_str = parts[-1].replace('.png', '')
                    # Try to convert to integer, otherwise use as string
                    try:
                        label = int(label_str) if label_str.isdigit() else label_str
                    except:
                        label = label_str
                    labels.append(label)
        
        return np.array(images), np.array(labels)
    
    def save_weights(self, filepath: str):
        """Save model weights."""
        if self.base_network is not None:
            self.base_network.save_weights(filepath)
    
    def load_weights(self, filepath: str):
        """Load model weights."""
        if self.base_network is None:
            self.build()
        self.base_network.load_weights(filepath)
