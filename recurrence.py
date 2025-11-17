"""
Recurrence plot generation module.

Converts sequence embeddings into visual recurrence matrix representations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, Union
from scipy.spatial.distance import cdist


class RecurrencePlotGenerator:
    """
    Generates recurrence plots from embedded sequences.
    
    Recurrence plots visualize the recurrence of states in a dynamical system.
    Points where the distance between states falls below a threshold (epsilon)
    are marked, revealing temporal patterns and self-similarity.
    
    Parameters
    ----------
    epsilon : Optional[float], default=None
        Distance threshold. If None, set to 10% of std of distance matrix
    distance_metric : str, default='euclidean'
        Distance metric: 'euclidean', 'cosine', 'manhattan'
    image_size : Tuple[int, int], default=(128, 128)
        Size of output recurrence plot images
    colormap : str, default='binary'
        Matplotlib colormap for visualization
    """
    
    def __init__(
        self,
        epsilon: Optional[float] = None,
        distance_metric: str = 'euclidean',
        image_size: Tuple[int, int] = (128, 128),
        colormap: str = 'binary'
    ):
        self.epsilon = epsilon
        self.distance_metric = distance_metric
        self.image_size = image_size
        self.colormap = colormap
        
    def compute_recurrence_matrix(
        self,
        embedding: np.ndarray,
        epsilon: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute recurrence matrix from an embedding sequence.
        
        Parameters
        ----------
        embedding : np.ndarray
            Embedded sequence, shape (sequence_length, embedding_dim)
        epsilon : Optional[float]
            Distance threshold. If None, uses instance epsilon or auto-calculates
            
        Returns
        -------
        recurrence_matrix : np.ndarray
            Binary recurrence matrix, shape (sequence_length, sequence_length)
        """
        embedding = np.array(embedding)
        N = embedding.shape[0]
        
        # Compute distance matrix
        if self.distance_metric == 'euclidean':
            # Efficient pairwise Euclidean distance
            distance_matrix = np.linalg.norm(
                embedding[:, None, :] - embedding[None, :, :],
                axis=2
            )
        else:
            # Use scipy for other metrics
            distance_matrix = cdist(embedding, embedding, metric=self.distance_metric)
        
        # Determine epsilon
        if epsilon is None:
            epsilon = self.epsilon
        if epsilon is None:
            epsilon = 0.1 * np.std(distance_matrix)
            if epsilon == 0:
                epsilon = 0.001
        
        # Create recurrence matrix
        recurrence_matrix = np.zeros((N, N))
        recurrence_matrix[distance_matrix <= epsilon] = 1
        
        return recurrence_matrix
    
    def save_recurrence_plot(
        self,
        recurrence_matrix: np.ndarray,
        file_path: str,
        dpi: int = 100
    ):
        """
        Save recurrence matrix as an image.
        
        Parameters
        ----------
        recurrence_matrix : np.ndarray
            Binary recurrence matrix
        file_path : str
            Path to save the image
        dpi : int, default=100
            Resolution of the saved image
        """
        fig, ax = plt.subplots(figsize=(self.image_size[0]/dpi, self.image_size[1]/dpi), dpi=dpi)
        ax.imshow(recurrence_matrix, cmap=self.colormap, origin='lower')
        ax.axis('off')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig)
    
    def generate_plots(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        output_dir: str = 'recurrence_plots',
        label_names: Optional[dict] = None,
        verbose: bool = True
    ) -> list:
        """
        Generate and save recurrence plots for multiple embeddings.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Array of embeddings, shape (n_samples, sequence_length, embedding_dim)
        labels : Optional[np.ndarray]
            Class labels for each sample
        output_dir : str
            Directory to save plots
        label_names : Optional[dict]
            Mapping from label indices to names
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        file_paths : list
            List of paths to saved recurrence plot images
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        file_paths = []
        n_samples = len(embeddings)
        
        if verbose:
            print(f"Generating {n_samples} recurrence plots...")
        
        for i, embedding in enumerate(embeddings):
            # Compute recurrence matrix
            rec_matrix = self.compute_recurrence_matrix(embedding)
            
            # Determine filename
            if labels is not None:
                label = labels[i]
                if label_names is not None:
                    label_str = label_names.get(label, f"class_{label}")
                else:
                    label_str = f"class_{label}"
                filename = f"sample_{i:05d}_{label_str}.png"
            else:
                filename = f"sample_{i:05d}.png"
            
            file_path = os.path.join(output_dir, filename)
            
            # Save plot
            self.save_recurrence_plot(rec_matrix, file_path)
            file_paths.append(file_path)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{n_samples} plots")
        
        if verbose:
            print(f"Recurrence plots saved to: {output_dir}")
        
        return file_paths
    
    def visualize_comparison(
        self,
        embeddings: list,
        labels: list,
        label_names: dict,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Visualize recurrence plots side-by-side for comparison.
        
        Parameters
        ----------
        embeddings : list
            List of embeddings to compare
        labels : list
            List of corresponding labels
        label_names : dict
            Mapping from labels to names
        figsize : Tuple[int, int]
            Figure size
        save_path : Optional[str]
            Path to save comparison figure
        """
        n_samples = len(embeddings)
        fig, axes = plt.subplots(1, n_samples, figsize=figsize)
        
        if n_samples == 1:
            axes = [axes]
        
        for i, (embedding, label) in enumerate(zip(embeddings, labels)):
            rec_matrix = self.compute_recurrence_matrix(embedding)
            axes[i].imshow(rec_matrix, cmap=self.colormap, origin='lower')
            axes[i].set_title(f"{label_names[label]}", fontsize=12, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compute_recurrence_quantification(
        self,
        recurrence_matrix: np.ndarray
    ) -> dict:
        """
        Compute recurrence quantification analysis (RQA) measures.
        
        Parameters
        ----------
        recurrence_matrix : np.ndarray
            Binary recurrence matrix
            
        Returns
        -------
        measures : dict
            Dictionary of RQA measures
        """
        N = len(recurrence_matrix)
        
        # Recurrence rate
        RR = np.sum(recurrence_matrix) / (N * N)
        
        # Determinism (requires diagonal line analysis)
        diagonals = {}
        for offset in range(-N+1, N):
            diag = np.diagonal(recurrence_matrix, offset=offset)
            diagonals[offset] = diag
        
        measures = {
            'recurrence_rate': RR,
            'size': N
        }
        
        return measures
