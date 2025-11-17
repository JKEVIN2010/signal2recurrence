"""
Signal preprocessing and embedding module.

Handles tokenization, embedding, and sequence padding for various signal types.
"""

import numpy as np
import re
from typing import List, Optional, Union, Tuple
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.models import Model


class SignalPreprocessor:
    """
    Preprocesses sequential signals for recurrence plot generation.
    
    Supports multiple tokenization strategies and embedding types for flexible
    signal representation.
    
    Parameters
    ----------
    tokenization : str, default='character'
        Type of tokenization: 'character', 'word', or 'custom'
    embedding_type : str, default='learned'
        Type of embedding: 'learned', 'onehot', or 'pretrained'
    embedding_dim : int, default=32
        Dimension of learned embeddings
    max_length : Optional[int], default=None
        Maximum sequence length. If None, automatically determined from data
    padding : str, default='post'
        Padding strategy: 'post' or 'pre'
    truncation : str, default='post'
        Truncation strategy: 'post' or 'pre'
    vocab_size : Optional[int], default=None
        Vocabulary size for learned embeddings
    """
    
    def __init__(
        self,
        tokenization: str = 'character',
        embedding_type: str = 'learned',
        embedding_dim: int = 32,
        max_length: Optional[int] = None,
        padding: str = 'post',
        truncation: str = 'post',
        vocab_size: Optional[int] = None
    ):
        self.tokenization = tokenization
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.vocab_size = vocab_size
        
        self.token_to_index = {}
        self.index_to_token = {}
        self.embedding_model = None
        
    def fit(self, signals: List[Union[str, List]]) -> 'SignalPreprocessor':
        """
        Fit the preprocessor to the signals.
        
        Parameters
        ----------
        signals : List[Union[str, List]]
            List of signals (strings for text, lists for sequences)
            
        Returns
        -------
        self : SignalPreprocessor
            Fitted preprocessor
        """
        # Tokenize all signals
        tokenized_signals = [self._tokenize(signal) for signal in signals]
        
        # Build vocabulary
        all_tokens = set()
        for tokens in tokenized_signals:
            all_tokens.update(tokens)
        
        token_list = sorted(list(all_tokens))
        self.token_to_index = {token: idx for idx, token in enumerate(token_list)}
        self.index_to_token = {idx: token for token, idx in self.token_to_index.items()}
        self.vocab_size = len(token_list)
        
        # Determine max length if not specified
        if self.max_length is None:
            lengths = [len(tokens) for tokens in tokenized_signals]
            self.max_length = int(np.max(lengths))
            
        # Initialize embedding model if using learned embeddings
        if self.embedding_type == 'learned':
            self._build_embedding_model()
            
        return self
    
    def transform(self, signals: List[Union[str, List]]) -> np.ndarray:
        """
        Transform signals to embeddings.
        
        Parameters
        ----------
        signals : List[Union[str, List]]
            List of signals to transform
            
        Returns
        -------
        embeddings : np.ndarray
            Embedded representations, shape (n_samples, max_length, embedding_dim)
        """
        # Tokenize
        tokenized_signals = [self._tokenize(signal) for signal in signals]
        
        # Convert to indices
        indexed_signals = [self._tokens_to_indices(tokens) for tokens in tokenized_signals]
        
        # Pad sequences
        padded_sequences = pad_sequences(
            indexed_signals,
            maxlen=self.max_length,
            padding=self.padding,
            truncating=self.truncation
        )
        
        # Generate embeddings
        if self.embedding_type == 'learned':
            embeddings = self.embedding_model.predict(padded_sequences, verbose=0)
        elif self.embedding_type == 'onehot':
            embeddings = self._to_onehot(padded_sequences)
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
            
        return embeddings
    
    def fit_transform(self, signals: List[Union[str, List]]) -> np.ndarray:
        """
        Fit preprocessor and transform signals in one step.
        
        Parameters
        ----------
        signals : List[Union[str, List]]
            List of signals
            
        Returns
        -------
        embeddings : np.ndarray
            Embedded representations
        """
        return self.fit(signals).transform(signals)
    
    def _tokenize(self, signal: Union[str, List]) -> List:
        """Tokenize a single signal."""
        if self.tokenization == 'character':
            # Text preprocessing for character-level
            if isinstance(signal, str):
                signal = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', signal)
                signal = signal.lower()
                return list(signal)
            return signal
        elif self.tokenization == 'word':
            # Word-level tokenization
            if isinstance(signal, str):
                return signal.lower().split()
            return signal
        elif self.tokenization == 'custom':
            # Assume signal is already tokenized as a list
            return signal if isinstance(signal, list) else [signal]
        else:
            raise ValueError(f"Unsupported tokenization: {self.tokenization}")
    
    def _tokens_to_indices(self, tokens: List) -> List[int]:
        """Convert tokens to indices."""
        return [self.token_to_index.get(token, 0) for token in tokens]
    
    def _build_embedding_model(self):
        """Build Keras embedding layer model."""
        input_seq = Input(shape=(self.max_length,))
        embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length
        )
        embedded_seq = embedding_layer(input_seq)
        self.embedding_model = Model(inputs=input_seq, outputs=embedded_seq)
    
    def _to_onehot(self, sequences: np.ndarray) -> np.ndarray:
        """Convert sequences to one-hot embeddings."""
        batch_size, seq_length = sequences.shape
        onehot = np.zeros((batch_size, seq_length, self.vocab_size))
        
        for i, seq in enumerate(sequences):
            for j, idx in enumerate(seq):
                if idx < self.vocab_size:
                    onehot[i, j, idx] = 1
                    
        return onehot
    
    def get_vocab(self) -> dict:
        """Return vocabulary mapping."""
        return self.token_to_index.copy()
    
    def get_config(self) -> dict:
        """Return configuration dictionary."""
        return {
            'tokenization': self.tokenization,
            'embedding_type': self.embedding_type,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'padding': self.padding,
            'truncation': self.truncation,
            'vocab_size': self.vocab_size
        }
