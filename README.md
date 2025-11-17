# Signal2Recurrence: Deep Metric Learning for Sequential Signal Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)

A generalizable deep learning pipeline for analyzing sequential signals through recurrence plot transformation and metric learning. Originally developed for character-level linguistic biomarkers in dementia detection (95.9% AUC), this methodology can be applied to any sequential signal data.

## 📖 Overview

This repository implements a novel methodology that transforms sequential signals into visual recurrence patterns, then learns discriminative embeddings using deep metric learning. The approach is domain-agnostic and has been validated on speech data but can be applied to:

- **Biomedical Signals**: ECG, EEG, EMG, speech patterns
- **Financial Time Series**: Stock prices, trading patterns
- **Industrial Sensors**: Manufacturing quality control, predictive maintenance
- **Behavioral Data**: User interaction sequences, activity recognition
- **Natural Language**: Character or word-level text analysis

## 🔬 Methodology

The pipeline consists of four key stages:

### 1. Signal Preprocessing
- Converts sequential data into fixed-length representations
- Supports custom tokenization/embedding strategies
- Handles variable-length sequences with padding

### 2. Recurrence Plot Generation
- Transforms signal embeddings into visual recurrence matrices
- Captures temporal dynamics and self-similarity patterns
- Uses Euclidean distance with adaptive epsilon thresholding

### 3. Deep Metric Learning (Siamese Network)
- Learns discriminative embeddings through contrastive loss
- Trains pairs of similar/dissimilar samples
- CNN-based architecture for feature extraction

### 4. Classification
- Uses learned embeddings for downstream tasks
- Supports any classifier (XGBoost, Random Forest, SVM, etc.)
- Includes cross-validation and performance metrics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/signal2recurrence.git
cd signal2recurrence

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from signal2recurrence import SignalPipeline
import pandas as pd

# Load your sequential data
# Format: DataFrame with 'signal' and 'label' columns
data = pd.read_csv('your_data.csv')

# Initialize pipeline
pipeline = SignalPipeline(
    embedding_dim=32,
    max_sequence_length=None,  # Auto-detect
    recurrence_epsilon=None,    # Auto-calculate
    image_size=(128, 128)
)

# Process signals and generate recurrence plots
pipeline.fit_transform(
    signals=data['signal'],
    labels=data['label'],
    save_plots=True,
    output_dir='recurrence_plots'
)

# Train deep metric learning model
pipeline.train_siamese_network(
    epochs=20,
    batch_size=16,
    validation_split=0.2
)

# Extract embeddings
embeddings = pipeline.get_embeddings()

# Train classifier
from xgboost import XGBClassifier
classifier = XGBClassifier(random_state=42)
classifier.fit(embeddings['train'], data['label_train'])

# Evaluate
accuracy = classifier.score(embeddings['test'], data['label_test'])
```

## 📁 Repository Structure

```
signal2recurrence/
├── signal2recurrence/          # Main package
│   ├── __init__.py
│   ├── preprocessing.py        # Signal preprocessing & embedding
│   ├── recurrence.py          # Recurrence plot generation
│   ├── siamese.py             # Siamese network implementation
│   ├── pipeline.py            # End-to-end pipeline
│   └── utils.py               # Utility functions
├── examples/                   # Usage examples
│   ├── speech_analysis.py     # Character-level speech example
│   ├── ecg_classification.py  # ECG signal example
│   └── custom_signals.py      # Generic signal template
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks
│   └── demo.ipynb             # Interactive demo
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🔧 Detailed Configuration

### Preprocessing Options

```python
from signal2recurrence.preprocessing import SignalPreprocessor

preprocessor = SignalPreprocessor(
    tokenization='character',     # 'character', 'word', 'custom'
    embedding_type='learned',     # 'learned', 'onehot', 'pretrained'
    embedding_dim=32,
    max_length=None,             # Auto-detect or specify
    padding='post',              # 'post' or 'pre'
    truncation='post'            # 'post' or 'pre'
)
```

### Recurrence Plot Parameters

```python
from signal2recurrence.recurrence import RecurrencePlotGenerator

rp_generator = RecurrencePlotGenerator(
    epsilon=None,                # Auto-calculate or specify
    distance_metric='euclidean', # 'euclidean', 'cosine', 'manhattan'
    image_size=(128, 128),
    colormap='binary'
)
```

### Siamese Network Architecture

```python
from signal2recurrence.siamese import SiameseNetwork

siamese = SiameseNetwork(
    input_shape=(128, 128, 1),
    base_filters=32,
    embedding_dim=128,
    learning_rate=0.001,
    margin=1.0                   # Contrastive loss margin
)
```

## 📊 Performance Metrics

The original implementation achieved:
- **ROC AUC**: 95.9% (character-level linguistic biomarkers)
- **Stratified 5-Fold CV**: 0.9589 ± 0.0142
- **Precision/Recall**: Balanced across classes

## 🎯 Use Cases

### Medical Applications
- Early detection of cognitive decline
- Parkinson's disease voice analysis
- Sleep apnea detection from breathing patterns

### Industrial IoT
- Anomaly detection in sensor data
- Predictive maintenance from vibration signals
- Quality control in manufacturing

### Finance
- Fraud detection in transaction sequences
- Market regime classification
- Trading pattern recognition

## 📚 Citation

If you use this methodology in your research, please cite:

```bibtex
@article{yourlastname2024signal2recurrence,
  title={Character-Level Linguistic Biomarkers with Deep Metric Learning for Dementia Detection},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Developed at Penn State University, Industrial Engineering Department
- Funded by NSF I-Corps ($50K)
- Forbes 30 Under 30 Healthcare Recognition

## 📧 Contact

Kevin - [@DementiAnalytics](https://twitter.com/yourhandle)

Project Link: [https://github.com/yourusername/signal2recurrence](https://github.com/yourusername/signal2recurrence)

---

**Note**: This is a research tool. For medical applications, consult with healthcare professionals and follow appropriate regulatory guidelines.
