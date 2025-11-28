# Run snippets in order:
╔════════════════════════════════════════════════════════════════════╗
║         MULTI-LABEL TEXT CLASSIFICATION - EXECUTION GUIDE         ║
╚════════════════════════════════════════════════════════════════════╝

EXECUTION ORDER:
================

✓ SNIPPET 1: Setup & Configuration
  → Installs deps, loads config, initializes MLflow

✓ SNIPPET 2: Data Loading & Preprocessing
  → Loads dataset: Twitter Multilabel Classification (https://www.kaggle.com/datasets/prox37/twitter-multilabel-classification-dataset)
  → Tokenizes & cleans text
  → Creates multi-hot labels 

✓ SNIPPET 3: Dense Embeddings Generation [CRITICAL]
  → Trains Word2Vec, GloVe, FastText on tokens
  → Converts each text → fixed-size dense vector (mean pooling)
  → Extracts BERT [CLS] embeddings (768-dim)
  → Output: Dense embedding matrices per model

✓ SNIPPET 4: Dataset Preparation
  → Creates PyTorch datasets with dense embeddings
  → Creates train/val/test dataloaders
  → Input: Dense vectors, Output: Labels

✓ SNIPPET 5: Neural Network Architecture
  → Builds multi-label classifier with dense input
  → Architecture: Dense → FC layer → hidden layers → 28 outputs
  → Loss: BCEWithLogitsLoss

✓ SNIPPET 6: Training & Evaluation Functions
  → train_model(): 1 epoch of training
  → evaluate_model(): Compute metrics (F1, precision, recall, hamming)

✓ SNIPPET 7: Training Loop with MLflow [MAIN]
  → For each embedding model:
    • Create neural network
    • Train for N epochs with early stopping
    • Log all metrics to MLflow
    • Save best model
    • Test evaluation

✓ SNIPPET 8: Results Comparison & Visualization
  → Create comparison DataFrame
  → Generate 4-subplot visualization
  → Log results to MLflow
  → Show rankings

✓ SNIPPET 9: MLflow Tracking Summary
  → View all runs and experiments
  → Display MLflow UI command

╔════════════════════════════════════════════════════════════════════╗
║                        KEY REQUIREMENTS MET                        ║
╚════════════════════════════════════════════════════════════════════╝

✓ 1) Deep Neural Network in PyTorch
   ✓ MultiLabelClassifier with configurable FC layers

✓ 2) Multi-label Text Classification
   ✓ 28 emotion labels per text
   ✓ BCEWithLogitsLoss

✓ 3) ONLY After Converting to Dense Embeddings
   ✓ Text → Embeddings (CRITICAL STEP in SNIPPET 3)
   ✓ Embeddings → Dataset (SNIPPET 4)
   ✓ Dataset → Neural Network (SNIPPET 5-7)

✓ 4) Compare 4 Embedding Models with MLflow
   ✓ Word2Vec (100-dim, skip-gram)
   ✓ GloVe (100-dim, CBOW)
   ✓ FastText (100-dim, skip-gram with subwords)
   ✓ BERT (768-dim, contextual)
   ✓ All tracked in MLflow


