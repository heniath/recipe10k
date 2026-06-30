# Recipe10K (Im2Recipe) - Kaggle Training Module

This repository contains the modularized PyTorch implementation of Im2Recipe, specifically optimized for training on Kaggle notebooks using a 10K recipe subset.

## Features
- **End-to-End Training:** Avoids massive pre-computed Skip-thought matrices by utilizing `nn.Embedding` for instructions.
- **Pre-trained Word2Vec Integration:** Seamlessly loads ingredient embeddings (`vocab.bin`) using `gensim`.
- **Batch-All Triplet Loss:** Upgraded from the paper's original random matching to calculate Cosine Margin Loss over the entire batch for better gradient flow.
- **Kaggle Ready:** Designed to run via CLI directly on cloud instances.

## Usage on Kaggle

1. Upload your subset data (including `layer1_subset.json`, `layer2_subset.json`, `images/`, and `vocab.bin`) as a Kaggle Dataset.
2. In a Kaggle Notebook (with GPU enabled), run:

```bash
# Clone the repository
!git clone https://github.com/heniath/recipe10k.git
%cd recipe10k

# Install required dependencies
!pip install gensim tqdm

# Train the model
!python train.py \
    --data_dir /kaggle/input/your-dataset-name \
    --word2vec_path /kaggle/input/your-dataset-name/vocab.bin \
    --epochs 10 \
    --batch_size 32
```

## Structure
- `dataset.py`: Dataloader and Vocabulary logic.
- `model.py`: Neural network architectures (Instruction Encoder, Ingredient Encoder, Vision MLP).
- `loss.py`: Cosine Margin Loss implementation.
- `train.py`: Main training loop with `argparse`.
