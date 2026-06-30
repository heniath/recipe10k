import os
import argparse
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
import gensim.models.keyedvectors as word2vec

from model import Im2RecipeModel
from dataset import get_dataloaders
from loss import Im2RecipeLoss

def load_word2vec_weights(w2v_path):
    print(f"Loading pre-trained Word2Vec from {w2v_path}...")
    w2v = word2vec.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    
    # We shift indices by 1 because index 0 is reserved for <pad>
    vocab_size = len(w2v.key_to_index) + 1
    embed_dim = w2v.vector_size
    
    weights = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for word, idx in w2v.key_to_index.items():
        weights[idx + 1] = w2v[word]
        
    print(f"Loaded {vocab_size - 1} words. Embedding dimension: {embed_dim}")
    return w2v, torch.from_numpy(weights)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Word2Vec
    if not os.path.exists(args.word2vec_path):
        raise FileNotFoundError(f"Could not find word2vec file at {args.word2vec_path}")
        
    w2v_model, pretrained_weights = load_word2vec_weights(args.word2vec_path)
    
    # Get DataLoaders and Vocab
    train_loader, val_loader, instr_vocab, ingr_vocab = get_dataloaders(
        data_dir=args.data_dir,
        w2v_model=w2v_model,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    
    instr_vocab_size = len(instr_vocab)
    print(f"Instruction Vocab size: {instr_vocab_size}")
    
    # Initialize Model
    print("Initializing model...")
    # 1048 semantic classes from the paper
    model = Im2RecipeModel(
        instr_vocab_size=instr_vocab_size, 
        ingr_pretrained_weights=pretrained_weights,
        num_classes=1048, 
        embed_dim=1024, 
        use_semantic_reg=False
    )
    model = model.to(device)
    
    # Initialize Loss and Optimizer
    criterion = Im2RecipeLoss(margin=0.3, sem_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Optional: Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
    
    print("Starting training loop...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_trip = 0.0
        running_sem = 0.0
        
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for i, batch in enumerate(pbar):
            img, instr, instr_len, ingr, ingr_len, label = batch
            
            # Move to device
            img = img.to(device)
            instr = instr.to(device)
            instr_len = instr_len.to(device)
            ingr = ingr.to(device)
            ingr_len = ingr_len.to(device)
            label = label.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(img, instr, instr_len, ingr, ingr_len)
            
            # Compute loss
            total_loss, trip_loss, sem_loss = criterion(output, label)
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            # Update metrics
            running_loss += total_loss.item()
            running_trip += trip_loss.item()
            running_sem += sem_loss.item()
            
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'Loss': f"{total_loss.item():.4f}"})
                
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_trip = 0.0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in pbar_val:
                img, instr, instr_len, ingr, ingr_len, label = batch
                
                img = img.to(device)
                instr = instr.to(device)
                instr_len = instr_len.to(device)
                ingr = ingr.to(device)
                ingr_len = ingr_len.to(device)
                label = label.to(device)
                
                output = model(img, instr, instr_len, ingr, ingr_len)
                loss, t_loss, _ = criterion(output, label)
                
                val_loss += loss.item()
                val_trip += t_loss.item()
                
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        print(f"--- Epoch {epoch+1} Completed in {epoch_time:.2f}s ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\\n")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"=> Saved new best model to {save_path}")

    print("Training Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Im2Recipe Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory containing json files and images')
    parser.add_argument('--img_dir', type=str, default=None, help='Path to the images directory, if separate from data_dir')
    parser.add_argument('--word2vec_path', type=str, required=True, help='Path to the vocab.bin file for ingredients')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the best model')
    parser.add_argument('--resume', type=str, default='', help='Path to a checkpoint to resume from')
    
    args = parser.parse_args()
    
    train(args)
