import os
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F

from model import Im2RecipeModel
from dataset import get_dataloaders
from train import load_word2vec_weights

def test_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Could not find model weights at {args.weights_path}")
        
    from dataset import get_dataloaders, Recipe1MDataset
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    print("Loading vocab from training dataset...")
    w2v_model, pretrained_weights = load_word2vec_weights(args.word2vec_path)
    
    # We need the vocabs built from the training set
    _, _, instr_vocab, ingr_vocab = get_dataloaders(
        data_dir=args.data_dir,
        w2v_model=w2v_model,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    
    print("Loading Test Dataset...")
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = Recipe1MDataset(data_dir=args.data_dir, split='test', 
                                  instr_vocab=instr_vocab, ingr_vocab=ingr_vocab, 
                                  transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    print("Initializing model...")
    model = Im2RecipeModel(
        instr_vocab_size=len(instr_vocab), 
        ingr_pretrained_weights=pretrained_weights,
        num_classes=1048, 
        embed_dim=1024, 
        use_semantic_reg=False
    )
    
    print(f"Loading weights from {args.weights_path}")
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    img_embs, rec_embs = [], []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Extracting features for {args.num_samples} samples")
        for batch in pbar:
            img, instr, instr_len, ingr, ingr_len, _ = batch
            
            output = model(img.to(device), instr.to(device), instr_len.to(device), 
                           ingr.to(device), ingr_len.to(device))
            
            visual_emb, recipe_emb, _, _ = output
            
            # L2 normalize embeddings before cosine similarity
            visual_emb = F.normalize(visual_emb, p=2, dim=1)
            recipe_emb = F.normalize(recipe_emb, p=2, dim=1)
            
            img_embs.append(visual_emb.cpu())
            rec_embs.append(recipe_emb.cpu())
            
            if len(img_embs) * test_loader.batch_size >= args.num_samples:
                break
                
    img_embs = torch.cat(img_embs, dim=0)[:args.num_samples]
    rec_embs = torch.cat(rec_embs, dim=0)[:args.num_samples]
    
    num_samples_extracted = img_embs.size(0)
    print(f"Calculating Similarity Matrix for {num_samples_extracted} pairs...")
    
    sims = torch.matmul(img_embs, rec_embs.t())
    
    ranks = []
    for i in range(num_samples_extracted):
        d = torch.argsort(sims[i], descending=True)
        rank = (d == i).nonzero(as_tuple=True)[0].item()
        ranks.append(rank + 1)
        
    ranks = np.array(ranks)
    print("\n" + "="*40)
    print(" TEST RESULTS: IMAGE-TO-RECIPE RETRIEVAL ")
    print("="*40)
    print(f" Median Rank (MedR) : {np.median(ranks):.1f}")
    print(f" Recall@1  (R@1)    : {100.0 * len(ranks[ranks <= 1]) / num_samples_extracted:.2f}%")
    print(f" Recall@5  (R@5)    : {100.0 * len(ranks[ranks <= 5]) / num_samples_extracted:.2f}%")
    print(f" Recall@10 (R@10)   : {100.0 * len(ranks[ranks <= 10]) / num_samples_extracted:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Im2Recipe Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--word2vec_path', type=str, required=True, help='Path to the vocab.bin file')
    parser.add_argument('--weights_path', type=str, default='./best_model.pth', help='Path to the trained model weights')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to evaluate for MedR')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers')
    
    args = parser.parse_args()
    test_model(args)
