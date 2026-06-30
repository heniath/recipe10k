import torch
from model import Im2RecipeModel
from dataset import DummyRecipeDataset

def test_forward_pass():
    print("Testing forward pass of the Im2Recipe model...")
    # Initialize model
    model = Im2RecipeModel(vocab_size=5000, num_classes=1048, embed_dim=1024, use_semantic_reg=True)
    model.eval()
    
    # Get a single dummy sample
    dataset = DummyRecipeDataset(size=1)
    img, instr, instr_len, ingr, ingr_len, label = dataset[0]
    
    # Add batch dimension
    img = img.unsqueeze(0)
    instr = instr.unsqueeze(0)
    instr_len = torch.tensor([instr_len])
    ingr = ingr.unsqueeze(0)
    ingr_len = torch.tensor([ingr_len])
    
    # Forward pass
    with torch.no_grad():
        output = model(img, instr, instr_len, ingr, ingr_len)
        
    print("Forward pass successful!")
    visual_emb, recipe_emb, visual_sem, recipe_sem = output
    
    print(f"Visual Embedding Shape: {visual_emb.shape}")
    print(f"Recipe Embedding Shape: {recipe_emb.shape}")
    print(f"Visual Semantic Shape: {visual_sem.shape}")
    print(f"Recipe Semantic Shape: {recipe_sem.shape}")

if __name__ == "__main__":
    test_forward_pass()
