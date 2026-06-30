import torch
import torch.nn as nn
import torchvision.models as models

def l2_norm(input_tensor, p=2, dim=1, eps=1e-12):
    """L2 normalization for embeddings"""
    return input_tensor / input_tensor.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input_tensor)

class InstructionEncoder(nn.Module):
    """
    Equivalent to stRNN in the original paper.
    Encodes recipe instructions using an LSTM. 
    We use nn.Embedding so it can be trained end-to-end on raw text.
    """
    def __init__(self, vocab_size, embed_dim=300, hidden_size=1024):
        super(InstructionEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=False, batch_first=True)
                
    def forward(self, x, seq_lengths):
        """
        x: [batch_size, max_seq_len] (word indices)
        seq_lengths: 1D tensor of valid lengths for each sequence in the batch
        """
        x = self.embedding(x) # [batch_size, max_seq_len, embed_dim]

        sorted_len, sorted_idx = seq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu(), batch_first=True
        )
        
        out, (hidden, _) = self.lstm(packed_seq)

        _, original_idx = sorted_idx.sort(0, descending=False)
        
        unsorted_idx = original_idx.view(1, -1, 1).expand_as(hidden)
        output = hidden.gather(1, unsorted_idx)
        
        output = output.squeeze(0)
        return output 


class IngredientEncoder(nn.Module):
    """
    Equivalent to ingRNN in the original paper.
    Encodes ingredients using a bidirectional LSTM over pre-trained word2vec embeddings.
    """
    def __init__(self, pretrained_weights, embed_dim=300, hidden_size=300):
        super(IngredientEncoder, self).__init__()
        vocab_size = pretrained_weights.size(0)
        
        self.embs = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Copy pre-trained weights
        self.embs.weight.data.copy_(pretrained_weights)
        # Allow fine-tuning (if you want to freeze them, set requires_grad = False)
        self.embs.weight.requires_grad = True
        
        self.irnn = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x, seq_lengths):
        """
        x: [batch_size, max_seq_len] of ingredient indices
        seq_lengths: 1D tensor of valid lengths
        """
        x = self.embs(x) 

        sorted_len, sorted_idx = seq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu(), batch_first=True
        )
        
        out, (hidden, _) = self.irnn(packed_seq)

        _, original_idx = sorted_idx.sort(0, descending=False)
        
        unsorted_idx = original_idx.view(1, -1, 1).expand_as(hidden)
        output = hidden.gather(1, unsorted_idx)
        
        output = output.transpose(0, 1).contiguous()
        output = output.view(output.size(0), -1)
        
        return output


class Im2RecipeModel(nn.Module):
    """
    The full Im2Recipe architecture.
    """
    def __init__(self, instr_vocab_size, ingr_pretrained_weights, num_classes=1048, embed_dim=1024, use_semantic_reg=False):
        super(Im2RecipeModel, self).__init__()
        self.use_semantic_reg = use_semantic_reg
        
        # Vision Branch (ResNet-50)
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.vision_mlp = nn.Sequential(*modules)
        
        self.visual_embedding = nn.Sequential(
            nn.Linear(2048, embed_dim), 
            nn.Tanh()
        )
        
        # Recipe Branch
        self.instruction_encoder = InstructionEncoder(vocab_size=instr_vocab_size, embed_dim=300, hidden_size=1024)
        
        # Ingredient Branch uses pre-trained weights
        self.ingredient_encoder = IngredientEncoder(pretrained_weights=ingr_pretrained_weights, embed_dim=300, hidden_size=300)
        
        # 1024 (stRNN) + 300*2 (ingRNN bi-LSTM) = 1624
        self.recipe_embedding = nn.Sequential(
            nn.Linear(1624, embed_dim),
            nn.Tanh()
        )
        
        # Semantic Regularization Branch
        if self.use_semantic_reg:
            self.semantic_branch = nn.Linear(embed_dim, num_classes)

    def forward(self, img, instr, instr_lens, ingr, ingr_lens):
        """
        img: [batch, 3, 224, 224]
        instr: [batch, max_instr_len]
        instr_lens: [batch]
        ingr: [batch, max_ingr_len]
        ingr_lens: [batch]
        """
        # --- Recipe Embedding ---
        instr_out = self.instruction_encoder(instr, instr_lens)
        ingr_out = self.ingredient_encoder(ingr, ingr_lens)
        
        recipe_feat = torch.cat([instr_out, ingr_out], dim=1)
        
        recipe_emb = self.recipe_embedding(recipe_feat)
        recipe_emb = l2_norm(recipe_emb)
        
        # --- Visual Embedding ---
        visual_feat = self.vision_mlp(img)
        visual_feat = visual_feat.view(visual_feat.size(0), -1) 
        
        visual_emb = self.visual_embedding(visual_feat)
        visual_emb = l2_norm(visual_emb)
        
        # --- Output ---
        if self.use_semantic_reg:
            visual_sem = self.semantic_branch(visual_emb)
            recipe_sem = self.semantic_branch(recipe_emb)
            return visual_emb, recipe_emb, visual_sem, recipe_sem
        else:
            return visual_emb, recipe_emb
