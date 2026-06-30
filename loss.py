import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineMarginLoss(nn.Module):
    """
    Cosine margin triplet loss as used in the Im2Recipe paper.
    It encourages positive pairs (image, recipe) to be closer than negative pairs.
    """
    def __init__(self, margin=0.3):
        super(CosineMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, im_emb, rec_emb):
        """
        im_emb: [batch_size, embed_dim]
        rec_emb: [batch_size, embed_dim]
        Assumes both embeddings are already L2 normalized.
        """
        batch_size = im_emb.size(0)
        
        # Calculate cosine similarity matrix between all images and all recipes in the batch
        # Since they are L2 normalized, cosine similarity is just the dot product
        scores = torch.mm(im_emb, rec_emb.t()) # [batch_size, batch_size]
        
        # The diagonal contains the positive pairs (i=j)
        diagonal = scores.diag().view(batch_size, 1) # [batch_size, 1]
        
        # Margin loss: max(0, margin - cos(pos) + cos(neg))
        
        # Cost for image to recipe matching (im as anchor)
        # We want diag > scores + margin for all non-diagonal elements
        cost_im = (self.margin + scores - diagonal).clamp(min=0)
        
        # Cost for recipe to image matching (rec as anchor)
        # We want diag > scores + margin for all non-diagonal elements
        cost_rec = (self.margin + scores - diagonal.t()).clamp(min=0)
        
        # Clear diagonals since they correspond to positive pairs (i=j) and cost should be 0
        cost_im = cost_im.fill_diagonal_(0)
        cost_rec = cost_rec.fill_diagonal_(0)
        
        # Sum all costs and average by batch size
        # Alternatively, we could take the max violating negative (hard negative mining)
        # But standard implementation sums over all negatives in the batch
        return (cost_im.sum() + cost_rec.sum()) / (batch_size * (batch_size - 1))

class Im2RecipeLoss(nn.Module):
    """
    Combined loss function including Semantic Regularization.
    """
    def __init__(self, margin=0.3, sem_weight=0.1):
        super(Im2RecipeLoss, self).__init__()
        self.triplet_loss = CosineMarginLoss(margin=margin)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sem_weight = sem_weight

    def forward(self, output, target_label):
        """
        output: [visual_emb, recipe_emb, visual_sem, recipe_sem] if using semantic reg
                [visual_emb, recipe_emb] otherwise
        target_label: [batch_size] ground truth class labels
        """
        if len(output) == 4:
            visual_emb, recipe_emb, visual_sem, recipe_sem = output
            
            trip_loss = self.triplet_loss(visual_emb, recipe_emb)
            
            sem_loss_im = self.cross_entropy(visual_sem, target_label)
            sem_loss_rec = self.cross_entropy(recipe_sem, target_label)
            
            total_loss = trip_loss + self.sem_weight * (sem_loss_im + sem_loss_rec)
            return total_loss, trip_loss, (sem_loss_im + sem_loss_rec)
        else:
            visual_emb, recipe_emb = output
            trip_loss = self.triplet_loss(visual_emb, recipe_emb)
            return trip_loss, trip_loss, torch.tensor(0.0)
