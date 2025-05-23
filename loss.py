import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """NT-Xent loss for SimCLR
    
    This loss encourages representations of augmented versions of the same image
    to be similar, while pushing apart representations of different images.
    
    For a batch of N images, we create 2N augmented views. For each view,
    we treat the other augmented view of the same image as positive, and
    all other 2N-2 views as negatives.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Projections of first augmented views (batch_size, projection_dim)
            z_j: Projections of second augmented views (batch_size, projection_dim)
        """
        batch_size = z_i.shape[0]
        
        # Concatenate projections: [z_i; z_j] -> (2*batch_size, projection_dim)
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute cosine similarity matrix
        similarity_matrix = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0))
        
        # Create mask to remove self-similarity (diagonal elements)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Create positive pairs mask
        # For indices [0, 1, 2, ..., N-1, N, N+1, ..., 2N-1]
        # Positive pairs are: (0,N), (1,N+1), ..., (N-1,2N-1), (N,0), (N+1,1), ..., (2N-1,N-1)
        positive_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z.device)
        for i in range(batch_size):
            positive_mask[i, batch_size + i] = True
            positive_mask[batch_size + i, i] = True
        
        # Extract positive similarities
        positive_similarities = similarity_matrix[positive_mask].view(2 * batch_size, 1)
        
        # Extract negative similarities
        negative_similarities = similarity_matrix[~positive_mask].view(2 * batch_size, -1)
        
        # Concatenate positive and negative similarities
        logits = torch.cat([positive_similarities, negative_similarities], dim=1) / self.temperature
        
        # Labels: positive pairs are always at index 0
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        
        # Compute cross-entropy loss
        loss = self.criterion(logits, labels) / (2 * batch_size)
        
        return loss