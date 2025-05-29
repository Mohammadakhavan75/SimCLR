import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """NT-Xent loss for SimCLR - Paper Exact Implementation
    
    This loss encourages representations of augmented versions of the same image
    to be similar, while pushing apart representations of different images.
    
    For a batch of N images, we create 2N augmented views. For each view,
    we treat the other augmented view of the same image as positive, and
    all other 2N-2 views as negatives.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Projections of first augmented views (batch_size, projection_dim)
            z_j: Projections of second augmented views (batch_size, projection_dim)
        """
        batch_size = z_i.shape[0]
        
        # L2 normalize features (important for SimCLR)
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        
        # Concatenate projections: [z_i; z_j] -> (2*batch_size, projection_dim)
        z = torch.cat([z_i, z_j], dim=0)
        
        # Compute cosine similarity matrix using dot product (since vectors are normalized)
        similarity_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Create labels for positive pairs
        # For first N samples, positive is at index N+i
        # For next N samples, positive is at index i-N  
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)]).to(z.device)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = self.criterion(similarity_matrix, labels) / (2 * batch_size)
        
        return loss