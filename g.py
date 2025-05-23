# SimCLR PyTorch Implementation
# Based on: "A Simple Framework for Contrastive Learning of Visual Representations"
# by Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton (2020)
# https://arxiv.org/abs/2002.05709

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import math # for LARS optimizer

# --- 1. Data Augmentation ---
# SimCLR uses a specific set of augmentations.
# The paper emphasizes strong augmentations for good performance.

class SimCLRAugmentation:
    """
    Implements the augmentations used in SimCLR.
    Two augmented views (xi, xj) are generated from the same image.
    """
    def __init__(self, image_size, s=1.0, p_blur=0.5):
        """
        Args:
            image_size (int or tuple): The size to resize images to.
            s (float): Strength of color distortion.
            p_blur (float): Probability of applying Gaussian blur.
        """
        self.image_size = image_size
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)

        # Color jitter parameters from the paper (Sec 3.1, Appendix A)
        # Brightness, contrast, saturation, hue
        self.color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        self.transform = T.Compose([
            T.RandomResizedCrop(size=self.image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([self.color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            # Gaussian blur is applied with a probability of 0.5
            # Kernel size is 10% of the image height/width
            # Sigma is chosen uniformly at random between (0.1, 2.0)
            T.RandomApply([T.GaussianBlur(kernel_size=int(0.1 * self.image_size[0]), sigma=(0.1, 2.0))], p=p_blur),
            T.ToTensor(),
            # Normalization: SimCLR paper used per-channel mean and std of ImageNet
            # but often custom dataset stats are used. For simplicity, using common values.
            # For replicating paper results, ImageNet stats should be used if training on ImageNet.
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
        ])

    def __call__(self, x):
        # Apply the transform twice to get two augmented views
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj

# --- 2. Encoder (ResNet) ---
# The paper uses ResNet-50 as the base encoder f(.).
# We'll use a pre-trained ResNet-50 and modify it.

def get_resnet(name='resnet50', pretrained=True, **kwargs):
    """
    Returns a ResNet encoder.
    Args:
        name (str): Name of the ResNet architecture (e.g., 'resnet18', 'resnet50').
        pretrained (bool): Whether to use a pretrained model.
    """
    if name == 'resnet18':
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None, **kwargs)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None, **kwargs)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None, **kwargs)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None, **kwargs)
    else:
        raise ValueError(f"Unsupported ResNet: {name}")

    # The SimCLR paper removes the final fully connected layer (classifier)
    # The output of the average pooling layer is used as h_i
    model.fc = nn.Identity() # Replace fc layer with identity
    return model

# --- 3. Projection Head ---
# A small neural network g(.) that maps representations to the space where contrastive loss is applied.
# The paper uses a 2-layer MLP (MLP head).
# h_i = f(x_i), z_i = g(h_i)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        """
        Args:
            input_dim (int): Dimension of the input features (output of ResNet).
                             For ResNet-50, this is 2048.
            hidden_dim (int): Dimension of the hidden layer. Paper uses same as input_dim.
            output_dim (int): Dimension of the output projection. Paper uses 128.
        """
        super().__init__()
        # Paper: "a MLP with one hidden layer to obtain z_i = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity"
        # No batch norm mentioned for projection head in original paper, but some implementations add it.
        # The original paper found non-linear projection better than linear or no projection.
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.head(x)

# --- 4. SimCLR Model ---
# Combines the encoder and projection head.

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_head):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x):
        """
        Input x can be a single batch of images, or a tuple of two augmented views.
        If single batch, it means we are doing inference/evaluation after training.
        """
        if not isinstance(x, tuple): # For evaluation or single view processing
            h = self.encoder(x)
            return h # Return features before projection head for downstream tasks

        xi, xj = x
        hi = self.encoder(xi)
        hj = self.encoder(xj)

        zi = self.projection_head(hi)
        zj = self.projection_head(hj)

        # L2 normalize the projections (z_i, z_j) as per paper (Sec 3.1)
        # "l2 normalization is applied to z_i and z_j"
        zi = F.normalize(zi, p=2, dim=1)
        zj = F.normalize(zj, p=2, dim=1)

        return zi, zj

# --- 5. Contrastive Loss (NT-Xent) ---
# Normalized Temperature-scaled Cross Entropy Loss.

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device='cuda'): # Paper uses 0.07 for ImageNet, 0.5 for CIFAR-10
        """
        Args:
            temperature (float): Temperature scaling parameter.
            device (str): Device to run computations on.
        """
        super().__init__()
        self.temperature = temperature
        self.device = device
        # For calculating similarity matrix
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # Sum over batch, then average over GPUs if any
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, zis, zjs):
        """
        zis: Batch of projections of the first augmented view (N, D)
        zjs: Batch of projections of the second augmented view (N, D)
        N: Batch size, D: Projection dimension
        """
        N = zis.shape[0] # Batch size

        # Concatenate all projections: [z_i_1, ..., z_i_N, z_j_1, ..., z_j_N]
        # Shape: (2N, D)
        representations = torch.cat([zis, zjs], dim=0)

        # Calculate similarity matrix (2N, 2N)
        # sim_matrix[k, l] = cos_sim(z_k, z_l) / temperature
        # Using einsum for clarity and efficiency in calculating pairwise cosine similarity
        # A more direct way:
        # sim_matrix = self.similarity_f(representations.unsqueeze(1), representations.unsqueeze(0)) / self.temperature
        
        # Manual cosine similarity and temperature scaling
        dot_similarity = torch.matmul(representations, representations.T) # (2N, 2N)
        # representations_norm = torch.norm(representations, p=2, dim=1, keepdim=True) # Already normalized
        # cosine_similarity = dot_similarity / (torch.matmul(representations_norm, representations_norm.T) + 1e-8)
        # Since representations are L2 normalized, dot product is cosine similarity.
        sim_matrix = dot_similarity / self.temperature # (2N, 2N)

        # Mask out diagonal part (similarity of a sample with itself)
        # This is important to prevent trivial solutions.
        # sim_matrix.fill_diagonal_(-float('inf')) # Or a very large negative number
        # A better way to create the mask for positive pairs:
        # The positive pairs are (z_i_k, z_j_k) and (z_j_k, z_i_k)

        # Create labels: positive pairs are (i, i+N) and (i+N, i)
        # For a batch of N, the first N are z_i and the next N are z_j
        # So, z_i_k corresponds to index k, and z_j_k corresponds to index k+N
        # The positive pair for z_i_k is z_j_k (index k+N)
        # The positive pair for z_j_k is z_i_k (index k)

        labels = torch.arange(2 * N, device=self.device).long()
        # For row i (0 to N-1), positive is i+N
        # For row i (N to 2N-1), positive is i-N
        mask = torch.eye(2 * N, dtype=torch.bool, device=self.device) # Mask for self-similarity
        labels = labels.roll(N, dims=0) # Positive pair for z_i_k is z_j_k, and for z_j_k is z_i_k

        # Remove self-similarity from logits by setting diagonal to a very small number
        # before softmax. This prevents log(0) and ensures these are not chosen.
        logits = sim_matrix[~mask].view(2 * N, -1) # (2N, 2N-1), remove self-similarity
        
        # The labels for CrossEntropyLoss need to be adjusted because we removed the diagonal
        # For each row k in the original sim_matrix (0 to 2N-1):
        #   - The positive sample was at index `labels[k]`.
        #   - In the `logits` matrix (where diagonal is removed), if `labels[k]` was > k,
        #     its new index is `labels[k] - 1`.
        #   - If `labels[k]` was < k, its new index is `labels[k]`.
        
        # Simpler way: use the original sim_matrix and filter out the diagonal in the loss calculation
        # The paper's formula is:
        # L_k = -log [ exp(sim(z_i, z_j)/T) / (sum_{l!=k} exp(sim(z_i, z_l)/T)) ]
        # This is equivalent to CrossEntropyLoss where logits for positive pair are sim(z_i, z_j)/T
        # and logits for negative pairs are sim(z_i, z_l)/T.

        # Let's use the full sim_matrix and filter out the diagonal for the denominator.
        # The numerator is the similarity between positive pairs.
        # Positive pairs are (z_i_k, z_j_k) and (z_j_k, z_i_k)
        
        # Create positive pair mask
        pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        pos_mask[torch.arange(N), torch.arange(N) + N] = True
        pos_mask[torch.arange(N) + N, torch.arange(N)] = True

        # Numerator: sum of log(exp(sim(positive_pairs)/T)) for all positive pairs
        # This is sum of sim(positive_pairs)/T
        numerator = sim_matrix[pos_mask].view(2 * N, -1) # Should be (2N, 1)

        # Denominator: sum over exp(sim(z_k, z_l)/T) for l != k
        # This is logsumexp of non-diagonal elements for each row
        neg_mask = ~torch.eye(2 * N, dtype=torch.bool, device=self.device)
        denominator_logits = sim_matrix[neg_mask].view(2 * N, -1) # (2N, 2N-1)
        
        log_prob = numerator - torch.logsumexp(denominator_logits, dim=1, keepdim=True)
        loss = -log_prob.mean() # Mean over all 2N samples

        # Alternative using CrossEntropyLoss directly (more common implementation)
        # The positive pair for z_i (index k) is z_j (index k+N)
        # The positive pair for z_j (index k+N) is z_i (index k)
        
        # Create labels for CrossEntropyLoss
        # For the first N samples (zis), the positive is the corresponding zjs (indices N to 2N-1)
        # For the next N samples (zjs), the positive is the corresponding zis (indices 0 to N-1)
        # Example: N=2. zis=[z0,z1], zjs=[z2,z3] (originally z0', z1')
        # Representations: [z0,z1,z2,z3]
        # For z0, positive is z2. For z1, positive is z3.
        # For z2, positive is z0. For z3, positive is z1.
        # Labels: [2,3,0,1] for N=2
        
        # Create labels for CrossEntropyLoss
        # labels_ce[k] = k + N if k < N else k - N
        labels_ce = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(self.device).long()

        # We need to filter out the similarity of a sample with itself for the denominator
        # The CrossEntropyLoss does this implicitly if the diagonal is not the target.
        # However, the diagonal (self-similarity) should not be part of the candidates.
        # So, we create logits by masking out the diagonal.
        
        # Create a mask to exclude self-similarity (diagonal elements)
        diag_mask = torch.eye(2 * N, device=self.device, dtype=torch.bool)
        
        # sim_matrix has shape (2N, 2N)
        # logits_ce should be (2N, 2N-1) where the j-th column is sim(i, j) if j!=i
        # This is tricky to feed directly to CrossEntropyLoss which expects (N, C) and labels (N,)
        # where C is number of classes. Here, each sample is a "class" for its positive pair.

        # Simpler NT-Xent formulation:
        # For each z_k in the combined batch of 2N representations:
        # Positive pair: z_p (e.g., if z_k is an xi, z_p is its corresponding xj)
        # Negative pairs: all other 2N-2 representations in the batch.
        # Loss_k = -log [ exp(sim(z_k, z_p)/T) / (sum_{m!=k} exp(sim(z_k, z_m)/T)) ]
        # The sum in the denominator includes the positive pair z_p.
        # The loss is then averaged over all 2N samples.

        # Let's use the formulation from: https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
        # This one seems robust and widely used.

        # sim_matrix is (2N, 2N)
        # Create positive mask (diagonal in top-right and bottom-left blocks)
        # And negative mask (all others except main diagonal)
        
        # Positive pairs: (zi, zj)
        # zis: (N, D), zjs: (N, D)
        # Concatenated: Z = [zis, zjs] (2N, D)
        # Similarity matrix: S = Z @ Z.T (2N, 2N)
        
        # For each zi_k (row k in S, 0 <= k < N):
        #   Positive is zj_k (entry S[k, k+N])
        #   Negatives are all S[k, l] where l != k and l != k+N
        # For each zj_k (row k+N in S, 0 <= k < N):
        #   Positive is zi_k (entry S[k+N, k])
        #   Negatives are all S[k+N, l] where l != k+N and l != k

        # Create labels:
        # For the first N rows (zis), the positive sample is at index [N, N+1, ..., 2N-1]
        # For the next N rows (zjs), the positive sample is at index [0, 1, ..., N-1]
        # This is what `labels_ce` defined earlier does.
        
        # The `sim_matrix` contains similarities of (all_samples, all_samples).
        # `sim_matrix[i, j]` is sim(sample_i, sample_j).
        # `labels_ce[i]` is the index of the positive pair for sample_i.
        # CrossEntropyLoss(sim_matrix, labels_ce) would work if sim_matrix didn't include self-similarity
        # as a candidate for the denominator.

        # Let's use the explicit formula:
        # Numerator: exp(sim(z_i, z_j)/T)
        # Denominator: sum_{k'=1 to 2N, k'!=i} exp(sim(z_i, z_k')/T)

        # Extract positive similarities
        # For i from 0 to N-1 (zis): positive is sim(zis[i], zjs[i]) -> sim_matrix[i, i+N]
        # For i from N to 2N-1 (zjs): positive is sim(zjs[i-N], zis[i-N]) -> sim_matrix[i, i-N]
        
        exp_sim_matrix = torch.exp(sim_matrix)
        
        # Mask for diagonal elements (self-similarity)
        diag_mask = torch.eye(2 * N, device=self.device, dtype=torch.bool)
        
        # Sum of exp similarities for denominator (excluding self)
        # For each row i, sum_l!=i exp_sim_matrix[i, l]
        denominator_sum = exp_sim_matrix.masked_fill(diag_mask, 0).sum(dim=1) # (2N,)

        # Numerator terms: exp(sim(positive_pair)/T)
        # For i in 0..N-1 (zi_i): positive pair is zj_i (index i+N). Numerator: exp_sim_matrix[i, i+N]
        # For i in N..2N-1 (zj_{i-N}): positive pair is zi_{i-N} (index i-N). Numerator: exp_sim_matrix[i, i-N]
        
        num_terms = torch.zeros(2 * N, device=self.device)
        num_terms[:N] = exp_sim_matrix[torch.arange(N), torch.arange(N) + N]
        num_terms[N:] = exp_sim_matrix[torch.arange(N) + N, torch.arange(N)]

        # Loss for each sample
        loss_per_sample = -torch.log(num_terms / (denominator_sum + 1e-8)) # Add epsilon for stability
        
        # Total loss: average over all 2N samples
        loss = loss_per_sample.sum() / (2 * N) # Paper: "loss is computed across all GPUs" and averaged. Here, just batch.
        
        # The `criterion = nn.CrossEntropyLoss(reduction="sum")` approach with correct logits/labels is cleaner:
        # Logits for CrossEntropy: sim_matrix, but diagonal elements should be masked out for denominator.
        # If we set diagonal elements of sim_matrix to -infinity, then exp(diag) -> 0.
        # Then CrossEntropyLoss(sim_matrix_with_neg_inf_diag, labels_ce) should work.
        
        logits_for_ce = sim_matrix.clone()
        logits_for_ce.masked_fill_(diag_mask, -float('inf')) # Mask out self-similarity for CE denominator
        
        # labels_ce are [N, N+1, ..., 2N-1, 0, 1, ..., N-1]
        loss_ce = self.criterion(logits_for_ce, labels_ce)
        loss_ce = loss_ce / (2 * N) # Average loss per sample

        return loss_ce # Using the CrossEntropyLoss based one, it's more standard.


# --- 6. LARS Optimizer ---
# Layer-wise Adaptive Rate Scaling. The paper found this crucial for large batch sizes.
# Implementation from: https://github.com/facebookresearch/simsiam/blob/main/simsiam/lars.py
# Or from PyTorch Optimizers: torch.optim.LARS (if available in your PyTorch version)
# For simplicity, we'll use AdamW first, then note LARS.
# If PyTorch version is new enough, LARS is available.
# Let's try to implement a basic LARS if not available.

class LARS(optim.Optimizer):
    """
    LARS optimizer, implementation from PyTorch Lightning Bolts
    https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/optimizers/lars.py
    """
    def __init__(
        self,
        params,
        lr,
        momentum=0.9,
        weight_decay=1e-6, # Paper: 10^-6 for SimCLR
        trust_coefficient=0.001,
        eps=1e-8,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if trust_coefficient <= 0.0:
            raise ValueError("Invalid trust_coefficient value: {}".format(trust_coefficient))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            trust_coefficient = group["trust_coefficient"]
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Apply weight decay
                if weight_decay != 0:
                    # Perform weight decay by subtracting L2 penalty from gradient
                    # As in AdamW: grad = grad + weight_decay * p.data
                    # SimCLR paper mentions LARS + Adam style weight decay.
                    # Standard LARS applies WD to the param update.
                    # Let's follow common LARS: p.data.add_(p.data, alpha=-weight_decay * lr)
                    # Or, add to gradient:
                    grad.add_(p.data, alpha=weight_decay)


                # Compute LARS trust ratio
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(grad)

                # Compute local learning rate
                # local_lr = lr * trust_coefficient * param_norm / (grad_norm + eps) if param_norm > 0 and grad_norm > 0 else lr
                if param_norm > 0 and grad_norm > 0:
                    local_lr = trust_coefficient * param_norm / (grad_norm + eps + weight_decay * param_norm) # WD term in denom for stability
                else:
                    local_lr = 1.0 # If either is zero, use global LR scaling factor of 1


                # Update the momentum term
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.clone(grad).detach()
                
                buf = param_state["momentum_buffer"]
                buf.mul_(momentum).add_(grad, alpha=local_lr) # Update momentum buffer: m = beta*m + local_lr*g
                
                # Update parameters
                p.data.add_(buf, alpha=-lr) # p = p - global_lr * m


        return loss

# --- 7. Training Configuration & Loop ---
def main_simclr():
    # Hyperparameters from SimCLR paper (Table 3, Appendix B.1 for ImageNet)
    # For CIFAR-10, different params might be better (e.g., smaller batch size, higher temperature)
    
    # Common settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    IMAGE_SIZE = 224 # For ImageNet. For CIFAR-10, often 32 or upscaled. Paper used 224 for ImageNet.
    BATCH_SIZE = 512 # Paper: 256 to 8192. 4096 was common. Requires lots of memory/TPUs.
                     # Let's use a smaller one for typical GPU.
    EPOCHS = 100     # Paper: 100 for ResNet-50 on ImageNet-1k. 800 for larger models/datasets.
    LEARNING_RATE = 0.3 * (BATCH_SIZE / 256) # Linear scaling rule: LR = BaseLR * BatchSize / 256
                                             # BaseLR = 0.3 for LARS.
    WEIGHT_DECAY = 1e-6
    TEMPERATURE = 0.07 # For ImageNet with large batches. Paper suggests 0.5 for smaller batches/CIFAR-10.
    PROJECTION_DIM = 128
    ENCODER_NAME = 'resnet50' # 'resnet18' for faster training/testing

    # Dataset (Using CIFAR-10 as a placeholder, ImageNet is too large for a simple script)
    # For CIFAR-10, image size is 32x32. We might need to adjust augmentations or upsample.
    # Let's assume we are using a dataset compatible with ImageNet-style resizing (e.g. STL10 or a subset of ImageNet)
    # For simplicity, we'll use CIFAR10 and resize.
    
    print("Setting up augmentations...")
    # If using CIFAR-10 (32x32), resizing to 224 is a lot.
    # Let's use a smaller size for CIFAR-10, e.g., 32, and adjust temperature.
    # If you intend to replicate ImageNet results, use IMAGE_SIZE=224 and ImageNet dataset.
    
    # For CIFAR-10 example:
    CIFAR_IMAGE_SIZE = 32
    CIFAR_TEMPERATURE = 0.5 # Higher temp for smaller datasets/batches
    
    simclr_augmentations = SimCLRAugmentation(image_size=CIFAR_IMAGE_SIZE) # Adjust for dataset

    print("Loading dataset (CIFAR-10 example)...")
    # train_dataset = torchvision.datasets.ImageFolder('path/to/your/imagenet/subset/train', transform=simclr_augmentations)
    # For example, use CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=simclr_augmentations # This will apply the pair of augmentations
    )
    
    # The SimCLRAugmentation returns two views. DataLoader needs to handle this.
    # Default collate_fn should work fine if the transform returns a tuple (img1, img2).
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4, # Adjust based on your system
        pin_memory=True,
        drop_last=True # Important for SimCLR as NT-Xent assumes full batches for negatives
    )

    print("Initializing model...")
    encoder = get_resnet(name=ENCODER_NAME, pretrained=False) # Train from scratch for self-supervised
    # For ResNet-50, output dim is 2048
    # For ResNet-18, output dim is 512
    feature_dim = encoder.fc.in_features if hasattr(encoder.fc, 'in_features') else 512 # for resnet18, 2048 for resnet50
    if ENCODER_NAME == 'resnet50': feature_dim = 2048
    elif ENCODER_NAME == 'resnet18': feature_dim = 512
    
    projection_head = ProjectionHead(input_dim=feature_dim, output_dim=PROJECTION_DIM)
    
    model = SimCLR(encoder, projection_head).to(DEVICE)
    
    # Loss function
    criterion = NTXentLoss(temperature=CIFAR_TEMPERATURE, device=DEVICE).to(DEVICE) # Use CIFAR_TEMPERATURE

    # Optimizer
    # Paper uses LARS. AdamW is a common alternative if LARS is not tuned well or for smaller batches.
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # LARS optimizer parameters (from paper Appendix B.1)
    # LR = 0.3 * BatchSize / 256
    # Momentum = 0.9
    # Weight Decay = 1e-6 (applied to weights, not biases or BN params)
    
    # For LARS, it's common to exclude bias and batch norm parameters from weight decay.
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1: # Bias parameters
            param_biases.append(param)
        else:
            param_weights.append(param)
    
    # Note: SimCLR paper applies WD to all parameters for LARS.
    # "We use LARS optimizer [33] for all batch sizes, and apply it to all parameters including BN variables." (Appendix A)
    # "we apply weight decay to all parameters, including weights, biases and BN parameters." (Appendix B.1)
    
    optimizer = LARS(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)
    
    # Scheduler (Cosine annealing as per paper)
    # "We use a linear warmup for the first 10 epochs, followed by a cosine decay schedule without restarts"
    # Total steps = len(train_loader) * EPOCHS
    # Warmup steps = len(train_loader) * 10 (for ImageNet)
    
    # Simple cosine decay for now. For full replication, add linear warmup.
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*EPOCHS, eta_min=0)
    
    # For warmup + cosine decay:
    # Example of implementing warmup:
    # https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py#L249
    # For simplicity, let's use a standard CosineAnnealingLR for now.
    # A more accurate scheduler would be:
    # warmup_epochs = 10
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))
    # )
    # This LambdaLR needs to be called per epoch.
    # CosineAnnealingLR is per step if T_max is total steps.
    # If T_max = EPOCHS, then it's per epoch.
    # The paper's schedule is often implemented per step.
    
    # Let's use CosineAnnealingLR per epoch for simplicity here.
    # T_max = total number of epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0001) # eta_min is lr at the end

    print("Starting training...")
    model.train()
    global_step = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        for step, ((xis, xjs), _) in enumerate(train_loader): # Dataset returns (images_tuple, labels)
            xis = xis.to(DEVICE)
            xjs = xjs.to(DEVICE)

            optimizer.zero_grad()

            # Get projections
            zis, zjs = model((xis, xjs))

            # Calculate loss
            loss = criterion(zis, zjs)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if step % 50 == 0: # Log every 50 steps
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"End of Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step() # Update learning rate

        # Save model checkpoint (optional)
        if (epoch + 1) % 10 == 0: # Save every 10 epochs
            print(f"Saving model at epoch {epoch+1}")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'simclr_model_epoch_{epoch+1}.pth')

    print("Training finished.")
    print("Saving final model.")
    torch.save(model.state_dict(), 'simclr_final_model.pth')
    # For evaluation, you would typically take the encoder part:
    # torch.save(model.encoder.state_dict(), 'simclr_encoder_final.pth')


# --- Main execution ---
if __name__ == '__main__':
    # Set random seeds for reproducibility (optional)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        # CUDNN settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Can slow down if input sizes vary

    main_simclr()

