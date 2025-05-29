import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import os
import math # for LARS optimizer

from model import SimCLRModel
from loss import NTXentLoss
from data_augmentation import SimCLRAugmentation

from torch.utils.data import Subset

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

class SimCLRTrainer:
    """SimCLR training class - Paper Exact Configuration
    
    Handles the complete training pipeline with paper-exact hyperparameters.
    """
    def __init__(self, model, device, dataset='imagenet', batch_size=None, 
                 learning_rate=None, temperature=None, weight_decay=1e-4, epochs=None, one_idx_class=None):
        self.model = model.to(device)
        self.device = device
        self.dataset = dataset
        self.one_idx_class = one_idx_class
        # For replicating SimCLR paper results, one_idx_class should be None to use the full dataset.
        
        # Paper-exact configurations
        if dataset == 'cifar10':
            self.batch_size = batch_size or 512
            self.learning_rate = learning_rate or 1.0
            self.temperature = temperature or 0.5
            self.epochs = epochs or 1000
        else:  # ImageNet
            self.batch_size = batch_size or 4096
            self.learning_rate = learning_rate or 0.075  # With sqrt scaling
            self.temperature = temperature or 0.1
            self.epochs = epochs or 100
        
        self.weight_decay = weight_decay
        
        # Loss function
        self.criterion = NTXentLoss(temperature=self.temperature)
        
        # Optimizer - Paper uses LARS for large batch sizes
        if self.batch_size >= 1024:
            self.optimizer = LARS(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001
            )
        else:
            # For smaller batches, use SGD as in paper
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        
        # Learning rate scheduler - Cosine annealing as in paper
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=0
        )
    
    def adjust_learning_rate(self, epoch, batch_idx, num_batches):
        """Learning rate warmup and scaling as in paper"""
        warmup_epochs = 10
        if epoch < warmup_epochs:
            # Linear warmup
            lr = self.learning_rate * (epoch * num_batches + batch_idx) / (warmup_epochs * num_batches)
        else:
            # Cosine annealing
            lr = self.learning_rate * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (self.epochs - warmup_epochs)))
        
        # Apply sqrt scaling for large batch sizes
        if self.batch_size >= 1024:
            lr = lr * math.sqrt(self.batch_size / 256)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def train_epoch(self, dataloader, epoch, num_batches):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # Adjust learning rate
            current_lr = self.adjust_learning_rate(epoch, batch_idx, num_batches)
            
            # images is a tuple of two augmented views
            view1, view2 = images
            view1, view2 = view1.to(self.device), view2.to(self.device)
            
            # Forward pass
            _, z1 = self.model(view1)
            _, z2 = self.model(view2)
            
            # Compute contrastive loss
            loss = self.criterion(z1, z2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, dataloader, save_dir="checkpoints"):
        """Complete training loop with paper configuration"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training Configuration (Paper Exact):")
        print(f"Dataset: {self.dataset}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Temperature: {self.temperature}")
        print(f"Epochs: {self.epochs}")
        print(f"Weight Decay: {self.weight_decay}")
        print(f"Optimizer: {'LARS' if self.batch_size >= 1024 else 'SGD'}")
        print(f"one_idx_class: {self.one_idx_class}")
        num_batches = len(dataloader)
        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(dataloader, epoch, num_batches)
            
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                checkpoint_path = os.path.join(save_dir, f"idx_{self.one_idx_class}_simclr_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'one_idx_class': self.one_idx_class,
                    'config': {
                        'dataset': self.dataset,
                        'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate,
                        'temperature': self.temperature
                    }
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")



def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def create_dataloader(dataset_name='cifar10', batch_size=512, num_workers=4, one_idx_class=None):
    """Create dataloader with paper-exact augmentations"""
    if dataset_name == 'cifar10':
        transform = SimCLRAugmentation(size=32, dataset='cifar10', s=0.5)  # Paper uses s=0.5 for CIFAR-10
        dataset = datasets.CIFAR10(
            root='D:/Datasets/data/', train=True, download=True, transform=transform
        )

    elif dataset_name == 'imagenet':
        transform = SimCLRAugmentation(size=224, dataset='imagenet', s=1.0)
        dataset = datasets.ImageNet(
            root='D:/Datasets/data/imagenet', split='train', transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if one_idx_class:
        # For replicating SimCLR paper results, one_idx_class should typically be None.
        # This filter is for specific debugging or analysis on a single class.
        dataset = get_subclass_dataset(dataset, classes=one_idx_class)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    return dataloader

if __name__ == "__main__":
    # Paper-exact training configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # CIFAR-10 configuration (paper exact)
    one_idx_class = 3 # Set to None for full dataset, or specify a class index for filtering.
                          # For replicating SimCLR paper results, this should be None.
    dataset_name = 'cifar10'
    model = SimCLRModel(base_model='resnet18', out_dim=128)
    
    # Create trainer with paper-exact settings
    trainer = SimCLRTrainer(
        model=model,
        device=device,
        dataset=dataset_name
        # All other parameters will use paper defaults
    )
    
    # Create dataloader with paper-exact settings
    dataloader = create_dataloader(
        dataset_name=dataset_name,
        batch_size=trainer.batch_size,
        num_workers=4,
        one_idx_class=one_idx_class  
    )
    
    # Train the model
    trainer.train(dataloader)