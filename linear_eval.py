import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

from model import SimCLRModel

class LinearClassifier(nn.Module):
    """Linear classifier for evaluation"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)

class LinearEvaluator:
    """Linear evaluation of pre-trained SimCLR model
    
    This evaluates the quality of learned representations by training
    a linear classifier on top of frozen features.
    """
    def __init__(self, pretrained_model, num_classes, device):
        self.device = device
        self.num_classes = num_classes
        
        # Load pre-trained model and freeze it
        self.encoder = pretrained_model.backbone.to(device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Adjust size as needed
            feature_dim = self.encoder(dummy_input).shape[1]
        
        # Create linear classifier
        self.classifier = LinearClassifier(feature_dim, num_classes).to(device)
        
        # Optimizer for classifier only
        self.optimizer = optim.SGD(self.classifier.parameters(), lr=0.1, momentum=0.9)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 80], gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()
    
    def extract_features(self, dataloader):
        """Extract features using pre-trained encoder"""
        features = []
        labels = []
        
        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)
                feats = self.encoder(images)
                features.append(feats.cpu())
                labels.append(targets)
        
        return torch.cat(features), torch.cat(labels)
    
    def train_linear_classifier(self, train_loader, val_loader, epochs=100):
        """Train linear classifier on extracted features"""
        best_acc = 0
        
        for epoch in range(epochs):
            # Training
            self.classifier.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, targets in train_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features = self.encoder(images)
                
                # Forward pass through classifier
                outputs = self.classifier(features)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
            
            # Validation
            val_acc = self.evaluate(val_loader)
            train_acc = 100. * train_correct / train_total
            
            if val_acc > best_acc:
                best_acc = val_acc
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Best: {best_acc:.2f}%")
            
            self.scheduler.step()
        
        return best_acc
    
    def evaluate(self, dataloader):
        """Evaluate classifier"""
        self.classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Extract features
                features = self.encoder(images)
                
                # Classify
                outputs = self.classifier(features)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total

def create_eval_dataloaders(dataset_name='cifar10', batch_size=256):
    """Create dataloaders for linear evaluation"""
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)
        num_classes = 10
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, num_classes

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained SimCLR model
    model = SimCLRModel(base_model='resnet18', out_dim=128)
    checkpoint = torch.load('checkpoints/simclr_epoch_1000.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluation dataloaders
    train_loader, test_loader, num_classes = create_eval_dataloaders('cifar10')
    
    # Create evaluator
    evaluator = LinearEvaluator(model, num_classes, device)
    
    # Train and evaluate
    best_acc = evaluator.train_linear_classifier(train_loader, test_loader, epochs=100)
    final_acc = evaluator.evaluate(test_loader)
    
    print(f"Final test accuracy: {final_acc:.2f}%")