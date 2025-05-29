#!/usr/bin/env python3
"""
Verification script for SimCLR implementation
Tests all components to ensure paper-exact configuration
"""

import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader

from model import SimCLRModel
from loss import NTXentLoss
from data_augmentation import SimCLRAugmentation
from train import SimCLRTrainer, LARS

def test_data_augmentation():
    """Test data augmentation pipeline"""
    print("Testing data augmentation...")
    
    # Test CIFAR-10 augmentation
    aug_cifar = SimCLRAugmentation(size=32, dataset='cifar10', s=0.5)
    
    # Create dummy image
    dummy_img = torch.randint(0, 255, (32, 32, 3), dtype=torch.uint8)
    from PIL import Image
    dummy_pil = Image.fromarray(dummy_img.numpy())
    
    # Apply augmentation
    view1, view2 = aug_cifar(dummy_pil)
    
    assert view1.shape == (3, 32, 32), f"Expected (3, 32, 32), got {view1.shape}"
    assert view2.shape == (3, 32, 32), f"Expected (3, 32, 32), got {view2.shape}"
    print("✓ Data augmentation working correctly")

def test_model():
    """Test SimCLR model"""
    print("Testing model...")
    
    # Test ResNet-18
    model = SimCLRModel(base_model='resnet18', out_dim=128)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 32, 32)
    features, projections = model(dummy_input)
    
    assert features.shape == (4, 512), f"Expected (4, 512), got {features.shape}"
    assert projections.shape == (4, 128), f"Expected (4, 128), got {projections.shape}"
    print("✓ Model working correctly")

def test_loss():
    """Test NT-Xent loss"""
    print("Testing NT-Xent loss...")
    
    loss_fn = NTXentLoss(temperature=0.5)
    
    # Create dummy projections
    batch_size = 4
    proj_dim = 128
    z_i = torch.randn(batch_size, proj_dim)
    z_j = torch.randn(batch_size, proj_dim)
    
    # Compute loss
    loss = loss_fn(z_i, z_j)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"
    print("✓ NT-Xent loss working correctly")

def test_lars_optimizer():
    """Test LARS optimizer"""
    print("Testing LARS optimizer...")
    
    model = SimCLRModel(base_model='resnet18', out_dim=128)
    optimizer = LARS(model.parameters(), lr=1.0, momentum=0.9, weight_decay=1e-6)
    
    # Test optimization step
    dummy_input = torch.randn(2, 3, 32, 32)
    features, projections = model(dummy_input)
    loss = projections.sum()  # Dummy loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("✓ LARS optimizer working correctly")

def test_paper_configurations():
    """Test paper-exact configurations"""
    print("Testing paper-exact configurations...")
    
    device = torch.device('cpu')  # Use CPU for testing
    model = SimCLRModel(base_model='resnet18', out_dim=128)
    
    # Test CIFAR-10 configuration
    trainer_cifar = SimCLRTrainer(
        model=model,
        device=device,
        dataset='cifar10'
    )
    
    assert trainer_cifar.batch_size == 512, f"Expected batch_size=512, got {trainer_cifar.batch_size}"
    assert trainer_cifar.learning_rate == 1.0, f"Expected lr=1.0, got {trainer_cifar.learning_rate}"
    assert trainer_cifar.temperature == 0.5, f"Expected temp=0.5, got {trainer_cifar.temperature}"
    assert trainer_cifar.epochs == 1000, f"Expected epochs=1000, got {trainer_cifar.epochs}"
    
    # Test ImageNet configuration
    trainer_imagenet = SimCLRTrainer(
        model=model,
        device=device,
        dataset='imagenet'
    )
    
    assert trainer_imagenet.batch_size == 4096, f"Expected batch_size=4096, got {trainer_imagenet.batch_size}"
    assert trainer_imagenet.learning_rate == 0.3, f"Expected lr=0.3, got {trainer_imagenet.learning_rate}"
    assert trainer_imagenet.temperature == 0.07, f"Expected temp=0.07, got {trainer_imagenet.temperature}"
    assert trainer_imagenet.epochs == 100, f"Expected epochs=100, got {trainer_imagenet.epochs}"
    
    print("✓ Paper-exact configurations verified")

def test_integration():
    """Integration test with small batch"""
    print("Testing integration with small batch...")
    
    device = torch.device('cpu')
    model = SimCLRModel(base_model='resnet18', out_dim=128)
    
    # Create small dataloader
    transform = SimCLRAugmentation(size=32, dataset='cifar10', s=0.5)
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    
    trainer = SimCLRTrainer(
        model=model,
        device=device,
        dataset='cifar10',
        batch_size=8,
        epochs=1
    )
    
    # Test one epoch
    try:
        avg_loss = trainer.train_epoch(dataloader, epoch=0)
        assert avg_loss > 0, "Average loss should be positive"
        print(f"✓ Integration test passed. Average loss: {avg_loss:.4f}")
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        raise

def main():
    """Run all tests"""
    print("=== SimCLR Implementation Verification ===\n")
    
    tests = [
        test_data_augmentation,
        test_model,
        test_loss,
        test_lars_optimizer,
        test_paper_configurations,
        test_integration
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
        print()
    
    print(f"=== Results: {passed}/{len(tests)} tests passed ===")
    
    if passed == len(tests):
        print("🎉 All tests passed! Implementation is ready for training.")
    else:
        print("❌ Some tests failed. Please fix the issues before training.")

if __name__ == "__main__":
    main() 