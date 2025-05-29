import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import math
import os # For creating test data directory

# Adjust sys.path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import SimCLRTrainer, create_dataloader, LARS # NTXentLoss is also in train.py's scope
from model import SimCLRModel 
from loss import NTXentLoss # Explicitly import NTXentLoss if used for type checking criterion

class TestTrainerComponents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Dummy model needed for SimCLRTrainer
        cls.dummy_model = SimCLRModel(base_model='resnet18', out_dim=128).to(cls.device)
        cls.test_data_dir = './data_test_trainer' # For dataset specific tests
        os.makedirs(cls.test_data_dir, exist_ok=True)


    def test_trainer_init_cifar10_defaults(self):
        trainer = SimCLRTrainer(model=self.dummy_model, device=self.device, dataset='cifar10')
        self.assertEqual(trainer.batch_size, 512)
        self.assertEqual(trainer.learning_rate, 1.0) # base LR for SGD on CIFAR-10
        self.assertEqual(trainer.temperature, 0.5)
        self.assertEqual(trainer.epochs, 1000)
        self.assertIsInstance(trainer.optimizer, optim.SGD)
        self.assertIsInstance(trainer.criterion, NTXentLoss)
        self.assertEqual(trainer.criterion.temperature, 0.5)

    def test_trainer_init_imagenet_defaults(self):
        trainer = SimCLRTrainer(model=self.dummy_model, device=self.device, dataset='imagenet')
        self.assertEqual(trainer.batch_size, 4096)
        # self.learning_rate is the base LR before batch scaling for LARS
        self.assertEqual(trainer.learning_rate, 0.075) 
        self.assertEqual(trainer.temperature, 0.1)
        self.assertEqual(trainer.epochs, 100)
        self.assertIsInstance(trainer.optimizer, LARS)

    def test_trainer_init_optimizer_selection_override(self):
        trainer_sgd = SimCLRTrainer(model=self.dummy_model, device=self.device, dataset='cifar10', batch_size=256)
        self.assertIsInstance(trainer_sgd.optimizer, optim.SGD)
        
        trainer_lars = SimCLRTrainer(model=self.dummy_model, device=self.device, dataset='cifar10', batch_size=1024)
        self.assertIsInstance(trainer_lars.optimizer, LARS)

    def test_adjust_lr_warmup(self):
        base_lr = 1.0
        trainer = SimCLRTrainer(model=self.dummy_model, device=self.device, dataset='cifar10', 
                                batch_size=512, learning_rate=base_lr)
        num_batches = 100 
        warmup_epochs = 10
        
        # Epoch 0, first batch
        lr_epoch0_batch0 = trainer.adjust_learning_rate(epoch=0, batch_idx=0, num_batches=num_batches)
        optimizer_lr_e0b0 = trainer.optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(lr_epoch0_batch0, optimizer_lr_e0b0)
        # Expected: base_lr * (0*100 + 0) / (10*100) = 0, but not exactly due to +1 in implementation potentially
        # The formula is: self.learning_rate * (epoch * num_batches + batch_idx) / (warmup_epochs * num_batches)
        # If batch_idx is 0, it should be 0. If batch_idx is 1, it's base_lr * 1 / (10*100)
        # The test uses batch_idx=0, so it should be 0 or very small.
        # The code uses (epoch * num_batches + batch_idx), if batch_idx starts at 0, then for first step it's 0.
        # Let's test with batch_idx=1 for a non-zero check
        lr_epoch0_batch1 = trainer.adjust_learning_rate(epoch=0, batch_idx=1, num_batches=num_batches)
        self.assertLess(lr_epoch0_batch1, base_lr)
        self.assertGreater(lr_epoch0_batch1, 0)


        # Epoch 5 (mid-warmup), first batch
        lr_epoch5_batch0 = trainer.adjust_learning_rate(epoch=5, batch_idx=0, num_batches=num_batches)
        self.assertGreater(lr_epoch5_batch0, lr_epoch0_batch1) # batch_idx=0 vs batch_idx=1 for previous
        
        # End of warmup (epoch 9, last batch_idx)
        lr_epoch9_end = trainer.adjust_learning_rate(epoch=warmup_epochs-1, batch_idx=num_batches-1, num_batches=num_batches)
        
        # Expected LR at this point (before sqrt scaling if LARS)
        expected_lr_at_warmup_end = base_lr * ((warmup_epochs-1) * num_batches + (num_batches-1)) / (warmup_epochs * num_batches)
        # This should be very close to base_lr. The formula in paper implies it reaches base_lr *after* warmup.
        # The current implementation has it slightly less than base_lr at the very end of warmup.
        # For SGD (no sqrt scaling here as bs=512 for cifar10 test)
        self.assertAlmostEqual(lr_epoch9_end, expected_lr_at_warmup_end, delta=base_lr*0.05)


    def test_adjust_lr_cosine_annealing(self):
        base_lr = 1.0
        total_epochs = 100 # shorter for test
        warmup_epochs = 10
        trainer = SimCLRTrainer(model=self.dummy_model, device=self.device, dataset='cifar10', 
                                learning_rate=base_lr, epochs=total_epochs, batch_size=512)
        num_batches = 100
        
        # LR at start of cosine annealing (epoch 10)
        lr_epoch10 = trainer.adjust_learning_rate(epoch=warmup_epochs, batch_idx=0, num_batches=num_batches)
        expected_lr_epoch10 = base_lr # After warmup, before sqrt scaling
        self.assertAlmostEqual(lr_epoch10, expected_lr_epoch10, delta=1e-7)
        
        # LR mid-annealing
        mid_epoch = warmup_epochs + (total_epochs - warmup_epochs) // 2
        lr_mid_anneal = trainer.adjust_learning_rate(epoch=mid_epoch, batch_idx=0, num_batches=num_batches)
        self.assertLess(lr_mid_anneal, lr_epoch10)
        
        # LR towards end of annealing (last epoch)
        lr_last_epoch = trainer.adjust_learning_rate(epoch=total_epochs-1, batch_idx=0, num_batches=num_batches)
        self.assertLess(lr_last_epoch, lr_mid_anneal)
        # SimCLRTrainer.scheduler has eta_min=0, so LR should go to 0 (or very close for last step of cosine)
        expected_lr_last_epoch = base_lr * 0.5 * (1 + math.cos(math.pi * (total_epochs - 1 - warmup_epochs) / (total_epochs - warmup_epochs)))
        self.assertAlmostEqual(lr_last_epoch, expected_lr_last_epoch, delta=1e-5)


    def test_adjust_lr_sqrt_scaling(self):
        base_lr_imagenet = 0.075
        trainer = SimCLRTrainer(model=self.dummy_model, device=self.device, dataset='imagenet') # bs=4096, LARS
        self.assertIsInstance(trainer.optimizer, LARS) 
        
        num_batches = 100
        epoch_after_warmup = 10 # Example epoch after warmup
        
        # Expected LR from cosine annealing part
        lr_cosine_part = base_lr_imagenet * 0.5 * (1 + math.cos(math.pi * (epoch_after_warmup - 10) / (trainer.epochs - 10)))
        # Expected final LR with sqrt scaling
        expected_lr_with_sqrt_scale = lr_cosine_part * math.sqrt(trainer.batch_size / 256)
        
        actual_lr = trainer.adjust_learning_rate(epoch=epoch_after_warmup, batch_idx=0, num_batches=num_batches)
        self.assertAlmostEqual(actual_lr, expected_lr_with_sqrt_scale, places=5)

    @patch('train.datasets.CIFAR10') # Mock torchvision.datasets.CIFAR10
    def test_create_dataloader_basic_cifar10(self, mock_cifar10):
        # Setup mock CIFAR10 to return a dummy dataset
        dummy_tensor_dataset = TensorDataset(torch.randn(10, 3, 32, 32), torch.randint(0, 10, (10,)))
        mock_cifar10.return_value = dummy_tensor_dataset
        
        # Call create_dataloader, it will use the mocked CIFAR10
        # The 'root' path in create_dataloader is hardcoded, but mock bypasses actual disk I/O
        dataloader = create_dataloader(dataset_name='cifar10', batch_size=4, one_idx_class=None, num_workers=0)
        
        mock_cifar10.assert_called_once() # Check if datasets.CIFAR10 was called
        self.assertIsInstance(dataloader, DataLoader)
        self.assertTrue(dataloader.drop_last)
        self.assertEqual(dataloader.batch_size, 4)
        self.assertEqual(len(dataloader.dataset), 10) # Length of our dummy dataset

    @patch('train.datasets.CIFAR10')
    @patch('train.get_subclass_dataset') # Mock get_subclass_dataset
    def test_create_dataloader_one_idx_class_cifar10(self, mock_get_subclass_dataset, mock_cifar10):
        dummy_full_dataset = TensorDataset(torch.randn(20, 3, 32, 32), torch.randint(0, 10, (20,)))
        mock_cifar10.return_value = dummy_full_dataset
        
        dummy_subset_dataset = TensorDataset(torch.randn(5, 3, 32, 32), torch.randint(0, 1, (5,))) # Simulate a subset
        mock_get_subclass_dataset.return_value = dummy_subset_dataset

        class_index_to_filter = 1
        dataloader_one_class = create_dataloader(dataset_name='cifar10', batch_size=4, one_idx_class=class_index_to_filter, num_workers=0)
        
        mock_cifar10.assert_called_once()
        mock_get_subclass_dataset.assert_called_once_with(dummy_full_dataset, classes=class_index_to_filter)
        self.assertIsInstance(dataloader_one_class.dataset, TensorDataset) # It's the mocked subset
        self.assertEqual(len(dataloader_one_class.dataset), 5) # Length of the dummy subset


    @classmethod
    def tearDownClass(cls):
        # Clean up the test data directory
        if os.path.exists(cls.test_data_dir):
            # Remove dummy files/folders if any were created by actual dataset calls (though mocked here)
            for f_name in os.listdir(cls.test_data_dir):
                os.remove(os.path.join(cls.test_data_dir, f_name))
            os.rmdir(cls.test_data_dir)

if __name__ == '__main__':
    unittest.main()
