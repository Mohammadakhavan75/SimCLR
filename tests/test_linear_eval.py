import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Adjust sys.path to allow importing from the parent directory
import sys
import os # Added for robust path joining
from pathlib import Path # Pathlib is good, but os.path.join is also common

# Ensure the path adjustment is robust
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model import SimCLRModel 
from linear_eval import LinearEvaluator, create_eval_dataloaders, LinearClassifier

class TestLinearEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # This path might need adjustment if tests are run from a different root
        cls.cifar10_data_root = './data' # Default root for torchvision datasets

    def test_create_eval_dataloaders_cifar10(self):
        # Test if CIFAR-10 dataloaders are created correctly.
        # This test will attempt to download CIFAR-10 if not found in root './data'
        try:
            # Ensure the data root directory exists for downloading, if needed
            os.makedirs(self.cifar10_data_root, exist_ok=True)
            train_loader, test_loader, num_classes = create_eval_dataloaders(
                dataset_name='cifar10', batch_size=32 #, data_dir=self.cifar10_data_root # create_eval_dataloaders uses hardcoded './data'
            )
            self.assertIsInstance(train_loader, DataLoader)
            self.assertIsInstance(test_loader, DataLoader)
            self.assertEqual(num_classes, 10)
            self.assertTrue(len(train_loader.dataset) > 0)
            self.assertTrue(len(test_loader.dataset) > 0)
            
            # Check a batch
            images, labels = next(iter(train_loader))
            self.assertEqual(images.shape[0], 32) # Batch size
            self.assertEqual(images.shape[1], 3)  # Channels
            self.assertEqual(labels.shape[0], 32)

        except Exception as e:
            # Blanket skip for any dataset related issues in CI, can be refined
            if "Dataset not found" in str(e) or "download" in str(e).lower() or "timed out" in str(e).lower():
                self.skipTest(f"CIFAR-10 dataset download failed or dataset not found: {e}")
            else:
                raise e


    def test_linear_evaluator_initialization(self):
        dummy_simclr_model = SimCLRModel(base_model='resnet18', out_dim=128).to(self.device)
        num_classes = 10 # Example for CIFAR-10
        
        evaluator = LinearEvaluator(dummy_simclr_model, num_classes, self.device)
        
        # Check if encoder is frozen
        for param in evaluator.encoder.parameters():
            self.assertFalse(param.requires_grad, "Encoder parameters should be frozen.")
            
        # Check if classifier is trainable
        for param in evaluator.classifier.parameters():
            self.assertTrue(param.requires_grad, "Classifier parameters should be trainable.")
            
        # Check optimizer and scheduler types (as per recent modifications)
        self.assertIsInstance(evaluator.optimizer, optim.SGD)
        self.assertIsInstance(evaluator.scheduler, optim.lr_scheduler.MultiStepLR)
        # Check some optimizer params
        self.assertEqual(evaluator.optimizer.defaults['lr'], 0.1)
        self.assertEqual(evaluator.optimizer.defaults['momentum'], 0.9)
        # Check some scheduler params
        self.assertEqual(evaluator.scheduler.milestones, [30, 60, 80])
        self.assertEqual(evaluator.scheduler.gamma, 0.1)


    def test_linear_evaluator_train_and_evaluate_flow(self):
        # Use a very small SimCLRModel for speed, e.g. based on a few conv layers if possible,
        # but SimCLRModel is hardcoded to resnet18/50. We'll use resnet18.
        # Feature dim for resnet18 is 512.
        simclr_model_eval = SimCLRModel(base_model='resnet18', out_dim=128).to(self.device)
        num_classes = 2 # Simplified number of classes for faster test
        
        evaluator = LinearEvaluator(simclr_model_eval, num_classes, self.device)
        
        # Create truly minimal dummy data and DataLoaders
        # Encoder input for resnet18 is typically (N, 3, H, W), e.g. (N, 3, 32, 32) for cifar
        # The evaluator.encoder is the backbone.
        dummy_train_images = torch.randn(8, 3, 32, 32, device=self.device) 
        dummy_train_labels = torch.randint(0, num_classes, (8,), device=self.device)
        dummy_test_images = torch.randn(4, 3, 32, 32, device=self.device)
        dummy_test_labels = torch.randint(0, num_classes, (4,), device=self.device)

        train_dataset = TensorDataset(dummy_train_images, dummy_train_labels)
        test_dataset = TensorDataset(dummy_test_images, dummy_test_labels)

        # Use a small batch size for testing
        train_loader = DataLoader(train_dataset, batch_size=4)
        test_loader = DataLoader(test_dataset, batch_size=4)

        # Store original classifier params to check if they change
        original_classifier_params = [p.clone().detach() for p in evaluator.classifier.parameters()]

        # Train for a minimal number of epochs (e.g., 1 or 2)
        # Note: train_linear_classifier prints logs every 10 epochs. For 1-2 epochs, no logs.
        best_val_acc = evaluator.train_linear_classifier(train_loader, test_loader, epochs=2) 
        
        self.assertIsInstance(best_val_acc, float)
        self.assertGreaterEqual(best_val_acc, 0.0)
        self.assertLessEqual(best_val_acc, 100.0)

        # Check if classifier parameters changed
        params_changed = False
        for i, param in enumerate(evaluator.classifier.parameters()):
            if not torch.equal(param.data, original_classifier_params[i]):
                params_changed = True
                break
        self.assertTrue(params_changed, "Classifier parameters should change after training.")

        # Evaluate
        final_acc = evaluator.evaluate(test_loader)
        self.assertIsInstance(final_acc, float)
        self.assertGreaterEqual(final_acc, 0.0)
        self.assertLessEqual(final_acc, 100.0)

if __name__ == '__main__':
    unittest.main()
