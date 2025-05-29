import unittest
from unittest.mock import patch, MagicMock, call
import argparse # To help mock parse_args return value
import torch # For torch.device in ANY comparison

# Adjust sys.path
import sys
import os # For robust path joining
from pathlib import Path 
# Ensure the path adjustment is robust
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the main function or script.
import main as main_script 

class TestMainScript(unittest.TestCase):

    # Default device for ANY comparisons if needed
    # device_any = unittest.mock.ANY # This would be torch.device(...) which is tricky for ANY
    # Instead, we can capture and check type, or rely on the mocked functions not to care.
    # For now, assuming main determines device and passes it. Mocked functions won't use it.

    @patch('main.SimCLRTrainer')
    @patch('main.SimCLRModel')
    @patch('main.create_dataloader')
    @patch('main.torch.device') # Mock torch.device to control its return value
    def test_pretrain_mode_defaults(self, mock_torch_device, mock_create_dataloader, mock_SimCLRModel, mock_SimCLRTrainer):
        # Mock return values
        mock_device_instance = torch.device("cpu") # Example device
        mock_torch_device.return_value = mock_device_instance

        mock_model_instance = MagicMock()
        mock_SimCLRModel.return_value = mock_model_instance
        
        # Configure the trainer instance to have a batch_size attribute, as main.py uses it
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.batch_size = 512 # Default from SimCLRTrainer for cifar10
        mock_SimCLRTrainer.return_value = mock_trainer_instance
        
        mock_dataloader_instance = MagicMock()
        mock_create_dataloader.return_value = mock_dataloader_instance

        # Simulate command line arguments
        args = argparse.Namespace(
            mode='pretrain', 
            dataset='cifar10', # Default
            model='resnet18',   # Default
            checkpoint=None,
            batch_size=None,    # Will use trainer's default
            lr=None,
            temperature=None,
            epochs=None,
            one_idx_class=None
        )
        # Patch argparse to return these specific args
        with patch('argparse.ArgumentParser.parse_args', return_value=args):
            main_script.main()

        mock_torch_device.assert_called_once_with("cuda" if torch.cuda.is_available() else "cpu")
        mock_SimCLRModel.assert_called_once_with(base_model='resnet18', out_dim=128)
        mock_SimCLRTrainer.assert_called_once_with(
            model=mock_model_instance,
            device=mock_device_instance, 
            dataset='cifar10',
            batch_size=None, 
            learning_rate=None,
            temperature=None,
            epochs=None,
            one_idx_class=None
        )
        mock_create_dataloader.assert_called_once_with(
            dataset_name='cifar10',
            batch_size=mock_trainer_instance.batch_size, 
            one_idx_class=None
        )
        mock_trainer_instance.train.assert_called_once_with(mock_dataloader_instance)

    @patch('main.LinearEvaluator')
    @patch('main.create_eval_dataloaders')
    @patch('main.torch.load')
    @patch('main.SimCLRModel')
    @patch('main.torch.device')
    def test_linear_eval_mode(self, mock_torch_device, mock_SimCLRModel, mock_torch_load, 
                              mock_create_eval_dataloaders, mock_LinearEvaluator):
        mock_device_instance = torch.device("cpu")
        mock_torch_device.return_value = mock_device_instance

        mock_model_instance = MagicMock()
        mock_SimCLRModel.return_value = mock_model_instance
        
        mock_evaluator_instance = MagicMock()
        mock_LinearEvaluator.return_value = mock_evaluator_instance
        
        mock_train_loader, mock_test_loader, num_classes = MagicMock(), MagicMock(), 10
        mock_create_eval_dataloaders.return_value = (mock_train_loader, mock_test_loader, num_classes)
        
        mock_checkpoint_data = {'model_state_dict': 'dummy_state', 'config': {'param': 1}}
        mock_torch_load.return_value = mock_checkpoint_data

        args = argparse.Namespace(
            mode='linear_eval', 
            dataset='cifar10', 
            model='resnet50',
            checkpoint='path/to/checkpoint.pth',
            batch_size=None, lr=None, temperature=None, epochs=None,
            one_idx_class=None 
        )
        with patch('argparse.ArgumentParser.parse_args', return_value=args):
            main_script.main()

        mock_SimCLRModel.assert_called_once_with(base_model='resnet50', out_dim=128)
        mock_torch_load.assert_called_once_with('path/to/checkpoint.pth', map_location=mock_device_instance)
        mock_model_instance.load_state_dict.assert_called_once_with('dummy_state')
        mock_create_eval_dataloaders.assert_called_once_with('cifar10')
        mock_LinearEvaluator.assert_called_once_with(mock_model_instance, num_classes, mock_device_instance)
        mock_evaluator_instance.train_linear_classifier.assert_called_once_with(mock_train_loader, mock_test_loader, epochs=100)
        mock_evaluator_instance.evaluate.assert_called_once_with(mock_test_loader)

    @patch('main.torch.device') # Still need to mock device as it's called early
    def test_linear_eval_mode_no_checkpoint(self, mock_torch_device):
        mock_torch_device.return_value = torch.device("cpu")
        args = argparse.Namespace(
            mode='linear_eval', 
            dataset='cifar10', 
            model='resnet18',
            checkpoint=None, # No checkpoint
            batch_size=None, lr=None, temperature=None, epochs=None,
            one_idx_class=None
        )
        with patch('argparse.ArgumentParser.parse_args', return_value=args):
            with self.assertRaises(ValueError) as context:
                main_script.main()
            self.assertTrue("Checkpoint path required for linear evaluation" in str(context.exception))
            
    @patch('main.SimCLRTrainer')
    @patch('main.SimCLRModel')
    @patch('main.create_dataloader')
    @patch('main.torch.device')
    def test_pretrain_mode_specific_args(self, mock_torch_device, mock_create_dataloader, 
                                         mock_SimCLRModel, mock_SimCLRTrainer):
        mock_device_instance = torch.device("cpu")
        mock_torch_device.return_value = mock_device_instance

        mock_model_instance = MagicMock()
        mock_SimCLRModel.return_value = mock_model_instance
        
        mock_trainer_instance = MagicMock()
        # Simulate that the trainer instance will have its batch_size set based on input or default
        mock_trainer_instance.batch_size = 1024 # As per args
        mock_SimCLRTrainer.return_value = mock_trainer_instance
        
        mock_dataloader_instance = MagicMock()
        mock_create_dataloader.return_value = mock_dataloader_instance

        args = argparse.Namespace(
            mode='pretrain', 
            dataset='imagenet',
            model='resnet50',
            checkpoint=None,
            batch_size=1024,
            lr=0.5,
            temperature=0.2,
            epochs=200,
            one_idx_class=1
        )
        with patch('argparse.ArgumentParser.parse_args', return_value=args):
            main_script.main()

        mock_SimCLRModel.assert_called_once_with(base_model='resnet50', out_dim=128)
        mock_SimCLRTrainer.assert_called_once_with(
            model=mock_model_instance,
            device=mock_device_instance,
            dataset='imagenet',
            batch_size=1024,
            learning_rate=0.5,
            temperature=0.2,
            epochs=200,
            one_idx_class=1
        )
        # The batch_size for create_dataloader comes from the trainer instance
        mock_create_dataloader.assert_called_once_with(
            dataset_name='imagenet',
            batch_size=mock_trainer_instance.batch_size, 
            one_idx_class=1
        )
        mock_trainer_instance.train.assert_called_once_with(mock_dataloader_instance)

if __name__ == '__main__':
    unittest.main()
