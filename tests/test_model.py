import unittest
import torch
import torch.nn as nn
import torchvision.models as torchvision_models # For type checking

# Adjust sys.path to allow importing from the parent directory
import sys
import os # Added for robust path joining
from pathlib import Path # Pathlib is good, but os.path.join is also common

# Ensure the path adjustment is robust
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model import ProjectionHead, SimCLRModel

class TestModelComponents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_projection_head_output(self):
        batch_size, input_d, hidden_d, output_d = 32, 2048, 2048, 128
        head = ProjectionHead(input_dim=input_d, hidden_dim=hidden_d, output_dim=output_d).to(self.device)
        head.eval() # Set to eval mode
        dummy_input = torch.randn(batch_size, input_d).to(self.device)
        output = head(dummy_input)
        
        self.assertEqual(output.shape, (batch_size, output_d), "Output shape mismatch.")
        # Check if output is different from a simple slice of input (if output_d < input_d)
        if output_d < input_d:
            self.assertFalse(torch.allclose(output, dummy_input[:, :output_d]), 
                             "Output seems to be a slice of input.")
        # A more general check for transformation
        self.assertFalse(torch.allclose(output, dummy_input) and input_d == output_d,
                             "Output should be transformed, not identical to input even if dims match.")


    def test_simclr_model_instantiation_resnet18(self):
        out_dim = 128
        model = SimCLRModel(base_model='resnet18', out_dim=out_dim).to(self.device)
        
        self.assertIsInstance(model.backbone, torchvision_models.resnet.ResNet, "Backbone is not a ResNet instance.")
        self.assertIsInstance(model.backbone.fc, nn.Identity, "Backbone fc layer is not nn.Identity.")
        self.assertIsInstance(model.projection_head, ProjectionHead, "Projection head is not correct type.")
        # Check if feature_dim for projection head was 512 (specific to ResNet18)
        # Accessing internal layers of nn.Sequential like this is a bit fragile,
        # but common for checking dimensions if no direct attribute is exposed.
        self.assertEqual(model.projection_head.projection_head[0].in_features, 512,
                         "Projection head input dim mismatch for ResNet18.")

    def test_simclr_model_instantiation_resnet50(self):
        out_dim = 128
        model = SimCLRModel(base_model='resnet50', out_dim=out_dim).to(self.device)
        
        self.assertIsInstance(model.backbone, torchvision_models.resnet.ResNet, "Backbone is not a ResNet instance.")
        self.assertIsInstance(model.backbone.fc, nn.Identity, "Backbone fc layer is not nn.Identity.")
        self.assertIsInstance(model.projection_head, ProjectionHead, "Projection head is not correct type.")
        # Check if feature_dim for projection head was 2048 (specific to ResNet50)
        self.assertEqual(model.projection_head.projection_head[0].in_features, 2048,
                         "Projection head input dim mismatch for ResNet50.")

    def test_simclr_model_forward_pass(self):
        batch_size, out_dim = 4, 128
        # Using resnet18, feature_dim is 512
        feature_dim_expected = 512 
        model = SimCLRModel(base_model='resnet18', out_dim=out_dim).to(self.device)
        model.eval() # Set to eval mode for consistent behavior

        dummy_input = torch.randn(batch_size, 3, 32, 32).to(self.device) # Typical CIFAR10 size
        features, projections = model(dummy_input)
        
        self.assertIsInstance(features, torch.Tensor, "Features output is not a tensor.")
        self.assertIsInstance(projections, torch.Tensor, "Projections output is not a tensor.")
        
        self.assertEqual(features.shape, (batch_size, feature_dim_expected), "Features shape mismatch.")
        self.assertEqual(projections.shape, (batch_size, out_dim), "Projections shape mismatch.")
        
        # Check L2 normalization of projections (recently added requirement)
        for i in range(batch_size):
            norm = torch.norm(projections[i], p=2)
            self.assertAlmostEqual(norm.item(), 1.0, places=5, 
                                   msg=f"Projections row {i} should be L2 normalized.")
            
        # Ensure features and projections are different.
        # Given feature_dim_expected (512) != out_dim (128), their shapes are different.
        # A check for non-equality of values is still good.
        # Taking a slice of features to compare with projections.
        if feature_dim_expected >= out_dim:
             self.assertFalse(torch.allclose(features[:, :out_dim], projections, atol=1e-5),
                             "Projections should be different from the first part of features.")
        else: # Should not happen with current ResNet settings
             self.assertFalse(torch.allclose(features, projections[:, :feature_dim_expected], atol=1e-5),
                             "Features should be different from the first part of projections.")


    def test_unsupported_base_model(self):
        with self.assertRaises(ValueError) as context:
            SimCLRModel(base_model='resnet_unsupported', out_dim=128)
        self.assertTrue('Unsupported model' in str(context.exception) or 
                        'Unsupported ResNet' in str(context.exception))


if __name__ == '__main__':
    unittest.main()
