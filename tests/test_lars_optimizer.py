import unittest
import torch
import torch.nn as nn
import os
import sys

# Adjust path to import from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import LARS 

class TestLARSOptimizer(unittest.TestCase):

    def test_parameter_update(self):
        model = nn.Linear(10, 1, bias=True) # Simple model
        # Initialize weights and bias to known values for easier checking
        with torch.no_grad(): # Ensure initialization is not tracked
            nn.init.ones_(model.weight)
            nn.init.zeros_(model.bias)
        
        optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, trust_coefficient=0.001)
        
        # Store original parameters
        original_weight = model.weight.clone().detach()
        original_bias = model.bias.clone().detach()

        # Dummy forward and backward pass
        dummy_input = torch.randn(1, 10)
        dummy_output = model(dummy_input)
        dummy_loss = dummy_output.sum()
        dummy_loss.backward() # Populates .grad

        # Ensure gradients exist before optimizer step
        self.assertIsNotNone(model.weight.grad)
        self.assertIsNotNone(model.bias.grad)
        self.assertTrue(torch.sum(torch.abs(model.weight.grad)) > 0)
        self.assertTrue(torch.sum(torch.abs(model.bias.grad)) > 0)

        optimizer.step()

        self.assertFalse(torch.equal(model.weight.data, original_weight), "Weights should change")
        self.assertFalse(torch.equal(model.bias.data, original_bias), "Bias should change")

    def test_weight_decay_effect_on_zero_grad(self):
        # Test on a parameter with zero gradient
        # LARS implementation from g.py applies weight decay by grad.add_(p.data, alpha=weight_decay)
        # So, if grad is zero, grad becomes weight_decay * p.data.
        # Then this modified grad is used in momentum and update.
        # p.data.add_(buf, alpha=-lr) where buf includes local_lr * (wd * p.data)
        
        param_val = 1.0
        param = nn.Parameter(torch.full((5,), param_val))
        original_param_abs_sum = torch.sum(torch.abs(param.data.clone().detach()))
        
        # Using momentum=0 to isolate WD effect more directly on the update via grad modification
        optimizer = LARS([param], lr=0.01, momentum=0.0, weight_decay=1e-2, trust_coefficient=0.001) 
        
        # Zero gradient
        param.grad = torch.zeros_like(param)
        
        optimizer.step()
        
        # With weight decay, the parameter's effective gradient becomes wd * param.
        # The update will be -lr * local_lr * (wd * param).
        # So param should move towards zero if param_val is positive.
        self.assertTrue(torch.sum(torch.abs(param.data)) < original_param_abs_sum, 
                        "Parameter sum of abs values should decrease due to weight decay with zero gradient.")

    def test_momentum_buffer_updated(self):
        model = nn.Linear(5, 1)
        with torch.no_grad():
            nn.init.normal_(model.weight) # Random init
            nn.init.normal_(model.bias)

        optimizer = LARS(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
        
        dummy_input = torch.randn(1, 5)
        dummy_output = model(dummy_input)
        dummy_loss = dummy_output.sum()
        dummy_loss.backward()
        
        optimizer.step() # First step initializes momentum buffer
        
        param_state = optimizer.state[model.weight]
        self.assertTrue('momentum_buffer' in param_state)
        # Buffer should be non-zero if grad was non-zero
        if model.weight.grad is not None and torch.sum(torch.abs(model.weight.grad)) > 0:
             self.assertFalse(torch.all(param_state['momentum_buffer'] == 0))

        # Second step, buffer should change again if grad is still present
        old_buffer = param_state['momentum_buffer'].clone().detach()
        
        # New forward/backward pass
        optimizer.zero_grad() # Clear gradients before new backward pass
        dummy_output_2 = model(dummy_input) 
        dummy_loss_2 = dummy_output_2.sum()
        dummy_loss_2.backward()

        if model.weight.grad is not None and torch.sum(torch.abs(model.weight.grad)) > 0:
            optimizer.step()
            self.assertFalse(torch.equal(param_state['momentum_buffer'], old_buffer),
                            "Momentum buffer should change after a new step with new gradients.")


    def test_local_learning_rate_application(self):
        model = nn.Linear(3, 1, bias=False) # Bias=False for simplicity
        with torch.no_grad():
            model.weight.data = torch.tensor([[1.0, 1.0, 1.0]]) # Known weights
        
        optimizer = LARS(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0, trust_coefficient=0.001)

        original_weight_data = model.weight.data.clone().detach()

        dummy_input = torch.tensor([[1.0, 2.0, 3.0]]) # Leads to grad = [[1.0, 2.0, 3.0]] for weight
        dummy_output = model(dummy_input)
        dummy_loss = dummy_output.sum()
        dummy_loss.backward()
        
        self.assertIsNotNone(model.weight.grad)
        expected_grad = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertTrue(torch.allclose(model.weight.grad, expected_grad))

        param_norm_actual = torch.norm(model.weight.data)
        grad_norm_actual = torch.norm(model.weight.grad.data)
        
        trust_coeff = optimizer.param_groups[0]['trust_coefficient']
        eps = optimizer.param_groups[0]['eps']
        # wd = optimizer.param_groups[0]['weight_decay'] # wd is 0 in this test
        
        # LARS local_lr scaling factor
        expected_local_lr_scale_factor = trust_coeff * param_norm_actual / (grad_norm_actual + eps) # + wd * param_norm_actual simplified as wd=0
        
        optimizer.step()
        
        # The 'local_lr' in the LARS code is the scaling factor for the gradient in the momentum update.
        # buf.mul_(momentum).add_(grad, alpha=local_lr)
        # Then p.data.add_(buf, alpha=-lr)
        # So, if momentum is 0, update is: p.data = p.data - lr * local_lr_scale_factor * grad
        
        # Check that local_lr_scale_factor is not 1.0 (unless norms happen to make it so)
        self.assertNotAlmostEqual(expected_local_lr_scale_factor.item(), 1.0, places=5, 
                                  msg="Local LR scale factor should ideally not be 1.0 for this test setup.")
        
        # Calculate the expected updated weight if LARS logic is applied
        lars_expected_update = original_weight_data - optimizer.param_groups[0]['lr'] * expected_local_lr_scale_factor * model.weight.grad.data
        self.assertTrue(torch.allclose(model.weight.data, lars_expected_update),
                        "LARS update does not match expected calculation with local learning rate scaling.")

        # Compare against plain SGD update
        plain_sgd_update = original_weight_data - optimizer.param_groups[0]['lr'] * model.weight.grad.data
        self.assertFalse(torch.allclose(model.weight.data, plain_sgd_update) or torch.allclose(lars_expected_update, plain_sgd_update), 
                         "LARS update should differ from plain SGD update if local_lr_scale_factor is not 1.")


if __name__ == '__main__':
    unittest.main()
