import unittest
import torch
import sys
import os

# Adjust path to import from parent directory if 'loss.py' is in the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from loss import NTXentLoss

class TestNTXentLoss(unittest.TestCase):

    def test_loss_basic_properties(self):
        loss_fn = NTXentLoss(temperature=0.5, device='cpu') # Specify device for consistency
        batch_size, proj_dim = 32, 128
        z_i = torch.randn(batch_size, proj_dim)
        z_j = torch.randn(batch_size, proj_dim)
        
        # L2 normalize as done in the model before loss
        z_i = torch.nn.functional.normalize(z_i, p=2, dim=-1)
        z_j = torch.nn.functional.normalize(z_j, p=2, dim=-1)

        loss = loss_fn(z_i, z_j)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0) # Loss should be positive

    def test_temperature_effect(self):
        batch_size, proj_dim = 32, 128
        z_i = torch.randn(batch_size, proj_dim)
        z_j = torch.randn(batch_size, proj_dim)
        z_i = torch.nn.functional.normalize(z_i, p=2, dim=-1)
        z_j = torch.nn.functional.normalize(z_j, p=2, dim=-1)

        loss_fn_low_temp = NTXentLoss(temperature=0.1, device='cpu')
        loss_low_temp = loss_fn_low_temp(z_i, z_j)

        loss_fn_high_temp = NTXentLoss(temperature=1.0, device='cpu')
        loss_high_temp = loss_fn_high_temp(z_i, z_j)
        
        self.assertNotEqual(loss_low_temp.item(), loss_high_temp.item())
        # Generally, lower temp leads to higher loss values
        self.assertGreater(loss_low_temp.item(), loss_high_temp.item())


    def test_perfectly_correlated_pairs(self):
        loss_fn = NTXentLoss(temperature=0.5, device='cpu')
        batch_size, proj_dim = 4, 128 # Smaller batch for easier debugging
        
        z_i = torch.randn(batch_size, proj_dim)
        z_i = torch.nn.functional.normalize(z_i, p=2, dim=-1)
        z_j = z_i.clone() # Perfect positive pairs

        loss_correlated = loss_fn(z_i, z_j)
        
        z_j_anti = -z_i.clone() # Perfect anti-correlation (cosine sim is -1 after normalization)
                                # Need to re-normalize if -z_i is not guaranteed to be normalized,
                                # but for L2 norm, ||-x|| = ||x||.
        loss_anti_correlated = loss_fn(z_i, z_j_anti)

        self.assertLess(loss_correlated.item(), loss_anti_correlated.item())
        # Theoretical loss for N=4, T=0.5 with perfect correlation (sim=1 for positive, others 0)
        # Positive logit: 1/0.5 = 2. Others: 0.
        # log_prob = pos_logit - log(sum(exp(all_logits_for_sample)))
        # Denominator for one sample: exp(2) + (2N-2)*exp(0) = exp(2) + (8-2)*1 = 7.389 + 6 = 13.389
        # log_prob = 2 - log(13.389) = 2 - 2.594 = -0.594. Loss = 0.594
        # This is an approximation assuming other vectors are orthogonal.
        # Given the setup, the other pairs in the batch are also perfectly correlated.
        # So, for z_i[0], negative examples include z_i[1], z_i[2], z_i[3], z_j[1], z_j[2], z_j[3].
        # Their similarity to z_i[0] is random.
        # print(f"Correlated Loss (N=4, T=0.5): {loss_correlated.item()}")
        # The important part is that it's lower than anti-correlated.
        # And should be relatively small. log(2N-1) = log(7) ~ 1.94 is a loose upper bound.
        self.assertLess(loss_correlated.item(), 2.0) # Check it's not excessively high

    def test_batch_size_invariance(self):
        proj_dim = 128
        loss_fn = NTXentLoss(temperature=0.5, device='cpu')

        # Batch size 1 (e.g., 32)
        bs1 = 32
        z_i_bs1 = torch.nn.functional.normalize(torch.randn(bs1, proj_dim), p=2, dim=-1)
        z_j_bs1 = torch.nn.functional.normalize(torch.randn(bs1, proj_dim), p=2, dim=-1)
        # Ensure some similarity for positive pairs to make the loss meaningful
        z_j_bs1 = z_i_bs1 * 0.5 + z_j_bs1 * 0.5 
        z_j_bs1 = torch.nn.functional.normalize(z_j_bs1, p=2, dim=-1)
        loss_bs1 = loss_fn(z_i_bs1, z_j_bs1)

        # Batch size 2 (e.g., 64)
        bs2 = 64
        z_i_bs2 = torch.nn.functional.normalize(torch.randn(bs2, proj_dim), p=2, dim=-1)
        z_j_bs2 = torch.nn.functional.normalize(torch.randn(bs2, proj_dim), p=2, dim=-1)
        # Ensure some similarity for positive pairs
        z_j_bs2 = z_i_bs2 * 0.5 + z_j_bs2 * 0.5
        z_j_bs2 = torch.nn.functional.normalize(z_j_bs2, p=2, dim=-1)
        loss_bs2 = loss_fn(z_i_bs2, z_j_bs2)

        # The losses should be roughly comparable due to mean reduction.
        # print(f"Loss (bs={bs1}): {loss_bs1.item()}, Loss (bs={bs2}): {loss_bs2.item()}")
        # Delta needs to account for stochastic nature of random inputs and different negative sets.
        self.assertAlmostEqual(loss_bs1.item(), loss_bs2.item(), delta=0.3)


if __name__ == '__main__':
    unittest.main()
