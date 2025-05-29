import unittest
import torch
from PIL import Image
import os
import sys

# Adjust path to import from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_augmentation import SimCLRAugmentation 

class TestSimCLRAugmentation(unittest.TestCase):

    def create_dummy_image(self, size=(256, 256), color='blue'):
        return Image.new('RGB', size, color=color)

    def test_output_format_cifar10(self):
        img_size = 32
        # Note: s=0.5 is default for cifar10 in the implementation, matching paper
        augmenter = SimCLRAugmentation(size=img_size, dataset='cifar10') 
        dummy_img = self.create_dummy_image(size=(img_size, img_size))
        
        view1, view2 = augmenter(dummy_img)
        
        self.assertIsInstance(view1, torch.Tensor)
        self.assertIsInstance(view2, torch.Tensor)
        self.assertEqual(view1.shape, (3, img_size, img_size))
        self.assertEqual(view2.shape, (3, img_size, img_size))
        self.assertEqual(view1.dtype, torch.float32)
        self.assertEqual(view2.dtype, torch.float32)
        
        # Check that views are different
        # With random augmentations, it's highly unlikely they are identical for a complex image.
        # For a single color dummy image, some transforms might not change it much,
        # but RandomResizedCrop should still make them different.
        self.assertFalse(torch.equal(view1, view2), "Augmented views should be different")

    def test_output_format_imagenet(self):
        img_size = 224
        # Note: s=1.0 is default for imagenet in the implementation
        augmenter = SimCLRAugmentation(size=img_size, dataset='imagenet') 
        dummy_img = self.create_dummy_image(size=(img_size, img_size))
        
        view1, view2 = augmenter(dummy_img)
        
        self.assertIsInstance(view1, torch.Tensor)
        self.assertIsInstance(view2, torch.Tensor)
        self.assertEqual(view1.shape, (3, img_size, img_size))
        self.assertEqual(view2.shape, (3, img_size, img_size))
        self.assertEqual(view1.dtype, torch.float32)
        self.assertEqual(view2.dtype, torch.float32)
        
        self.assertFalse(torch.equal(view1, view2), "Augmented views should be different for ImageNet")

    def test_unsupported_dataset(self):
        with self.assertRaises(ValueError) as context:
            SimCLRAugmentation(size=32, dataset='unknown_dataset')
        self.assertTrue('Unsupported dataset' in str(context.exception))


    def test_different_views_strong_check(self):
        # This test uses a slightly more complex image to better ensure views are different.
        img_size = 64 # Use a reasonable size
        augmenter = SimCLRAugmentation(size=img_size, dataset='cifar10') # CIFAR-10 pipeline has enough randomness
        
        # Create an image with a gradient or some pattern
        img_array = [[[x+y for x in range(img_size)] for y in range(img_size)] for _ in range(3)]
        img_array_torch = torch.tensor(img_array, dtype=torch.uint8).permute(2,0,1) # CHW
        pil_img = Image.fromarray(img_array_torch.permute(1,2,0).numpy(), 'RGB') # HWC for PIL

        view1, view2 = augmenter(pil_img)
        
        self.assertFalse(torch.equal(view1, view2), "Augmented views for patterned image should be different")
        
        # Also check general range of values (normalized)
        # Mean can be around 0, std around 1, so values are often in [-3, 3]
        # This is a loose check.
        self.assertTrue(view1.min() < 0 and view1.max() > 0, "Normalized view1 values seem off-center")
        self.assertTrue(view2.min() < 0 and view2.max() > 0, "Normalized view2 values seem off-center")


if __name__ == '__main__':
    unittest.main()
