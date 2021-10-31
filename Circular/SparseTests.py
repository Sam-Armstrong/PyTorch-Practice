import unittest
from Sparse import Sparse
import torch

class SparseTests(unittest.TestCase):
    
    # Ensures that Sparse is outputting the correct size tensor
    def test_output_size(self):
        sp = Sparse(784, 784)
        x = torch.zeros((1, 784))
        y = sp(x)
        self.assertTrue(y.shape == (1, 784))


if __name__ == '__main__':
    unittest.main()