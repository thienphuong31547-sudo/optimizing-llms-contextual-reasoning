import unittest
from src.utils import compute_accuracy
import torch

class TestBenchmark(unittest.TestCase):
    def test_accuracy(self):
        logits = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))
        acc = compute_accuracy(logits, labels)
        self.assertTrue(0.0 <= acc <= 1.0)

if __name__ == "__main__":
    unittest.main()
