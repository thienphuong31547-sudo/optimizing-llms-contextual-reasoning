import unittest
from src.model import ContextualReasoningModel

class TestContextualReasoningModel(unittest.TestCase):
    def test_model_creation(self):
        model = ContextualReasoningModel()
        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()
