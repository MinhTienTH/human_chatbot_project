import unittest
from unittest.mock import patch
from src.model import fine_tune_rag_model

class TestModel(unittest.TestCase):
    @patch('src.model.fine_tune_rag_model')
    def test_model_initialization(self, mock_fine_tune_rag_model):
        mock_fine_tune_rag_model.return_value = ("mock_model", "mock_tokenizer")
        
        model, tokenizer = fine_tune_rag_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        self.assertEqual(model, "mock_model")
        self.assertEqual(tokenizer, "mock_tokenizer")

if __name__ == "__main__":
    unittest.main()