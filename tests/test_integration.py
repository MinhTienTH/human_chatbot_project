import unittest
from src.model import fine_tune_rag_model
from src.langchain_integration import ConversationManager
from src.utils import detect_language, translate_text

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.model, self.tokenizer = fine_tune_rag_model()
        self.conversation_manager = ConversationManager()

    def test_conversation_flow(self):
        user_input = "How are you today, Dad?"
        response = self.conversation_manager.handle_conversation(user_input, self.model, self.tokenizer, detect_language, translate_text)
        self.assertIsNotNone(response)
        self.assertIn("Dad", response)

    def test_role_switching(self):
        self.conversation_manager.switch_role("comedian")
        role_response = self.conversation_manager.generate_emotional_response("warmth")
        self.assertIn("I love you", role_response)

if __name__ == "__main__":
    unittest.main()