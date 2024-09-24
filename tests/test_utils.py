import unittest
from src.utils import detect_language, translate_text

class TestUtils(unittest.TestCase):
    def test_detect_language(self):
        self.assertEqual(detect_language("Hello"), "en")
        self.assertEqual(detect_language("안녕하세요"), "ko")

    def test_translate_text(self):
        translated_text = translate_text("Hello", "ko")
        self.assertIn("안녕하세요", translated_text)

        translated_text = translate_text("안녕하세요", "en")
        self.assertIn("Hello", translated_text)

if __name__ == "__main__":
    unittest.main()