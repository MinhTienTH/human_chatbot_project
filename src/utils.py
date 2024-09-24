from langdetect import detect
from googletrans import Translator

def detect_language(text):
    return detect(text)

def translate_text(text, dest_language):
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text