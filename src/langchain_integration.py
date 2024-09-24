class ConversationManager:
    def __init__(self):
        self.history = []
        self.emotional_responses = self.load_emotional_responses()
        self.current_role = "default"

    def load_emotional_responses(self):
        return {
            "warmth": ["I love you, Dad.", "You mean the world to me."],
            "concern": ["How are you feeling today?", "Is there anything I can do to help?"],
            "frustration": ["I wish things were different.", "This is so hard for me."],
            "nostalgia": ["Remember when we used to go fishing?", "I miss the old days."]
        }

    def add_to_history(self, message):
        self.history.append(message)

    def get_history(self):
        return self.history

    def switch_role(self, role):
        self.current_role = role

    def generate_emotional_response(self, emotion):
        return self.emotional_responses.get(emotion, ["I don't know what to say."])[0]

    def generate_response(self, user_input, model, tokenizer):
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def handle_conversation(self, user_input, model, tokenizer, detect_language, translate_text):
        language = detect_language(user_input)
        if language == 'ko':
            user_input = translate_text(user_input, 'en')

        response = self.generate_response(user_input, model, tokenizer)

        if language == 'ko':
            response = translate_text(response, 'ko')

        self.add_to_history(user_input)
        self.add_to_history(response)

        return response