import torch.multiprocessing as mp
from model import fine_tune_rag_model
from langchain_integration import ConversationManager
from utils import detect_language, translate_text
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='multiprocessing.resource_tracker')

def worker_function(user_input):
    model, tokenizer = fine_tune_rag_model()
    conversation_manager = ConversationManager()
    conversation_manager.add_to_history(user_input)

    response = conversation_manager.handle_conversation(user_input, model, tokenizer, detect_language, translate_text)
    print(response)

    conversation_manager.switch_role("comedian")
    role_response = conversation_manager.generate_emotional_response("warmth")
    print(f"Role (comedian) response: {role_response}")

def main():
    user_input = "How are you today, Dad?"
    process = mp.Process(target=worker_function, args=(user_input,))
    process.start()
    process.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
