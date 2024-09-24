import os
import torch
from datasets import load_dataset, Dataset, load_from_disk
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

from transformers import (
    RagTokenizer, 
    RagRetriever, 
    RagSequenceForGeneration, 
    BartTokenizerFast,  # Use BartTokenizerFast explicitly
    DPRQuestionEncoderTokenizer, 
    DPRQuestionEncoder,
    Trainer, 
    TrainingArguments
)

def load_and_preprocess_data():
    data_files = {
        "train": ["data/너는 누구니.txt", "data/Who are You_.txt", "data/Game Rules for _Which One is It__.txt"]
    }
    dataset = load_dataset('text', data_files=data_files)

    def preprocess_function(examples):
        return {
            "title": [f"Text {i}" for i in range(len(examples["text"]))],
            "text": examples["text"]
        }

    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    return tokenized_datasets

def create_embeddings(dataset, batch_size=32):
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    
    def encode_batch(batch):
        inputs = tokenizer(batch['text'], truncation=True, padding=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            embeddings = model(**inputs).pooler_output.numpy()
        return {'embeddings': embeddings}
    
    dataset = dataset.map(encode_batch, batched=True, batch_size=batch_size)
    return dataset

def fine_tune_rag_model():
    # Load and preprocess data
    tokenized_datasets = load_and_preprocess_data()

    # Paths
    dataset_path = "./data/rag_dataset"
    index_dir = "./data/rag_index"
    index_name = "custom"
    
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    # Create embeddings and save dataset
    train_dataset = create_embeddings(tokenized_datasets["train"])
    train_dataset.save_to_disk(dataset_path)

    # Create and save FAISS index
    train_dataset.add_faiss_index(column='embeddings')
    index_path = os.path.join(index_dir, f"{index_name}.faiss")
    train_dataset.get_index('embeddings').save(index_path)

    # Initialize RAG components
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-sequence-nq",
        index_name="custom",
        passages_path=dataset_path,
        index_path=index_path
    )

    # Load RAG model and explicitly use BartTokenizerFast
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

    # Force re-download the tokenizer
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq", cache_dir="./cache", force_download=True)  

    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Prepare dataset for training
    def prepare_train_features(examples):
        inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
        }

    train_dataset = train_dataset.map(prepare_train_features, remove_columns=train_dataset.column_names)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    return model, tokenizer

if __name__ == "__main__":
    fine_tune_rag_model()
