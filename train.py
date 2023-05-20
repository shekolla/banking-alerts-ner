import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

# Use a DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


"""
"B-bank" and "I-bank" for the name of the bank.
"B-currency" and "I-amount" for the amount of money.
"B-method" for the transaction method (like "ATM").
"B-card" and "I-card" for the debit card.
"B-account" and "I-account" for the account number.
"B-atm_id" and "I-atm_id" for the ATM ID.
"B-" and "I-" prefixes are standard in NER and stand for "Beginning" and "Inside". They indicate that a particular token is the start of an entity or inside an entity. A single-token entity would be labeled with "B-" prefix.
"""
# Example sentences and corresponding tags
texts = [
    ["Dear", "Customer", ",", "Greetings", "from", "ICICI", "Bank", "."],
    ["Cash", "Withdrawal", "of", "INR", "10,000.00", "has", "been", "made", "at", "an", "ATM", "."],
    ["Your", "Debit", "Card", "linked", "to", "Account", "XX048", "was", "used", "."],
    ["Info", ":", "ATM", "*", "S1CWK458", "*", "CA", "."],
    ["The", "Available", "Balance", "in", "your", "Account", "is", "INR", "7,753.24", "."]
]

# Corresponding labels
tags = [
    ["O", "O", "O", "O", "O", "B-bank", "I-bank", "O"],
    ["O", "O", "O", "B-currency", "I-amount", "O", "O", "O", "O", "O", "B-method", "O"],
    ["O", "B-card", "I-card", "O", "O", "B-account", "I-account", "O", "O", "O"],
    ["O", "O", "B-atm_id", "I-atm_id", "I-atm_id", "I-atm_id", "I-atm_id", "O"],
    ["O", "O", "O", "O", "O", "B-account", "O", "B-currency", "I-amount", "O"]
]

unique_tags = ['O', 'B-bank', 'I-bank', 'B-currency', 'I-amount', 'B-method', 'B-card', 'I-card', 'B-account', 'I-account', 'B-atm_id', 'I-atm_id']

tag2id = {tag: id for id, tag in enumerate(unique_tags)}

encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
labels = encode_tags(tags, encodings)

# Create the PyTorch Dataset
dataset = NERDataset(encodings, labels)

# Load the model and pass the number of labels
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=len(unique_tags))

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Create the Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)
trainer.train()
trainer.save_model("text_finance_tag")
