import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from constants.constants import unique_tags
# import csv
# import json
# import pandas as pd

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

# Note: uncomment below use-case option for your train data

# # Option 1: CSV file using native csv module
# csv_file_path = "path/to/your/csv/train.csv"
# with open(csv_file_path, "r") as f:
#     csv_reader = csv.reader(f)
#     texts = []
#     labels = []
#     for row in csv_reader:
#         texts.append(row[0])  # Assuming the text is in the first column
#         labels.append(row[1])  # Assuming the labels are in the second column

# # Option 2: JSON file
# json_file_path = "path/to/your/json/train.json"
# with open(json_file_path, "r") as f:
#     json_data = json.load(f)
#     texts = [entry["text"] for entry in json_data]
#     labels = [entry["labels"] for entry in json_data]

# # Option 3: CSV file using pandas, assuming text and labels are provided as expected
# csv_file_path = "path/to/your/csv/train.csv"
# df = pd.read_csv(csv_file_path)
# texts = df["text"].tolist()
# labels = df["labels"].tolist()

# # Option 4: JSON file using pandas
# json_file_path = "path/to/your/json/train.json"
# with open(json_file_path, "r") as f:
#     json_data = json.load(f)
# df = pd.DataFrame(json_data)
# texts = df["text"].tolist()
# labels = df["labels"].tolist()



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
