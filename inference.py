from transformers import pipeline
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

# Load the model
model = DistilBertForTokenClassification.from_pretrained("text_finance_tag")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Create the pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Now you can use the `nlp` object to make predictions. 
# Here's an example using a single sentence:

sentence = "Cash Withdrawal of INR 10,000.00 has been made at an ATM."
result = nlp(sentence)

# The result is a list of dictionary. Each dictionary contains an entity from the input text, its start and end indices in the text, and its predicted label.
for entity in result:
    print("{entity['entity']}: {entity['score']}, {entity['index']}, {entity['start']}, {entity['end']}")
    print(f"{entity['entity']}: {entity['score']}, {entity['index']}, {entity['start']}, {entity['end']}")

unique_tags = ['O', 'B-bank', 'I-bank', 'B-currency', 'I-amount', 'B-method', 'B-card', 'I-card', 'B-account', 'I-account', 'B-atm_id', 'I-atm_id']

tag2id = {tag: id for id, tag in enumerate(unique_tags)}

id2tag = {id: tag for tag, id in tag2id.items()}  # reverse the tag2id dictionary

for entity in result:
    print(f"Entity: {sentence[entity['start']:entity['end']]}")
    print(f"Predicted label: {id2tag[int(entity['entity'].split('_')[-1])]}")
    print(f"Confidence score: {entity['score']}")
    print()
