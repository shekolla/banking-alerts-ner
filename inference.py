from transformers import pipeline
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
from constants.constants import unique_tags
from tabulate import tabulate

# Load the model
model = DistilBertForTokenClassification.from_pretrained("text_finance_tag")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Create the pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Now you can use the `nlp` object to make predictions. 
# Here's an example using a single sentence:

sentence = "Cash Withdrawal of INR 10,000.00 has been made at an ATM."
entities = nlp(sentence)

tag2id = {tag: id for id, tag in enumerate(unique_tags)}

id2tag = {id: tag for tag, id in tag2id.items()}  # reverse the tag2id dictionary
headers = ["Entity", "Entity label", "Predicted label", "Confidence Score", "Index", "Start", "End"]
table = [[sentence[entity['start']:entity['end']], entity["entity"], id2tag[int(entity['entity'].split('_')[-1])], entity["score"], entity["index"], entity["start"], entity["end"]] for entity in entities]

print(tabulate(table, headers, tablefmt="grid"))
