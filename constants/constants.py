# Please update the entity after adding new entities

"""
"B-bank" and "I-bank" for the name of the bank.
"B-currency" and "I-amount" for the amount of money.
"B-method" for the transaction method (like "ATM").
"B-card" and "I-card" for the debit card.
"B-account" and "I-account" for the account number.
"B-atm_id" and "I-atm_id" for the ATM ID.
"B-" and "I-" prefixes are standard in NER and stand for "Beginning" and "Inside". They indicate that a particular token is the start of an entity or inside an entity. A single-token entity would be labeled with "B-" prefix.
"""

unique_tags = ['O', 'B-bank', 'I-bank', 'B-currency', 'I-amount', 'B-method', 'B-card', 'I-card', 'B-account', 'I-account', 'B-atm_id', 'I-atm_id']
