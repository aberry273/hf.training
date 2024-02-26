from lib2to3.pgen2.tokenize import tokenize
from sre_parse import Tokenizer
from unittest.util import _MAX_LENGTH
import torch
import numpy as np
import pandas as pd
from transformers import pipeline

from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

print(config)

# Building the model from the config
#model = BertModel(config)

model = BertModel.from_pretrained("bert-base-cased")

#model.save_pretrained("output/")

sequences = ["Hello!", "Cool.", "Nice!"]
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

import torch

model_inputs = torch.tensor(encoded_sequences)

output = model(model_inputs)

#print(output)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

decoded_string = tokenizer.decode(ids)
print(decoded_string)


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)

# tokenizers expect multiple sequences by default, so we add our ids to an array and pass in
input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)