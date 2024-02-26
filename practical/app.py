# defaults
from itertools import accumulate
import torch
import numpy as np
import pandas as pd

# charts
from matplotlib import pyplot as plt

# huggingface

## transformers
from transformers import pipeline
from transformers import BertConfig, BertModel, AutoTokenizer, DataCollatorWithPadding, AdamW, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

## utilities
from lib2to3.pgen2.tokenize import tokenize
from sre_parse import Tokenizer
from unittest.util import _MAX_LENGTH
from tqdm.auto import tqdm

## datasets
from datasets import load_dataset, DatasetDict, Dataset

## metrics
import evaluate
import nltk
from nltk.tokenize import sent_tokenize

classification_ds = "cnmoro/Instruct-PTBR-ENUS-11M"
_col_summarised_text = "INSTRUCTION"
_col_full_text = "RESPONSE"

def is_en(x): return ( x["LANGUAGE"] == "en" )

def split_train_test_valid(ds):
    # Split train/validation/test sets -  90% train, 10% test + validation
    train_testvalid = ds.train_test_split(test_size=0.1, train_size=0.9)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    return DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})


def show_samples(dataset, features, num_samples=3, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
            for feature in features:
                print(f"\n'>> {feature}: {example[feature]}'")

def get_word_count(dataset, colname):
    #samples = dataset.shuffle(seed=42).select(range(50000))
    dataset.set_format("pandas")
    df = dataset[:]
    # Get word count for instructions
    df[_col_summarised_text+"_COUNT"] = df[colname].apply(lambda x: len(str(x).split(' ')))

    dataset.reset_format()
    return dataset

def show_histogram(df, colname):
    hist = df.hist(column=colname,legend=True)
    plt.show()

# Load datasets
dataset = load_dataset(classification_ds, split="train")
# Filter english only
# Ignore, include portugese so the model doesn't overfit to english
filtered_dict = dataset.filter(is_en)
filtered_dict = filtered_dict.shuffle(seed=42).select(range(50000))

get_word_count(filtered_dict, _col_summarised_text)
print(dataset)
# show_histogram(df_count, "INSTRUCTION_COUNT")

train_test_valid_ds = split_train_test_valid(filtered_dict)

print(filtered_dict)
#show_samples(train_test_valid_ds, filtered_dict.features)

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

inputs = tokenizer("I loved reading the Hunger Games!")
tokenizer.convert_ids_to_tokens(inputs.input_ids)

max_input_length = 512
max_target_length = 30

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples[_col_full_text],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples[_col_summarised_text], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = train_test_valid_ds.map(preprocess_function, batched=True)

generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"

rouge_score = evaluate.load("rouge")
scores = rouge_score.compute(predictions=[generated_summary], references=[reference_summary])

print(scores)
#mid_score = scores["rouge1"].mid
#print(mid_score)

# import punctuation module to retrieve first few sentences of instructions 
nltk.download("punkt")



def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])

print(three_sentence_summary(train_test_valid_ds["train"][3][_col_full_text]))

# extract sentences from ds
def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset[_col_full_text]]
    return metric.compute(predictions=summaries, references=dataset[_col_summarised_text])

# compute scores 
import pandas as pd

score = evaluate_baseline(train_test_valid_ds["valid"], rouge_score)
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, round(score[rn] * 100, 2)) for rn in rouge_names)
print(rouge_dict)

# load t53
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Create summaries
from transformers import Seq2SeqTrainingArguments

batch_size = 8
num_train_epochs = 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-{classification_ds}",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

tokenized_datasets = tokenized_datasets.remove_columns(
    train_test_valid_ds["train"].column_names
)

features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

score = trainer.evaluate()

print(score)