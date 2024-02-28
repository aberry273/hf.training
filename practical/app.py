# defaults
from itertools import accumulate
import torch
import numpy as np
import pandas as pd

# charts
from matplotlib import pyplot as plt

# huggingface

## transformers
from transformers import pipeline, BertConfig, BertModel, AutoTokenizer, DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, get_scheduler

from accelerate import Accelerator
from huggingface_hub import Repository, get_full_repo_name, create_repo, repo_info
#git_pull, 
## utilities
from lib2to3.pgen2.tokenize import tokenize
from sre_parse import Tokenizer
from unittest.util import _MAX_LENGTH
from tqdm.auto import tqdm

## datasets
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

## metrics
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
from typing import Optional

classification_ds = "cnmoro/Instruct-PTBR-ENUS-11M"
_col_summarised_text = "INSTRUCTION"
_col_full_text = "RESPONSE"

## FN - DATASET

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

def init_dataset():
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

    return train_test_valid_ds, tokenizer, model_checkpoint
    
max_input_length = 512
max_target_length = 30

train_test_valid_ds, tokenizer, model_checkpoint = init_dataset()

## FN - TOKENIZATION

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

#Accelerate - set format to torch
tokenized_datasets.set_format("torch")

generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"

rouge_score = evaluate.load("rouge")
scores = rouge_score.compute(predictions=[generated_summary], references=[reference_summary])

# import punctuation module to retrieve first few sentences of instructions 
nltk.download("punkt")

def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])

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
print("LOADED MODEL")
# Create summaries

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

## FN - METRICS

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
    result = {key: value* 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

tokenized_datasets = tokenized_datasets.remove_columns(
    train_test_valid_ds["train"].column_names
)

features = [tokenized_datasets["train"][i] for i in range(2)]
data_collator(features)

def sequential_training():
    
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    #trainer.train()

    score = trainer.evaluate()

    print(score)

    trainer.push_to_hub(commit_message="Training complete", tags="summarization")


# Accelerate - Use DataLoaders


# split generated ummaries into separate chunks
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


model_name = "test-bert-finetuned-squad-accelerate"
repo_name = get_full_repo_name(model_name)

output_dir = "results-mt5-finetuned-squad-accelerate"


def create_model_repo():
    try:
        create_repo(model_name)
    except:
        print("Exists")

def repo_exists(repo_id: str, repo_type: Optional[str] = None, token: Optional[str] = None) -> bool:
    try:
        repo_info(repo_id, repo_type=repo_type, token=token)
        return True
    except RepositoryNotFoundError:
        return False
    
repo = Repository(output_dir, clone_from=repo_name)
#repo = git_pull(o)

from tqdm.auto import tqdm
import torch
import numpy as np


def batch_training(model):
    batch_size = 8
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["valid"], collate_fn=data_collator, batch_size=batch_size
    )

    #accelearate 
    print('pre-optimizer')
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # load model into accelerator
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # set learning_rate
    num_train_epochs = 10
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]

                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(
                    batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
                )

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                # Replace -100 in the labels as we can't decode them
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(
                    decoded_preds, decoded_labels
                )

                rouge_score.add_batch(predictions=decoded_preds, references=decoded_labels)

        # Compute metrics
        result = rouge_score.compute()
        # Extract the median ROUGE scores
        result = {key: value * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}
        print(f"Epoch {epoch}:", result)

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )

#sequential_training()
#batch_training(model)

def print_summary(idx, summarizer):
    review = train_test_valid_ds["test"][idx]["review_body"]
    title = train_test_valid_ds["test"][idx]["review_title"]
    summary = summarizer(train_test_valid_ds["test"][idx]["review_body"])[0]["summary_text"]
    print(f"'>>> Review: {review}'")
    print(f"\n'>>> Title: {title}'")
    print(f"\n'>>> Summary: {summary}'")

from transformers import pipeline

def use_model():

    hub_model_id = "aberry273/"+output_dir
    summarizer = pipeline("summarization", model=hub_model_id)
    print_summary(100, summarizer)


use_model()