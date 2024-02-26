from lib2to3.pgen2.tokenize import tokenize
from sre_parse import Tokenizer
from unittest.util import _MAX_LENGTH
import torch
import numpy as np
import pandas as pd
from transformers import pipeline

from datasets import load_dataset, DatasetDict, Dataset
from transformers import BertConfig, BertModel
from transformers import AutoTokenizer, DataCollatorWithPadding

from tqdm.auto import tqdm

import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

import evaluate

def processing_data():
    #Training on 2 sentences
    # Same as before
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    sequences = [
        "I've been waiting for a HuggingFace course my whole life.",
        "This course is amazing!",
    ]
    batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

    # This is new
    batch["labels"] = torch.tensor([1, 1])

    optimizer = AdamW(model.parameters())
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()


    # Running on MRPC

    raw_datasets = load_dataset("glue", "mrpc")
    print(raw_datasets)

    raw_train_dataset = raw_datasets["train"]
    print(raw_train_dataset[0])

    print(raw_train_dataset.features)


raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(ds):
    return tokenizer(ds["sentence1"], ds["sentence2"], truncation=True)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def finetune_model():
    from transformers import TrainingArguments, AutoModelForSequenceClassification

    checkpoint = "bert-base-uncased"
    training_args = TrainingArguments("test-trainer")

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    from transformers import Trainer

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #trainer = Trainer(
    #    model,
    #    training_args,
    #    train_dataset=tokenized_datasets["train"],
    #    eval_dataset=tokenized_datasets["validation"],
    #    data_collator=data_collator,
    #    tokenizer=tokenizer,
    #)

    #trainer.train()

    #predictions = trainer.predict(tokenized_datasets["validation"])
    #print(predictions.predictions.shape, predictions.label_ids.shape)
    
    #preds = np.argmax(predictions.predictions, axis=-1)

    #metrics = compute_metrics(preds)
    #print(metrics)

    training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    predictions = trainer.predict(tokenized_datasets["validation"])
    #print(predictions.predictions.shape, predictions.label_ids.shape)
    
    preds = np.argmax(predictions.predictions, axis=-1)

    #metrics = compute_metrics(preds)
    #print(metrics)

#finetune_model()


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

def custom_finetune(tokenized_datasets):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # data processing
    # remove unnessary columns, cleanup columns, create tensors from colours
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    tokenized_datasets["train"].column_names

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    # inspect the batch to ensure okay
    for batch in train_dataloader:
        break
    {k: print(v.shape) for k, v in batch.items()}

    from transformers import AutoModelForSequenceClassification
    #load model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    #pass batch to model
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape)

    from transformers import AdamW

    optimizer = AdamW(model.parameters(), lr=5e-5)
    #determine number of steps we take (epochs*steps) to use lr weight decay method above
    from transformers import get_scheduler

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    # train model
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    #evaluate model
    import evaluate

    metric = evaluate.load("glue", "mrpc")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()

custom_finetune(tokenized_datasets)


def distributed_training(tokenized_datasets):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # data processing
    # remove unnessary columns, cleanup columns, create tensors from colours
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    tokenized_datasets["train"].column_names

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    from accelerate import Accelerator
    from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

    accelerator = Accelerator()

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    train_dl, eval_dl, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)