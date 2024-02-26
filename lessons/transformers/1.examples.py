from unittest.util import _MAX_LENGTH
import torch
import numpy as np
import pandas as pd
from transformers import pipeline


def analysis():
    classifier = pipeline("sentiment-analysis")
    return classifier(["I've been waiting for a HuggingFace course my whole life.", "I do not hate this but I don't like it"])

def classify():
    classifier = pipeline("zero-shot-classification")
    return classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business", "tpgphy"],
    )

def generate():
    generator = pipeline("text-generation")
    return generator("In this course, we will teach you how to", num_return_sequences=5)

def generate2():
    generator = pipeline("text-generation", model="distilgpt2")
    return generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,
    )

def fillmask():
    unmasker = pipeline("fill-mask")
    return unmasker("This course will teach you all about <mask> models.", top_k=2)

def named_entity_recognition():
    ner = pipeline("ner", grouped_entities=True)
    return ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

def qa():
    question_answerer = pipeline("question-answering")
    return question_answerer(
        question="Where do I work?",
        context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    )

def summarize():
    summarizer = pipeline("summarization")
    return summarizer(
        """
        America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science. As a result, there 
        are declining offerings in engineering subjects dealing with infrastructure, 
        the environment, and related issues, and greater concentration on high 
        technology subjects, largely supporting increasingly complex scientific 
        developments. While the latter is important, it should not be at the expense 
        of more traditional engineering.

        Rapidly developing economies such as China and India, as well as other 
        industrial countries in Europe and Asia, continue to encourage and advance 
        the teaching of engineering. Both China and India, respectively, graduate 
        six and eight times as many traditional engineers as does the United States. 
        Other industrial countries at minimum maintain their output, while America 
        suffers an increasingly serious decline in the number of engineering graduates 
        and a lack of well-educated engineers.
    """
    )

def translate():
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    return translator("Ce cours est produit par Hugging Face.")


#res1 = analysis()
#res2 = classify()
#res3 = generate2()
#res4 = fillmask()
#res5 = named_entity_recognition()
#res6 = qa()
#res7 = summarize()
res8 = translate()

print(res8)

