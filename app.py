from flask import Flask
from transformers import BertForTokenClassification,BertTokenizer
import torch
import numpy as np
    
def load_model_and_tokenizer(path):
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForTokenClassification.from_pretrained(path)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer("Nirmal-re/bert-finetuned-ner-for-deployment")

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/predict")
def predict():
    text = ["EU", "Nepal"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=True, return_offsets_mapping=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    print(predictions)
    return "predicted"

if __name__ == "__main__":
    app.run(debug=True)