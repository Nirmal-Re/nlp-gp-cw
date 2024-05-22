from flask import Flask, render_template, request
from transformers import BertForTokenClassification,BertTokenizerFast
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from collections import Counter
import json
from datetime import datetime 
import asyncio
import re
    
def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer("paullatham1/roberta-finetuned-ner-longforms")

app = Flask(__name__)


#Function to write data to file
def write_to_file(dictionary):
    try:
        with open("user_queries.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"all_queries": []}

    data["all_queries"].append(dictionary)
 
    with open("user_queries.json", "w") as f:
        json.dump(data, f)
    return "Data written to file"


#Landing Page endpoint
@app.route("/")
def hello():
    return render_template("index.html")

label_encoding = {0: "B-O", 1: "B-AC", 2: "B-LF", 3: "I-LF"}

def clean_token(token):
    # Remove Ġ (start of word token with RoBERTa tokenizer)
    return re.sub(r'^Ġ', '', token).strip()


#predit endpoint
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    tokens = tokenizer(message, return_tensors="pt", truncation=True, max_length=512)
    input_ids = tokens['input_ids'][0].tolist()
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    cleaned_tokens = [clean_token(token) for token in decoded_tokens if token not in ['<s>', '</s>']]

    with torch.no_grad():
        output = model(**tokens)

    predictions = torch.argmax(output.logits, dim=2).tolist()[0]
    labels = [label_encoding[p] for p in predictions]

    token_label_pairs = [
        (token, label_encoding[pred])
        for token, pred in zip(cleaned_tokens, predictions)
        if token not in ['<s>', '</s>']
    ]

    results = []
    for token, label in token_label_pairs:
        value = f"{token} => {label}"
        results.append(value)

    #print(f"Input: {message} | Prediction: {labels}")

    dictionary = {
        "query_datetime": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "query" : message,
        "labels" : labels,
        "result": results,
    }

    write_to_file(dictionary)

    #BERT code

    '''
    inputs = tokenizer(message, return_tensors="pt", truncation=True, is_split_into_words=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    true_predictions = [model.config.id2label[prediction] for prediction in predictions.tolist()[0]]
    tokens = [tokenizer.convert_ids_to_tokens(id) for id in inputs['input_ids'].tolist()][0]

    original_tokens = []
    one_to_one_predictions = []
    for token, predicted in zip(tokens, true_predictions):
        if token in ['[CLS]', '[SEP]']:
            continue

        if token.startswith("##"):
            original_tokens[-1] += token[2:]
            one_to_one_predictions[-1].append(predicted)
        else:
            original_tokens.append(token)
            one_to_one_predictions.append([predicted])
    results = []

    for i, pred in enumerate(one_to_one_predictions):
        if len(pred) == 1:
            value = f"{original_tokens[i]} => {pred[0]}"
            results.append(value)
        if len(pred) > 1:
             most_common_pred = Counter(pred).most_common(1)[0][0]
             value = f"{original_tokens[i]} => {most_common_pred}"
             results.append(value)
    dictionary = {
    "query_datetime": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "orginal_query": message,
    "tokenized_query": tokens,
    "predicted_labels": true_predictions,
    "results": results
    }
    write_to_file(dictionary)'''

    return {"results": results}

if __name__ == "__main__":
    app.run(host='localhost', port=5000)