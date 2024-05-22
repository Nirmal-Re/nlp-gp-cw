from flask import Flask, render_template, request
from transformers import BertForTokenClassification,BertTokenizerFast
import torch
from collections import Counter
import json
from datetime import datetime 
import asyncio

    
def load_model_and_tokenizer(path):
    tokenizer = BertTokenizerFast.from_pretrained(path)
    model = BertForTokenClassification.from_pretrained(path)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer("Nirmal-re/bert-finetuned-ner-for-deployment")

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



#predit endpoint
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    text = message.split(" ")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=True)
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
    write_to_file(dictionary)

    return {"results": results}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)