from flask import Flask, render_template, request
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from collections import Counter
    
def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForTokenClassification.from_pretrained(path)
    return tokenizer, model

#tokenizer, model = load_model_and_tokenizer("Nirmal-re/bert-finetuned-ner-for-deployment")
tokenizer, model = load_model_and_tokenizer("paullatham1/roberta-finetuned-ner-longforms")

label_encoding = {0: "B-O", 1: "B-AC", 2: "B-LF", 3: "I-LF"}

app = Flask(__name__)

def predict(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    predictions = torch.argmax(output.logits, dim=2).tolist()[0]
    labels = [label_encoding[p] for p in predictions]
    return labels

@app.route("/")
def hello():
    return render_template("index.html")

@app.post("/predict/")
async def get_prediction(text):
    labels = predict(text)
    #log_entry = f"Input: {text} | Prediction: {labels}"
    #logging.info(log_entry)
    return {"labels": labels}


'''def predict():
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
    print(original_tokens)
    for i, pred in enumerate(one_to_one_predictions):
        if len(pred) == 1:
            value = f"{original_tokens[i]}: {pred[0]}"
            results.append(value)
        if len(pred) > 1:
             most_common_pred = Counter(pred).most_common(1)[0][0]
             value = f"{original_tokens[i]}: {most_common_pred}"
             results.append(value)

    return {"results": results}'''


if __name__ == "__main__":
    app.run(host='localhost', port=5000)