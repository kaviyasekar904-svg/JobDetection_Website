import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer from saved_model folder
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained("saved_model")

model.eval()

def predict_job(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    if prediction == 1:
        return "Fraudulent Job"
    else:
        return "Legitimate Job"