import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from preprocessing import load_and_prepare_data

# Load dataset
df = load_and_prepare_data()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["job_content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Load saved model
tokenizer = BertTokenizer.from_pretrained("saved_model")
model = BertForSequenceClassification.from_pretrained("saved_model")
model.eval()

class JobDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=256
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

val_dataset = JobDataset(val_texts, val_labels)
val_loader = DataLoader(val_dataset, batch_size=8)

predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_loader:
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.tolist())
        true_labels.extend(batch["labels"].tolist())

# Metrics
accuracy = accuracy_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
cm = confusion_matrix(true_labels, predictions)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)