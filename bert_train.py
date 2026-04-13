import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from preprocessing import load_and_prepare_data

print("Loading dataset...")

df = load_and_prepare_data()

# Use smaller dataset for faster training (optional but helpful)
df = df.sample(5000, random_state=42)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["job_content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("Dataset loaded")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JobDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

print("Tokenizing data...")

train_dataset = JobDataset(train_texts, train_labels)

label_tensor = torch.tensor(train_labels.values, dtype=torch.long)
class_counts = torch.bincount(label_tensor)
class_weights_cpu = (1.0 / class_counts.float())
class_weights_cpu = class_weights_cpu / class_weights_cpu.sum() * len(class_counts)
sample_weights = class_weights_cpu[label_tensor]
class_weights = class_weights_cpu.to(device)
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=sampler
)

print("Data ready")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
model.to(device)

model.train()

epochs = 1

print("Training started...")

for epoch in range(epochs):

    total_loss = 0

    for i, batch in enumerate(train_loader):

        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
        )

        loss = loss_fn(outputs.logits, batch["labels"])

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(f"Batch {i} Loss: {loss.item()}")

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1} Average Loss: {avg_loss}")

print("Saving model...")

model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

print("BERT Training Completed")
