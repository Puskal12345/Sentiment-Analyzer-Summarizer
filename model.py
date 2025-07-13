import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification,T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 128

LABEl = {'negative': 0, 'neutral': 1, 'positive': 2}
IdLabel = {v: k for k, v in LABEl.items()}

def preprocessing(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_model(model, train_loader, val_loader, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        avg_val_loss, val_accuracy = evaluate(model, val_loader)
        print(f"\n Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")



def evaluate(model, val_loader):
    model.eval() 
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            val_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_predictions / total_predictions
    return avg_val_loss, val_accuracy

def train(csv_path="./data/Tweets.csv", save_path = "./saved/model_state.pth"):
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].apply(preprocessing)
    df['sentiment'] = df['sentiment'].map(LABEl)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['sentiment'].tolist(),
        test_size=0.2, random_state=RANDOM_SEED
    )

    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_model(model, train_loader, val_loader, optimizer)
    torch.save(model.state_dict(), save_path)
    print("Training Completed")


def load_sentiment_analyzer(path="model_state.pth"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def predict_sentiment(text, model, tokenizer):
    encoded = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return IdLabel[pred]


def load_summarizer():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def summarize(text, model, tokenizer, max_length=60, min_length=10):
    inputs = tokenizer.encode(text, return_tensors='pt', truncation=True).to(DEVICE)
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
