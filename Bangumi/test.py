import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error




df_test = pd.read_json('./Bangumi/test.jsonl', lines=True)
df_test = df_test.dropna(subset=['text', 'point'])


df_test['point'] = df_test['point'].astype(int)
df_test['point'] = df_test['point'] - 1


tokenizer = AutoTokenizer.from_pretrained('./bangumi_bert_model')
model = AutoModelForSequenceClassification.from_pretrained('./bangumi_bert_model')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


class BangumiDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)  # 确保 long 类型
        }


test_texts = df_test['text'].tolist()
test_labels = df_test['point'].tolist()

test_dataset = BangumiDataset(test_texts, test_labels, tokenizer, 128)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


model.eval() 

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits


        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")

mse = mean_squared_error(all_labels, all_preds)
print(f"Mean Squared Error: {mse:.4f}")