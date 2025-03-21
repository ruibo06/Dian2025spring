import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_scheduler
)
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_json('Bangumi/data/comments_and_ratings.jsonl', lines=True)

# 去掉 NaN
df = df.dropna(subset=['text', 'point'])

# 确保 point 是整数
df['point'] = df['point'].astype(int)
df['point'] = df['point'] - 1

# 划分数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['point'].tolist(), test_size=0.1, random_state=42
)

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained('D:/BERT')

# 加载 BERT 分类模型
num_classes = df['point'].nunique()
model = AutoModelForSequenceClassification.from_pretrained(
    'D:/BERT', num_labels=num_classes, use_safetensors=True
)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 数据集类
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

# 数据加载
train_dataset = BangumiDataset(train_texts, train_labels, tokenizer, 128)
val_dataset = BangumiDataset(val_texts, val_labels, tokenizer, 128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 优化器 & 学习率调度
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3
)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 训练循环
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    
    total_train_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

    print(f"  Training Loss: {total_train_loss / len(train_loader):.4f}")

# 保存模型
model.save_pretrained("./bangumi_bert_model")
tokenizer.save_pretrained("./bangumi_bert_model")
