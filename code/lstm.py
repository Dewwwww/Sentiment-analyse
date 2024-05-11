import pandas as pd
import re
import jieba.posseg as psg
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    if all(len(item) == 2 for item in batch):  # Check if every item has two elements
        tokens, labels = zip(*batch)  # This assumes each item in the batch is (token, label)
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        tokens = batch  # This assumes each item in the batch is just tokens
        labels = None

    tokens_padded = pad_sequence([torch.tensor(t, dtype=torch.long) for t in tokens], batch_first=True, padding_value=0)

    if labels is not None:
        return tokens_padded, labels
    return tokens_padded

# 文本清洗
def clean_text(text):
    pattern = re.compile('[a-zA-Z0-9]+|[\s]+')
    return pattern.sub('', text)

# 分词
def tokenize(text):
    return ' '.join([x.word for x in psg.cut(text)])

# 数据预处理
def preprocess_reviews(neg_file, pos_file):
    # 读取负面评论
    with open(neg_file, 'r', encoding='utf-8') as file:
        neg_reviews = file.readlines()
    neg_reviews = [clean_text(line.strip()) for line in neg_reviews]
    
    # 读取正面评论
    with open(pos_file, 'r', encoding='utf-8') as file:
        pos_reviews = file.readlines()
    pos_reviews = [clean_text(line.strip()) for line in pos_reviews]
    
    # 创建DataFrame
    data = {'comment': neg_reviews + pos_reviews,
            'label': [0] * len(neg_reviews) + [1] * len(pos_reviews)}
    df = pd.DataFrame(data)
    df['tokens'] = df['comment'].apply(tokenize)
    return df

# 建立词汇表
def build_vocab(reviews):
    vocab = {}
    for tokens in reviews['tokens']:
        for word in tokens.split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1  # 词汇表索引从1开始
    vocab['<pad>'] = 0
    return vocab

class ReviewsDataset(Dataset):
    def __init__(self, reviews, vocab, include_labels=True):
        self.reviews = reviews
        self.vocab = vocab
        self.include_labels = include_labels  # 标记是否包含标签

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        tokens = [self.vocab.get(word, 0) for word in self.reviews.iloc[idx]['tokens'].split()]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        if self.include_labels:
            label = self.reviews.iloc[idx]['label']
            return tokens_tensor, torch.tensor(label, dtype=torch.long)
        return tokens_tensor

# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        hidden = hidden[-1]
        output = self.fc(hidden)
        return output

# 训练模型
def train_model(model, data_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        for tokens_padded, labels in data_loader:
            optimizer.zero_grad()
            sequence_lengths = [len(list(filter(lambda x: x > 0, t.tolist()))) for t in tokens_padded]
            outputs = model(tokens_padded, sequence_lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def predict(model, data_loader, label_map):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, tuple) and len(batch) == 2:
                tokens_padded, _ = batch
            else:
                tokens_padded = batch  # Assuming batch returns only tokens without labels

            sequence_lengths = [len(t[t > 0]) for t in tokens_padded]  # calculate lengths ignoring padding
            outputs = model(tokens_padded, sequence_lengths)
            predicted_indices = torch.max(outputs, 1)[1]
            predicted_labels = [label_map[idx.item()] for idx in predicted_indices]
            predictions.extend(predicted_labels)
    return predictions

def preprocess_unlabeled_reviews(file_name, vocab):
    reviews = pd.read_csv(file_name)
    reviews['content'] = reviews['comment'].apply(clean_text)
    reviews['tokens'] = reviews['content'].apply(tokenize)
    return ReviewsDataset(reviews, vocab, include_labels=False)  # 使用标志位

# 主函数
def main():
    neg_file = 'negative.txt'
    pos_file = 'positive.txt'
    reviews = preprocess_reviews(neg_file, pos_file)
    vocab = build_vocab(reviews)
    dataset = ReviewsDataset(reviews, vocab)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)

    model = LSTMModel(len(vocab) + 1, 100, 256, 2)  # vocab_size, embedding_dim, hidden_dim, num_classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, data_loader, optimizer, criterion, num_epochs=10)

    # 加载未标记的数据集并进行预测
    unlabeled_data_loader = DataLoader(preprocess_unlabeled_reviews('1000.csv', vocab), batch_size=10, shuffle=False, collate_fn=collate_fn)
    predictions = predict(model, unlabeled_data_loader, {0: 'neg', 1: 'pos'})

    # 保存预测结果到CSV文件
    pd.DataFrame({
        'content': [data['content'] for data in unlabeled_data_loader.dataset.reviews.to_dict('records')],
        'predicted_label': predictions
    }).to_csv('predicted_reviews.csv', index=False)

    print("Predictions saved to 'predicted_reviews.csv'.")


if __name__ == "__main__":
    main()

