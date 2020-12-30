from string import punctuation
from collections import Counter
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

train_on_gpu = torch.cuda.is_available()


def load_test(path_reviews, path_labels):
    with open(path_reviews, 'r') as f:
        reviews = f.read()
    with open(path_labels, 'r') as r:
        labels = r.read()
    return reviews, labels


def preprocess_text(text):
    reviews = text.lower()
    all_text = ''.join([c for c in reviews if c not in punctuation])
    reviews_split = all_text.split('\n')
    all_text = ''.join(reviews_split)
    return all_text, reviews_split


def encode_text(text, split_text):
    words_text = text.split()
    counts = Counter(words_text)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    reviews_ints = []
    for review in split_text:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])
    return reviews_ints, vocab_to_int


def encode_labels(labels_text):
    labels_split = labels_text.split()
    encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])
    return encoded_labels


def outlier_removal(int_texts, encoded_labels):
    lengths = Counter([len(x) for x in int_texts])
    non_zero_indices = [ii for ii, review in enumerate(int_texts) if len(review) != 0]
    processed_int_texts = [int_texts[ii] for ii in non_zero_indices]
    processed_labels = np.array([encoded_labels[ii] for ii in non_zero_indices])
    return processed_int_texts, processed_labels


def pad_features(int_texts, seq_lengths):
    features = np.zeros((len(int_texts), seq_lengths), dtype=int)
    for i, row in enumerate(int_texts):
        features[i, -len(row):] = np.array(row)[:seq_lengths]
    return features


def generate_data_loaders(train_data, train_label, valid_data, valid_label, test_data, test_label):
    train_data_loader = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
    valid_data_loader = TensorDataset(torch.from_numpy(valid_data), torch.from_numpy(valid_label))
    test_data_loader = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
    return train_data_loader, valid_data_loader, test_data_loader


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, output_size, embed_dim, hidden_dim, n_layers, drop_prob = 0.5):
        super(SentimentRNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden














