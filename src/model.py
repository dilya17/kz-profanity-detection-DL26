"""Character vocabulary and character-level CNN classifier."""
import json
from collections import Counter

import torch
import torch.nn as nn

PAD, UNK = 0, 1


class CharVocab:
    """Maps each character to a number. Index 0 = padding, 1 = unknown character."""
    
    def __init__(self, chars):
        self.itos = ["<pad>", "<unk>"] + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, min_freq=2):
        """Built from TRAINING texts only."""
        counts = Counter(c for t in texts for c in t)
        return cls(sorted(c for c, n in counts.items() if n >= min_freq))

    def encode(self, text, max_len):
        ids = [self.stoi.get(c, UNK) for c in text[:max_len]]   # long comments are truncated
        return ids + [PAD] * (max_len - len(ids))                 # short ones are padded

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        json.dump(self.itos, open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        v = cls([]); v.itos = json.load(open(path, encoding="utf-8"))
        v.stoi = {c: i for i, c in enumerate(v.itos)}
        return v


class CharCNN(nn.Module):
    """Character-level CNN.
    Each filter slides over the text and reacts to short letter patterns (3, 4 or 5 characters),
    e.g. parts of swear words. Max-pooling keeps the strongest reaction of each filter,
    and the last layer turns them into one score (logit)."""
    
    def __init__(self, vocab_size, emb_dim=48, n_filters=96, kernel_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), 1)

    def forward(self, x):                        # x: (batch, max_len) character ids
        e = self.emb(x).transpose(1, 2)          # (batch, emb_dim, max_len)
        h = [torch.relu(c(e)).max(dim=2).values for c in self.convs]   # each: (batch, n_filters)
        return self.fc(self.drop(torch.cat(h, dim=1))).squeeze(1)      # logit; sigmoid -> probability
