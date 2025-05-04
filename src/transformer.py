import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # Linear projections
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1,2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)

        output = torch.matmul(scores, v)
        output = output.transpose(1,2).contiguous().view(bs, -1, self.num_heads * self.d_k)
        output = self.out(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, mask=None):
        x = self.embedding(src)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# Hyperparameters
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 128
BATCH_SIZE = 32

sample_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))

model = TransformerEncoder(vocab_size=VOCAB_SIZE)

output = model(sample_input)

print(output.shape) 

