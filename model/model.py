import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)

        return context_vector


class FeedForward(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_out, 4 * d_out), nn.GELU(), nn.Linear(4 * d_out, d_out)
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=dropout,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.ff = FeedForward(d_out)
        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        context_length,
        drop_rate,
        n_layers,
        n_heads,
        qkv_bias=False,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.dropout = nn.Dropout(drop_rate)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_in=emb_dim,
                    d_out=emb_dim,
                    context_length=context_length,
                    dropout=drop_rate,
                    num_heads=n_heads,
                    qkv_bias=qkv_bias,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x):
        batch_size, context_length = x.shape
        tok_embed = self.tok_emb(x)
        pos_embed = self.pos_emb(torch.arange(context_length, device=x.device))
        x = tok_embed + pos_embed
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
