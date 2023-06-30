from torch import nn
from detoken_constants import *

import torch
import math


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # self.embed_tokens = EmbeddingToken(sp_vocab_size, d_model)
        self.self_attn_layer_norm = LayerNormalization()
        self.self_attn = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.final_layer_norm = LayerNormalization()
        self.fc1 = FeedFowardLayer_FC1_ReLU()
        self.fc2 = FeedFowardLayer_FC2()
        self.drop_out_2 = nn.Dropout(drop_out_rate)
        self.relu = nn.ReLU()

    def forward(self, x, e_mask):
        x_1 = self.self_attn_layer_norm(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.self_attn(x_1, x_1, x_1, mask=e_mask)
        ) # (B, L, d_model)
        x_2 = self.final_layer_norm(x) # (B, L, d_model)
        x_2 = self.relu(self.fc1(x_2))
        # x_2 = self.fc1(x_2)
        x = x + self.drop_out_2(self.fc2(x_2)) # (B, L, d_model)

        return x # (B, L, d_model)
    


# class EmbeddingToken(nn.Module):
#     def __init__(self, vocab_size, d_model):
#         super().__init__()
#         self.embed_tokens = nn.Embedding(vocab_size, d_model)
    
#     def forward(self, vocab_size, d_model):
#         return embed_tokens(vocab_size, d_model)

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn_layer_norm = LayerNormalization()
        self.self_attn = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(drop_out_rate)

        self.encoder_attn_layer_norm = LayerNormalization()
        self.encoder_attn = MultiheadAttention()
        self.drop_out_2 = nn.Dropout(drop_out_rate)

        self.final_layer_norm = LayerNormalization()
        self.fc1 = FeedFowardLayer_FC1_ReLU()
        self.fc2 = FeedFowardLayer_FC2()

        self.drop_out_3 = nn.Dropout(drop_out_rate)
        # self.fc1 = self.
        # self.fc2 = self.build_fc2
        self.drop_out_2 = nn.Dropout(drop_out_rate)

    def forward(self, x, e_output, e_mask,  d_mask):
        x_1 = self.self_attn_layer_norm(x) # (B, L, d_model)
        x = x + self.drop_out_1(
            self.self_attn(x_1, x_1, x_1, mask=d_mask)
        ) # (B, L, d_model)
        x_2 = self.encoder_attn_layer_norm(x) # (B, L, d_model)
        x = x + self.drop_out_2(
            self.encoder_attn(x_2, e_output, e_output, mask=e_mask)
        ) # (B, L, d_model)
        x_3 = self.final_layer_norm(x) # (B, L, d_model)
        x_3 = self.fc1(x_3)
        x = x + self.drop_out_3(self.fc2(x_3)) # (B, L, d_model)

        return x # (B, L, d_model)


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inf = 1e9

        # W^Q, W^K, W^V in the paper
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_out_rate)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.q_proj(q).view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)
        k = self.k_proj(k).view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)
        v = self.v_proj(v).view(input_shape[0], -1, num_heads, d_k) # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask) # (B, num_heads, L, d_k)
        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, d_model) # (B, L, d_model)

        return self.out_proj(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(d_k)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1) # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v) # (B, num_heads, L, d_k)

        return attn_values


def FeedFowardLayer_FC1_ReLU():
    return nn.Linear(d_model, d_ff, bias=True)

def FeedFowardLayer_FC2():
    return nn.Linear(d_ff, d_model, bias=True)
# class FeedFowardLayer_FC1_ReLU(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(d_model, d_ff, bias=True)
#         self.relu = nn.ReLU()
#         # fc2 = nn.Linear(d_ff, d_model, bias=True)
#         dropout = nn.Dropout(drop_out_rate)

#     def forward(self, x):
#         x = self.relu(self.fc(x)) # (B, L, d_ff)
#         x = self.dropout(x)
#         # x = self.linear_2(x) # (B, L, d_model)

#         return x

# class FeedFowardLayer_FC2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # fc1 = nn.Linear(d_model, d_ff, bias=True)
#         # relu = nn.ReLU()
#         self.fc = nn.Linear(d_ff, d_model, bias=True)
#         # dropout = nn.Dropout(drop_out_rate)

#     def forward(self, x):
#         # x = self.relu(self.linear_1(x)) # (B, L, d_ff)
#         # x = self.dropout(x)
#         x = self.fc(x) # (B, L, d_model)

#         return x

def LayerNormalization():
    return nn.LayerNorm([d_model], elementwise_affine=True, eps=1-6)



# class LayerNormalization(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#         # self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=self.eps)

#     def forward(self, x):
#         x = nn.LayerNorm([d_model], elementwise_affine=True, eps=self.eps)

#         return x


# class PositionalEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Make initial positional encoding matrix with 0
#         pe_matrix= torch.zeros(seq_len, d_model) # (L, d_model)

#         # Calculating position encoding values
#         for pos in range(seq_len):
#             for i in range(d_model):
#                 if i % 2 == 0:
#                     pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
#                 elif i % 2 == 1:
#                     pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

#         pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
#         self.embed_positions = pe_matrix.to(device=device).requires_grad_(False)

#     def forward(self, x):
#         x = x * math.sqrt(d_model) # (B, L, d_model)
#         x = x + self.embed_positions # (B, L, d_model)

#         return x
