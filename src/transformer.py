from torch import nn
from detoken_constants import *
from layers import *

import torch


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        # self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        # self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)

        self.encoder = Encoder()
        self.decoder = Decoder()
        src_embedding = self.encoder.embed_tokens
        trg_embedding = self.encoder.embed_tokens
        positional_encoder = self.encoder.embed_positions
        positional_decoder = self.decoder.embed_positions

        # output_projection = nn.Linear(d_model, self.trg_vocab_size)
        output_projection = self.decoder.output_projection
        self.softmax = nn.LogSoftmax(dim=-1)

    # def upgrade_state_dict_named(self, state_dict, name):
    #     """Upgrade a (possibly old) state dict for new versions of fairseq."""
    #     if "encoder.embed_positions.embed_positions" in state_dict:
    #         state_dict["encoder.embed_positions._float_tensor"] = state_dict.pop("encoder.embed_positions.weight")
        
    #     if "decoder.embed_positions.embed_positions" in state_dict:
    #         state_dict["decoder.embed_positions._float_tensor"] = state_dict.pop("decoder.embed_positions.weight")
        
    #     return state_dict
        
    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        src_input = src_embedding(src_input) # (B, L) => (B, L, d_model)
        trg_input = trg_embedding(trg_input) # (B, L) => (B, L, d_model)
        src_input = positional_encoder(src_input) # (B, L, d_model) => (B, L, d_model)
        trg_input = positional_decoder(trg_input) # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask) # (B, L, d_model)
        d_output = self.decoder(trg_input, e_output, e_mask, d_mask) # (B, L, d_model)

        output = self.softmax(output_projection(d_output)) # (B, L, d_model) => # (B, L, trg_vocab_size)

        return output
    # for key in list(state_dict.keys()):
    #     state_dict[key.replace("decoder.embed_positions.embed_positions", "decoder.embed_positions._float_tensor"). \
    #                 replace("encoder.embed_positions.embed_positions", "encoder.embed_positions._float_tensor")] = state_dict.pop(key)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.version=0
        self.layers = nn.ModuleList([EncoderLayer() for i in range(num_layers)])
        # self.layer_norm = LayerNormalization()
        self.embed_tokens = nn.Embedding(33336, d_model)
        self.embed_positions = PositionalEncoder()

    def forward(self, x, e_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_mask)

        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trg_vocab_size = 33336
        self.layers = nn.ModuleList([DecoderLayer() for i in range(num_layers)])
        # self.layer_norm = LayerNormalization()
        self.embed_tokens = nn.Embedding(33336, d_model)
        self.embed_positions = PositionalEncoder()

        self.output_projection = nn.Linear(d_model, self.trg_vocab_size, bias=False)

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return x
    
    # def PositionalEncoder():
    #     pe_matrix = torch.zeros(seq_len, d_model)  # (L, d_model)
    #     for pos in range(seq_len):
    #         for i in range(d_model):
    #             if i % 2 == 0:
    #                 pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
    #             elif i % 2 == 1:
    #                 pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))
    #     pe_matrix = pe_matrix.unsqueeze(0)  # (1, L, d_model)
    #     # self.register_buffer("embed_positions", pe_matrix)

    #     x = x * math.sqrt(d_model)
    #     x = x + embed_positions
    #     return x


class PositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        pe_matrix = torch.zeros(seq_len, d_model)  # (L, d_model)
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))
        pe_matrix = pe_matrix.unsqueeze(0)  # (1, L, d_model)
        # pe_matrix = torch.squeeze(pe_matrix, dim=-1)
        sum_value = torch.sum(pe_matrix)
        new_tensor = torch.tensor([sum_value])
        self.register_buffer("embed_positions", new_tensor)

    def forward(self, x):
        x = x * math.sqrt(d_model)  # (B, L, d_model)
        x = x + self.embed_positions  # (B, L, d_model)
        # x = torch.mean(x, dim=(1, 2), keepdim=True)  # (B, 1)

        return x