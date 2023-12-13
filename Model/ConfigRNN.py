import torch
import torch.nn as nn
import torch.optim as optim

class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size=256, num_layers=4, num_heads=8, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layers = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.decoder = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
