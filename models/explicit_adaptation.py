import torch
import torch.nn as nn

class ExplicitAdaptation(nn.Module):
    def __init__(self, style_dim, embed_dim):
        super(ExplicitAdaptation, self).__init__()
        self.delta_w = nn.Linear(style_dim, embed_dim)
        self.scale = nn.Parameter(torch.tensor(5.0))    # Scaling Factor

    def forward(self, keys, values, style_embeddings):
        # Project style embeddings to match the embedding dimension of keys/values
        delta_keys = self.delta_w(style_embeddings)  # Shape: (1, 1, embed_dim)
        delta_values = self.delta_w(style_embeddings)  # Shape: (1, 1, embed_dim)

        # Expand style embeddings to match the batch size
        delta_keys = delta_keys.expand(keys.size(0), -1, -1)  # Shape: (batch_size, 1, embed_dim)
        delta_values = delta_values.expand(values.size(0), -1, -1)  # Shape: (batch_size, 1, embed_dim)

        keys = keys + self.scale * delta_keys
        values = values + self.scale * delta_values

        # print(f"Scale parameter: {self.scale.item()}")

        return keys, values