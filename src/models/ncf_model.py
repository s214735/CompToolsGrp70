# src/models/ncf_model.py
import torch
import torch.nn as nn

class NeuralCF(nn.Module):
    """
    Simple NCF (MLP) for explicit ratings.
    """
    def __init__(self, n_users, n_items, emb_dim=64, hidden=(128, 64), dropout=0.1):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        layers = []
        in_dim = emb_dim * 2
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        x = torch.cat([u, v], dim=-1)
        u = self.user_emb(user_idx); v = self.item_emb(item_idx)
        pred = self.mlp(x).squeeze(-1)
        pred = pred + self.user_bias(user_idx).squeeze(-1) + self.item_bias(item_idx).squeeze(-1)
        return pred

    @torch.no_grad()
    def item_embeddings(self):
        return self.item_emb.weight  # (n_items, emb_dim)

    @torch.no_grad()
    def user_embeddings(self):
        return self.user_emb.weight  # (n_users, emb_dim)
