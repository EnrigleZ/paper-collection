import torch
import torch.nn.functional as F
from torch import nn

class TestModel(nn.Module):
  def __init__(self, n_users, n_items, hidden_dims, dropout=False, scale=None):
    super().__init__()
    self.n_users = n_users
    self.n_items = n_items
    self.hidden_dims = hidden_dims
    self.dropout = dropout
    self.scale = scale

    self.u_embedding = nn.Embedding(n_users, hidden_dims)
    self.i_embedding = nn.Embedding(n_items, hidden_dims)

    self.fc = nn.Linear(hidden_dims * 2, hidden_dims)
    self.output_fc = nn.Linear(hidden_dims, 1)

  def forward(self, users, items):
    user_vecs = self.u_embedding(users)
    item_vecs = self.i_embedding(items)

    # [batch, hidden * 2]
    features = torch.cat([user_vecs, item_vecs], dim=1)

    features = self.fc(features)
    features = F.relu(features)
    features = F.dropout(features, self.dropout, training=self.training)

    logits = self.output_fc(features)
    outputs = torch.sigmoid(logits)

    if self.scale is not None:
      outputs = outputs * self.scale

    return outputs

