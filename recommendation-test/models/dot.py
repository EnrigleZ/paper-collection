import torch
import torch.nn.functional as F
from torch import nn

class DotModel(nn.Module):
  def __init__(self, n_users, n_items, hidden_dims, dropout=False, scale=None):
    super().__init__()
    self.n_users = n_users
    self.n_items = n_items
    self.hidden_dims = hidden_dims
    self.dropout = dropout
    self.scale = scale

    self.u_embedding = nn.Embedding(n_users, hidden_dims)
    self.i_embedding = nn.Embedding(n_items, hidden_dims)

    self.output_fc = nn.Linear(hidden_dims, 1)

  def forward(self, users, items):
    user_vecs = self.u_embedding(users)
    item_vecs = self.i_embedding(items)

    # [batch, hidden]
    multiply = user_vecs * item_vecs
    multiply = F.dropout(multiply, self.dropout, training=self.training)
    logits = self.output_fc(multiply)
    outputs = torch.sigmoid(logits)

    if self.scale is not None:
      outputs = outputs * self.scale

    return outputs

