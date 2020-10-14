import torch
from torch.utils.data import DataLoader

from data import loadDataset
from models.test import TestModel
from train import train_one_epoch, test_one_epoch
from utils import save_model, load_model

import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', nargs='?', default='movielens-small',
                      help='Choose a dataset.')
  parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs.')
  parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size.')
  parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden vector dimensions.')
  parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate.')
  parser.add_argument('--dropout', type=float, default=0,
                      help='Add dropout layer after each dense layer, with p = dropout_prob')
  parser.add_argument('--cuda', type=bool, default=True,
                      help='whether use gpu when it is available')
  parser.add_argument('--mode', type=str, default='train',
                      help='train or test')
  parser.add_argument('--model', type=str, default='test')

  return parser.parse_args()

def __main__():
  args = parse_args()
  if args.cuda and torch.cuda.is_available():
    device = torch.device('cuda:0')
  else:
    device = torch.device('cpu')

  train, validate, test, n_users, n_items, scale = loadDataset(args.dataset)
  print("n_users:", n_users, "n_items:", n_items)

  model = TestModel(n_users, n_items, 64, dropout=0.15, scale=scale).to(device)
  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters())
  load_model(model, args)

  if args.mode == 'train':
    best_loss = test_one_epoch(model, validate, loss_fn, device)
    for epoch in range(args.epochs):
      train_one_epoch(model, train, loss_fn, optimizer, device)
      loss = test_one_epoch(model, validate, loss_fn, device)
      if loss < best_loss:
        best_loss = loss
        save_model(model, args)

  loss = test_one_epoch(model, test, loss_fn, device, mode='test')

if __name__ == "__main__":
    __main__()
  