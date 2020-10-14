import torch
import numpy as np

from utils import timing

@timing
def train_one_epoch(model: torch.nn.Module, train, loss_fn, optimizer, device):
  steps = 0
  total_loss = 0
  total_record = 0

  model.train()
  n_batchs = len(train)
  for users, items, ratings in train:
    users = users.to(device)
    items = items.to(device)
    ratings = ratings.to(device)
    optimizer.zero_grad()

    predictions = model(users, items)
    ratings = ratings.float().view(predictions.size())  
    loss = loss_fn(predictions, ratings)

    total_loss += loss.detach().cpu() * len(users)
    total_record += len(users)

    loss.backward()
    optimizer.step()
    if steps % 10000 == 0:
      print("Training Step {} / {}, loss: {:.4f}".format(steps, n_batchs, np.sqrt(loss.detach().cpu().numpy())))
    steps += 1

  total_rmse_loss = np.sqrt(total_loss / total_record)
  print("train MSELoss: {:.6f}".format(total_rmse_loss))
  return total_rmse_loss

@timing
def test_one_epoch(model: torch.nn.Module, test, loss_fn, device, mode='eval'):
  total_loss = 0
  total_record = 0
  steps = 0

  with torch.no_grad():
    model.eval()
    n_batchs = len(test)
    for users, items, ratings in test:
      users = users.to(device)
      items = items.to(device)
      ratings = ratings.to(device)

      predictions = model(users, items)
      ratings = ratings.float().view(predictions.size())
      loss = loss_fn(predictions, ratings)

      total_loss += loss.detach().cpu() * len(users)
      total_record += len(users)

      if steps % 10000 == 0:
        print("Eval Step {} / {}".format(steps, n_batchs))
      steps += 1

  total_rmse_loss = np.sqrt(total_loss / total_record)
  print("{} MSELoss: {:.6f}".format(mode, total_rmse_loss))
  return total_rmse_loss

def np_rmse(preds, ratings):
  return np.sqrt(((preds - ratings)**2).mean())
