import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

DATASET_DIR = './datasets'

class MovieLensDataset(Dataset):
  def __init__(self, df):
    super().__init__()
    self.users = np.array(df['userId'] - 1)
    self.items = np.array(df['movieId'] - 1)
    self.ratings = np.array(df['rating'])
    assert len(self.users) == len(self.items) == len(self.ratings)

  def __getitem__(self, index):
    return self.users[index], self.items[index], self.ratings[index]

  def __len__(self):
    return len(self.users)


def loadDataset(dataset, batch_size=32):
  # assert dataset in ['movielens-small', 'ml-10m', ]
  dataset_path = os.path.join(DATASET_DIR, dataset)
  columns = ['userId', 'movieId', 'rating']

  train_dataset_path = os.path.join(dataset_path, "split_cache", "__train.csv")
  validate_dataset_path = os.path.join(dataset_path, "split_cache", "__validate.csv")
  test_dataset_path = os.path.join(dataset_path, "split_cache", "__test.csv")

  if os.path.isfile(train_dataset_path):
    train = pd.read_csv(train_dataset_path, usecols=columns)
    validate = pd.read_csv(validate_dataset_path, usecols=columns)
    test = pd.read_csv(test_dataset_path, usecols=columns)
    dfs = [train, validate, test]
    n_users = max(df['userId'].max() for df in dfs)
    n_items = max(df['movieId'].max() for df in dfs)
    scale = max(df['rating'].max() for df in dfs)

  else:
    df = pd.read_csv(os.path.join(dataset_path, "ratings.csv"), usecols=columns)
    n_users = df['userId'].max()
    n_items = df['movieId'].max()
    scale = df['rating'].max()

    train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

    os.mkdir(os.path.join(dataset_path, "split_cache"))
    train.to_csv(train_dataset_path, index=False)
    validate.to_csv(validate_dataset_path, index=False)
    test.to_csv(test_dataset_path, index=False)

  train = DataLoader(MovieLensDataset(train), batch_size=batch_size)
  validate = DataLoader(MovieLensDataset(validate), batch_size=batch_size)
  test = DataLoader(MovieLensDataset(test), batch_size=batch_size)

  return train, validate, test, n_users, n_items, scale
  