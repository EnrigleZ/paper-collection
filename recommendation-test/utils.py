import os
import torch

import time

SAVE_DIR = "./save"

def get_model_identifier(args):
  dataset = args.dataset
  hidden = args.hidden_dim
  model = args.model
  cuda = "gpu" if args.cuda and torch.cuda.is_available() else "cpu"

  name = "{}-{}-{}-{}".format(dataset, model, hidden, cuda)
  return name

def save_model(model, args):
  name = get_model_identifier(args) + ".pt"
  path = os.path.join(SAVE_DIR, name)
  torch.save(model.state_dict(), path)
  print("-- Save to '{}'\n".format(path))

def load_model(model, args):
  name = get_model_identifier(args) + ".pt"
  path = os.path.join(SAVE_DIR, name)
  if os.path.isfile(path):
    model.load_state_dict(torch.load(path))
    print("--Load from '{}'".format(path))
    return True
  return False

def timing(f):
  def wrap(*args, **kwargs):
    time1 = time.time()
    ret = f(*args, **kwargs)
    time2 = time.time()
    print('{:s} function took {:.3f} s'.format(f.__name__, time2 - time1))

    return ret
  return wrap
