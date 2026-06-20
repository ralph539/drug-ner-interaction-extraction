#! /usr/bin/python3

import sys, os
import random

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torchinfo import summary

from dataset import *
from codemaps import *

from network import nercLSTM, criterion

# [MOD-1.2] default seed, can be overridden via CLI param `seed=N`.
# set_seed() is called again inside do_train() after params parsing so
# the CLI override actually takes effect.
def set_seed(seed:int=2345):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(2345)

# use gpu if available
used_device = "cuda:0" if torch.cuda.is_available() else "cpu"

#----------------------------------------------
# [MOD-1.2] Build optimizer from params instead of hard-coding Adam().
# Supports: adam | adamw | sgd, with learning_rate and weight_decay.
def build_optimizer(network, params):
    name = str(params.get('optimizer', 'adam')).lower()
    lr = float(params.get('learning_rate', 1e-3))
    wd = float(params.get('weight_decay', 0.0))
    if name == 'adamw':
        return optim.AdamW(network.parameters(), lr=lr, weight_decay=wd)
    if name == 'sgd':
        mom = float(params.get('momentum', 0.9))
        return optim.SGD(network.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    return optim.Adam(network.parameters(), lr=lr, weight_decay=wd)

#----------------------------------------------
def train(network, epoch, train_loader, optimizer):   # [MOD-1.2] optimizer passed in
   network.to(torch.device(used_device))

   network.train()
   seen = 0
   acc_loss = 0
   for batch_idx, X in enumerate(train_loader):
      target = X.pop()
      optimizer.zero_grad()
      output = network(*X)
      output = output.flatten(0,1)
      target = target.flatten(0,1)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      acc_loss += loss.item()
      avg_loss = acc_loss/(batch_idx+1)
      seen += len(X[0])
      print('Train Epoch {}: batch:{}/{} sentence:{}/{} [{:.2f}%] Loss:{:.6f}\r'.format(
                   epoch,
                   batch_idx+1, len(train_loader),
                   seen, len(train_loader.dataset),
                   100.*(batch_idx+1)/len(train_loader),
                   avg_loss),
            flush=True, end='')
   print()

#----------------------------------------------
def validation(network, val_loader):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
       for X in val_loader:
          target = X.pop()
          output = network(*X)
          output = output.flatten(0,1)
          target = target.flatten(0,1)
          test_loss += criterion(output, target).item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
          total += target.size()[0]
    test_loss /= len(val_loader)
    acc = 100.*correct/total
    print('Validation set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
               test_loss,
               correct, total,
               acc))
    return acc

#----------------------------------------------
def encode_dataset(ds, codes, params) :
   X = codes.encode_words(ds)
   y = codes.encode_labels(ds)
   if used_device == "cuda:0" :
      X = [x.to(torch.device(used_device)) for x in X]
      y = y.to(torch.device(used_device))
   return DataLoader(TensorDataset(*X, y), 
                     batch_size=params['batch_size'])


#----------------------------------------------
def do_train(trainfile, valfile, params, modelname) :

    # set default values if some parameter is missing
    if 'max_len' not in params : params['max_len'] = 150
    if 'suf_len' not in params : params['suf_len'] = 5
    if 'batch_size' not in params : params['batch_size'] = 16
    if 'epochs' not in params : params['epochs'] = 10

    # [MOD-1.2] cast the int-only hyperparameters because run.py / CLI can
    # pass them in as strings. (Float params like learning_rate are cast
    # lazily inside network.py / build_optimizer, so they stay as strings.)
    for k in ('max_len', 'suf_len', 'batch_size', 'epochs'):
        params[k] = int(params[k])

    # [MOD-1.2] seed can now be overridden from the CLI
    if 'seed' in params:
        set_seed(int(params['seed']))

    # load pickle datasets (or parse if needed)
    traindata = Dataset(trainfile)
    valdata = Dataset(valfile)

    # create indexes from training data
    codes  = Codemaps(traindata, params)
    # encode datasets
    train_loader = encode_dataset(traindata, codes, params)
    val_loader = encode_dataset(valdata, codes, params)

    # build network
    # [MOD-1.2] pass params so architecture hyperparameters can be tuned
    network = nercLSTM(codes, params)

    # [MOD-1.2] build optimizer once, outside the epoch loop (original code
    # re-created it every epoch, which discards Adam's moment state!)
    optimizer = build_optimizer(network, params)

    summary(network)

    # save indexs
    os.makedirs(modelname,exist_ok=True)
    torch.save(network, os.path.join(modelname,"network.nn"))
    codes.save(os.path.join(modelname,"codemaps"))
    # train each epoch, keep the best model on validation
    best = 0       
    for epoch in range(params["epochs"]):
       train(network, epoch, train_loader, optimizer)   # [MOD-1.2] pass optimizer
       acc = validation(network, val_loader)
       if acc>best :
          best = acc
          torch.save(network, os.path.join(modelname,f"network.nn"))

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py train.pck devel.pck modelname [batch_size=N] [max_len=N] [suf_len=N]
## --

if __name__ == "__main__" :
    # files to process
    trainfile = sys.argv[1]
    validationfile = sys.argv[2]
    modelname = sys.argv[3]

    params={}
    # [MOD-1.2] keep command-line values as strings. network.py / build_optimizer
    # cast each value to the right numeric type, so floats (e.g. learning_rate)
    # are no longer truncated to int.
    for p in sys.argv[4:] :
       k,v = p.split("=")
       params[k]=v
       
    do_train(trainfile, validationfile, params, modelname)


