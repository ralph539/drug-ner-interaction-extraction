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

from network import ddiCNN, criterion

# use gpu if available
used_device = "cuda:0" if torch.cuda.is_available() else "cpu"

#----------------------------------------------
def set_seed(seed):
   random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.backends.cudnn.deterministic = True

# default seed
set_seed(2345)

#----------------------------------------------
def train(network, epoch, train_loader, lr=1e-3):
   # [MOD-2.2] surface learning rate (defaults match Adam's sklearn default of 1e-3)
   optimizer = optim.Adam(network.parameters(), lr=lr)
   network.to(torch.device(used_device))

   network.train()
   seen = 0
   acc_loss = 0
   for batch_idx, X in enumerate(train_loader):
      target = X.pop()
      optimizer.zero_grad()
      output = network(*X)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      acc_loss += loss.item()
      avg_loss = acc_loss/(batch_idx+1)
      seen += len(target)
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
          # accumulate loss and accuracy statistics 
          test_loss += criterion(output, target).item()
          pred = output.data.argmax(1)
          targ = target.data.argmax(1)
          correct += pred.eq(targ.data.view_as(pred)).sum()
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

    # [MOD-2.2] surface seed as a per-run knob (Phase S seed audit)
    if 'seed' in params:
       set_seed(int(params['seed']))

    # set default values if some parameter is missing
    if 'max_len' not in params : params['max_len'] = 150
    # [MOD-2.2] suf_len=0 means "no suffix input"; default to 0 to match
    # the shipped baseline (suf_len was previously ignored entirely)
    if 'suf_len' not in params : params['suf_len'] = 0
    if 'pref_len' not in params : params['pref_len'] = 0
    if 'batch_size' not in params : params['batch_size'] = 16
    if 'epochs' not in params : params['epochs'] = 10
    lr = float(params.get('lr', 1e-3))

    # [MOD-2.2] architecture params (Phase A)
    arch_params = {k: params[k] for k in (
       'emb_lw','emb_l','emb_p','emb_s','emb_pr','emb_e','emb_w',
       'lstm_h','lstm_layers','cnn_out','cnn_k','drop1','drop2',
       'use_lstm','use_cnn'
    ) if k in params}

    # load pickle datasets (or parse if needed)
    traindata = Dataset(trainfile)
    valdata = Dataset(valfile)

    # create indexes from training data
    codes  = Codemaps(traindata, params)
    # encode datasets
    train_loader = encode_dataset(traindata, codes, params)
    val_loader = encode_dataset(valdata, codes, params)

    # build network
    network = ddiCNN(codes, arch_params=arch_params)

    summary(network)

    # save indexs
    os.makedirs(modelname,exist_ok=True)
    torch.save(network, os.path.join(modelname,"network.nn"))
    codes.save(os.path.join(modelname,"codemaps"))
    # train each epoch, keep the best model on validation
    best = 0
    best_epoch = -1
    for epoch in range(params["epochs"]):
       train(network, epoch, train_loader, lr=lr)
       acc = validation(network, val_loader)
       if acc>best :
          best = acc
          best_epoch = epoch
          torch.save(network, os.path.join(modelname,f"network.nn"))
    print(f"[MOD-2.2] Best validation accuracy: {best:.2f}% at epoch {best_epoch}")

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
    for p in sys.argv[4:] :
       k,v = p.split("=")
       # [MOD-2.2] some params are floats / bool / strings; do not force int
       try:
          params[k] = int(v)
       except ValueError:
          try:
             params[k] = float(v)
          except ValueError:
             params[k] = v
       
    do_train(trainfile, validationfile, params, modelname)


