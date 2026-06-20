
import torch
import torch.nn as nn
import torch.nn.functional as func

criterion = nn.CrossEntropyLoss()

class ddiCNN(nn.Module):

   def __init__(self, codes, arch_params=None) :
      super(ddiCNN, self).__init__()
      # [MOD-2.2] optional architecture params for Phase A experiments
      ap = arch_params or {}
      embLW_sz = int(ap.get('emb_lw', 100))
      embL_sz  = int(ap.get('emb_l',  100))
      embP_sz  = int(ap.get('emb_p',  50))
      embS_sz  = int(ap.get('emb_s',  30))   # suffix
      embPR_sz = int(ap.get('emb_pr', 30))   # prefix
      embE_sz  = int(ap.get('emb_e',  20))   # etype indicator
      embW_sz  = int(ap.get('emb_w',  100))  # case-sensitive form
      embRP_sz = int(ap.get('emb_rp', 20))   # relative-position bucket
      lstm_out_sz = int(ap.get('lstm_h', 100))
      lstm_layers = int(ap.get('lstm_layers', 1))
      cnn_out_sz = int(ap.get('cnn_out', 64))
      cnn_kernel = int(ap.get('cnn_k', 2))
      dropout1 = float(ap.get('drop1', 0.2))
      dropout2 = float(ap.get('drop2', 0.2))
      self.use_lstm = bool(ap.get('use_lstm', True))
      self.use_cnn = bool(ap.get('use_cnn', True))

      # get sizes
      n_lc_words = codes.get_n_lc_words()
      n_lemmas = codes.get_n_lemmas()
      n_pos = codes.get_n_pos()
      n_labels = codes.get_n_labels()
      self.max_len = codes.maxlen

      # base 3 embeddings (always)
      self.embLW = nn.Embedding(n_lc_words, embLW_sz, padding_idx=0)
      self.embL  = nn.Embedding(n_lemmas,   embL_sz,  padding_idx=0)
      self.embP  = nn.Embedding(n_pos,      embP_sz,  padding_idx=0)
      in_size = embLW_sz + embL_sz + embP_sz

      # [MOD-2.2] optional extra embedding inputs
      self.has_suf  = codes.use_suffix()
      self.has_pref = codes.use_prefix()
      self.has_et   = codes.use_etype()
      self.has_form = codes.use_form()
      self.has_relpos = codes.use_relpos()  # [MOD-2.2]
      if self.has_suf:
         self.embS = nn.Embedding(codes.get_n_suffixes(), embS_sz, padding_idx=0)
         in_size += embS_sz
      if self.has_pref:
         self.embPR = nn.Embedding(codes.get_n_prefixes(), embPR_sz, padding_idx=0)
         in_size += embPR_sz
      if self.has_et:
         self.embE = nn.Embedding(codes.get_n_etypes(), embE_sz, padding_idx=0)
         in_size += embE_sz
      if self.has_form:
         self.embW = nn.Embedding(codes.get_n_words(), embW_sz, padding_idx=0)
         in_size += embW_sz
      if self.has_relpos:
         # two separate embeddings (one for distance to DRUG1, one to DRUG2)
         # sharing weights across DRUG1 and DRUG2 is also valid — we use two
         # to allow the model to learn separate priors for "left of the pair"
         # vs "right of the pair".
         self.embRP1 = nn.Embedding(codes.get_n_relpos(), embRP_sz, padding_idx=0)
         self.embRP2 = nn.Embedding(codes.get_n_relpos(), embRP_sz, padding_idx=0)
         in_size += 2 * embRP_sz

      if self.use_lstm:
         self.lstm = nn.LSTM(in_size, lstm_out_sz, num_layers=lstm_layers,
                             bidirectional=True, batch_first=True,
                             dropout=(dropout1 if lstm_layers > 1 else 0.0))
         seq_feat_dim = 2 * lstm_out_sz
      else:
         seq_feat_dim = in_size

      self.drop1 = nn.Dropout(dropout1)

      if self.use_cnn:
         self.cnn1 = nn.Conv1d(seq_feat_dim, cnn_out_sz,
                               kernel_size=cnn_kernel, stride=1, padding='same')
         final_feat_dim = cnn_out_sz
      else:
         final_feat_dim = seq_feat_dim
      self.drop2 = nn.Dropout(dropout2)

      self.out = nn.Linear(final_feat_dim, n_labels)


   def forward(self, *inputs):
      # First 3 inputs are always (lw, l, p). Extras follow in the order
      # declared by Codemaps.encode_words: suf, pref, etype, form.
      idx = 0
      lw = inputs[idx]; idx += 1
      l  = inputs[idx]; idx += 1
      p  = inputs[idx]; idx += 1
      parts = [self.embLW(lw), self.embL(l), self.embP(p)]
      if self.has_suf:
         parts.append(self.embS(inputs[idx])); idx += 1
      if self.has_pref:
         parts.append(self.embPR(inputs[idx])); idx += 1
      if self.has_et:
         parts.append(self.embE(inputs[idx])); idx += 1
      if self.has_form:
         parts.append(self.embW(inputs[idx])); idx += 1
      if self.has_relpos:
         parts.append(self.embRP1(inputs[idx])); idx += 1
         parts.append(self.embRP2(inputs[idx])); idx += 1
      x = torch.cat(parts, dim=2)

      if self.use_lstm:
         x = self.lstm(x)[0]
      x = self.drop1(x)
      x = x.permute(0, 2, 1)
      x = func.max_pool1d(x, kernel_size=4, stride=1, padding=1)
      if self.use_cnn:
         x = self.cnn1(x)
         x = func.relu(x)
      x = func.max_pool1d(x, kernel_size=self.max_len - 1)
      x = x.flatten(start_dim=1)
      x = self.drop2(x)
      x = self.out(x)
      return x
   


