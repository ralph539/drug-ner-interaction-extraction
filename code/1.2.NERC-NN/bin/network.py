
import torch
import torch.nn as nn
import torch.nn.functional as func


criterion = nn.CrossEntropyLoss()

# =============================================================================
# [MOD-1.2] Configurable nercLSTM
# -----------------------------------------------------------------------------
# The original baseline (see 1.2.NERC-nn.pdf) used hard-coded hyperparameters:
#     embW_sz=100, embS_sz=50, lstm_out_sz=200, dropout=0.1, 1 BiLSTM layer.
# To be able to run architecture / hyperparameter experiments without editing
# the code each time, we now accept a `params` dict that can override any
# structural choice. If a field is missing, we fall back to the baseline value
# so old behaviour is preserved.
#
# Exposed knobs:
#   Embedding dims:  embLW_sz, embW_sz, embS_sz
#                    embP_sz (prefix), embL_sz (lemma), embPOS_sz (PoS)
#   Feature toggles: use_pref (0/1), use_lemma (0/1), use_pos (0/1)
#                    — opt-in so defaults reproduce the baseline exactly
#   LSTM:            lstm_hidden, lstm_layers, lstm_dropout
#   Head:            fc_hidden, use_layernorm (0/1), activation (relu|tanh|gelu)
#   Regularisation:  emb_dropout
#
# The forward() signature always takes all 7 channels produced by
# codemaps.encode_words(); unused channels are silently ignored. This keeps
# the dataset / predict pipelines unchanged.
# =============================================================================

class nercLSTM(nn.Module):
   def __init__(self, codes, params=None) :
      super(nercLSTM, self).__init__()

      if params is None: params = {}

      # [MOD-1.2] optional pretrained word vectors for the lowercase-word
      # channel. When enabled, the embLW layer is initialised from spaCy
      # word vectors (en_core_web_md by default, 300d). Its dimension is
      # forced to match the pretrained dim regardless of embLW_sz.
      use_pretrained = bool(int(params.get('pretrained', 0)))
      pretrained_freeze = bool(int(params.get('pretrained_freeze', 0)))
      pretrained_model  = str(params.get('pretrained_model', 'en_core_web_md'))

      embLW_sz     = int(params.get('embLW_sz',     100))
      embW_sz      = int(params.get('embW_sz',      100))
      embS_sz      = int(params.get('embS_sz',       50))
      embP_sz      = int(params.get('embP_sz',       50))   # [MOD-1.2] prefix
      embL_sz      = int(params.get('embL_sz',      100))   # [MOD-1.2] lemma
      embPOS_sz    = int(params.get('embPOS_sz',     25))   # [MOD-1.2] pos
      lstm_hidden  = int(params.get('lstm_hidden',  200))
      lstm_layers  = int(params.get('lstm_layers',    1))
      lstm_dropout = float(params.get('lstm_dropout', 0.0))
      fc_hidden    = int(params.get('fc_hidden',    200))
      emb_dropout  = float(params.get('emb_dropout', 0.1))
      use_layernorm = bool(int(params.get('use_layernorm', 0)))
      activation   = str(params.get('activation', 'relu')).lower()

      # [MOD-1.2] opt-in flags: off by default → baseline unchanged
      self.use_pref  = bool(int(params.get('use_pref',  0)))
      self.use_lemma = bool(int(params.get('use_lemma', 0)))
      self.use_pos   = bool(int(params.get('use_pos',   0)))

      n_lc_words = codes.get_n_lc_words()
      n_words    = codes.get_n_words()
      n_sufs     = codes.get_n_sufs()
      n_prefs    = codes.get_n_prefs()    # [MOD-1.2]
      n_lemmas   = codes.get_n_lemmas()   # [MOD-1.2]
      n_pos      = codes.get_n_pos()      # [MOD-1.2]
      n_feat     = codes.get_n_features()
      n_labels   = codes.get_n_labels()

      # ---- embeddings (sizes are now parametric) -------------------------
      # [MOD-1.2] pretrained word vectors (spaCy md) → embLW
      if use_pretrained:
         mat, pre_dim = codes.build_pretrained_matrix(pretrained_model)
         embLW_sz = pre_dim
         self.embLW = nn.Embedding.from_pretrained(
            mat, freeze=pretrained_freeze, padding_idx=0)
      else:
         self.embLW = nn.Embedding(n_lc_words, embLW_sz)
      self.embW  = nn.Embedding(n_words,    embW_sz)
      self.embS  = nn.Embedding(n_sufs,     embS_sz)

      self.dropLW = nn.Dropout(emb_dropout)
      self.dropW  = nn.Dropout(emb_dropout)
      self.dropS  = nn.Dropout(emb_dropout)

      lstm_in_size = embLW_sz + embW_sz + embS_sz + n_feat

      # [MOD-1.2] optional prefix embedding
      if self.use_pref:
         self.embP  = nn.Embedding(n_prefs, embP_sz)
         self.dropP = nn.Dropout(emb_dropout)
         lstm_in_size += embP_sz

      # [MOD-1.2] optional lemma embedding
      if self.use_lemma:
         self.embL  = nn.Embedding(n_lemmas, embL_sz)
         self.dropL = nn.Dropout(emb_dropout)
         lstm_in_size += embL_sz

      # [MOD-1.2] optional PoS embedding
      if self.use_pos:
         self.embPOS  = nn.Embedding(n_pos, embPOS_sz)
         self.dropPOS = nn.Dropout(emb_dropout)
         lstm_in_size += embPOS_sz

      # ---- BiLSTM --------------------------------------------------------
      # [MOD-1.2] num_layers and inter-layer dropout now configurable
      self.lstm = nn.LSTM(lstm_in_size, lstm_hidden,
                          num_layers=lstm_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=lstm_dropout if lstm_layers > 1 else 0.0)

      # [MOD-1.2] optional LayerNorm on LSTM outputs
      self.use_layernorm = use_layernorm
      if use_layernorm:
          self.layernorm = nn.LayerNorm(2 * lstm_hidden)

      # ---- classification head ------------------------------------------
      self.linear = nn.Linear(2 * lstm_hidden, fc_hidden)
      self.out    = nn.Linear(fc_hidden, n_labels)

      # [MOD-1.2] selectable activation
      if   activation == 'relu': self.act = func.relu
      elif activation == 'tanh': self.act = torch.tanh
      elif activation == 'gelu': self.act = func.gelu
      else:                      self.act = func.relu

   # [MOD-1.2] 7-channel signature: (lw, w, s, pref, lemma, pos, f)
   # Channels beyond the original 4 are ignored unless their use_* flag is set.
   def forward(self, lw, w, s, pref, lemma, pos, f):
      x = self.embLW(lw)
      y = self.embW(w)
      z = self.embS(s)
      x = self.dropLW(x)
      y = self.dropW(y)
      z = self.dropS(z)

      parts = [x, y, z, f]

      if self.use_pref:
         p = self.dropP(self.embP(pref))
         parts.append(p)
      if self.use_lemma:
         l = self.dropL(self.embL(lemma))
         parts.append(l)
      if self.use_pos:
         q = self.dropPOS(self.embPOS(pos))
         parts.append(q)

      x = torch.cat(parts, dim=2)
      x = self.lstm(x)[0]

      if self.use_layernorm:
          x = self.layernorm(x)

      x = self.act(x)             # [MOD-1.2] configurable activation

      x = self.linear(x)
      x = self.out(x)
      return x
