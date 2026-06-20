import os
import string
import re
import torch

from dataset import *

class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, params) :
        # [MOD-2.2] keep params so __create_indexs can read suf_len / pref_len etc.
        self.params = params if params is not None else {}
        maxlen = params['max_len'] if 'max_len' in params else None
        suflen = int(params['suf_len']) if 'suf_len' in params else 0
        preflen = int(params['pref_len']) if 'pref_len' in params else 0
        # [MOD-2.2] etype / form must be explicitly enabled (default off) so
        # mod1/mod2/... isolate cleanly during ablation.
        self.use_etype_flag = bool(int(params.get('use_etype', 0)))
        self.params['use_form'] = bool(int(params.get('use_form', 0)))
        # [MOD-2.2] relative-position embedding: distance from each token
        # to <DRUG1> / <DRUG2> markers, bucketed. Classic relation-extraction
        # trick — the model otherwise has no explicit notion of "near the
        # target pair vs. far away in the sentence".
        self.params['use_relpos'] = bool(int(params.get('use_relpos', 0)))

        if isinstance(data,Dataset) and maxlen is not None:
            self.__create_indexs(data, maxlen, suflen, preflen)

        elif type(data) == str :
            print('Codemaps: ', end='')
            if maxlen is not None :
                print('Ignoring given params and ', end='')
            print(f'loading index from {data}.idx')
            self.__load(data)

        else:
            print(f'codemaps: Missing max_len and/or suf_len parameters in constructor. params={params}')
            exit()


    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen=0, preflen=0) :

        self.maxlen = maxlen
        # [MOD-2.2] suffix / prefix lengths (0 disables that input channel)
        self.suflen = suflen
        self.preflen = preflen
        words = set([])
        lc_words = set([])
        lems = set([])
        pos = set([])
        sufs = set([])    # [MOD-2.2] mod1 — suffix index
        prefs = set([])   # [MOD-2.2] mod2 — prefix index
        etypes = set([])  # [MOD-2.2] mod3 — entity-type indicator (drug/group/brand/drug_n/none)
        labels = set([])

        for s in data.sentences() :
            for t in s['sent'] :
                words.add(t['form'])
                lc_words.add(t['lc_form'])
                lems.add(t['lemma'])
                pos.add(t['pos'])
                if suflen > 0:
                    sufs.add(t['form'].lower()[-suflen:] if len(t['form']) >= suflen else t['form'].lower())
                if preflen > 0:
                    prefs.add(t['form'].lower()[:preflen] if len(t['form']) >= preflen else t['form'].lower())
                # entity-type marker: tokens that are <DRUG1>/<DRUG2>/<DRUG_OTHER> carry an 'etype' field
                etypes.add(t.get('etype', 'O'))
            labels.add(s['type'])

        self.word_index = {w: i+2 for i,w in enumerate(list(words))}
        self.word_index['PAD'] = 0 # Padding
        self.word_index['UNK'] = 1 # Unknown words

        self.lc_word_index = {w: i+2 for i,w in enumerate(list(lc_words))}
        self.lc_word_index['PAD'] = 0 # Padding
        self.lc_word_index['UNK'] = 1 # Unknown words

        self.lemma_index = {s: i+2 for i,s in enumerate(list(lems))}
        self.lemma_index['PAD'] = 0  # Padding
        self.lemma_index['UNK'] = 1  # Unseen lemmas

        self.pos_index = {s: i+2 for i,s in enumerate(list(pos))}
        self.pos_index['PAD'] = 0  # Padding
        self.pos_index['UNK'] = 1  # Unseen PoS tags

        # [MOD-2.2] mod1 — suffix index
        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0
        self.suf_index['UNK'] = 1

        # [MOD-2.2] mod2 — prefix index
        self.pref_index = {s: i+2 for i,s in enumerate(list(prefs))}
        self.pref_index['PAD'] = 0
        self.pref_index['UNK'] = 1

        # [MOD-2.2] mod3 — etype index
        self.etype_index = {s: i+1 for i,s in enumerate(list(etypes))}
        self.etype_index['PAD'] = 0

        # [MOD-2.2] relative-position bucket index (11 buckets + PAD)
        # Buckets used for both DRUG1 and DRUG2 distances.
        self.relpos_buckets = ["<=-10","-9..-5","-4..-2","-1","0","+1",
                                "+2..+4","+5..+9",">=+10","UNK"]
        self.relpos_index = {b: i+1 for i,b in enumerate(self.relpos_buckets)}
        self.relpos_index['PAD'] = 0

        self.label_index = {t: i for i,t in enumerate(list(labels))}

        
    ## --------- load indexs -----------
    def __load(self, name) :
        self.maxlen = 0
        self.suflen = 0
        self.preflen = 0
        self.word_index = {}
        self.lc_word_index = {}
        self.lemma_index = {}
        self.pos_index = {}
        # [MOD-2.2] new index slots
        self.suf_index = {}
        self.pref_index = {}
        self.etype_index = {}
        # [MOD-2.2] relative-position bucket index must be present even on
        # __load() path so encode_words can build the rel-pos tensors.
        self.relpos_buckets = ["<=-10","-9..-5","-4..-2","-1","0","+1",
                                "+2..+4","+5..+9",">=+10","UNK"]
        self.relpos_index = {b: i+1 for i,b in enumerate(self.relpos_buckets)}
        self.relpos_index['PAD'] = 0
        self.label_index = {}

        with open(name+".idx") as f :
            for line in f.readlines():
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)         # [MOD-2.2]
                elif t == 'PREFLEN' : self.preflen = int(k)       # [MOD-2.2]
                # [MOD-2.2] restore use_etype / use_form flags from saved idx
                # so the predict path produces the same set of input tensors
                # the trained network expects.
                elif t == 'USE_ETYPE': self.params['use_etype'] = int(k)
                elif t == 'USE_FORM': self.params['use_form'] = int(k)
                elif t == 'USE_RELPOS': self.params['use_relpos'] = int(k)
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'LCWORD': self.lc_word_index[k] = int(i)
                elif t == 'LEMMA': self.lemma_index[k] = int(i)
                elif t == 'POS': self.pos_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)        # [MOD-2.2]
                elif t == 'PREF': self.pref_index[k] = int(i)      # [MOD-2.2]
                elif t == 'ETYPE': self.etype_index[k] = int(i)    # [MOD-2.2]
                elif t == 'LABEL': self.label_index[k] = int(i)


    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w") as f :
            print ('MAXLEN', self.maxlen, "-", file=f)
            print ('SUFLEN', self.suflen, "-", file=f)              # [MOD-2.2]
            print ('PREFLEN', self.preflen, "-", file=f)            # [MOD-2.2]
            # [MOD-2.2] persist the optional-channel flags so predict mirrors train
            print ('USE_ETYPE', int(bool(self.params.get('use_etype', 0))), "-", file=f)
            print ('USE_FORM',  int(bool(self.params.get('use_form', 0))), "-", file=f)
            print ('USE_RELPOS', int(bool(self.params.get('use_relpos', 0))), "-", file=f)

            for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
            for key in self.lc_word_index : print('LCWORD', key, self.lc_word_index[key], file=f)
            for key in self.lemma_index : print('LEMMA', key, self.lemma_index[key], file=f)
            for key in self.pos_index : print('POS', key, self.pos_index[key], file=f)
            for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)        # [MOD-2.2]
            for key in self.pref_index : print('PREF', key, self.pref_index[key], file=f)    # [MOD-2.2]
            for key in self.etype_index : print('ETYPE', key, self.etype_index[key], file=f) # [MOD-2.2]
            
            
     ## --------- get code for key k in given index, or code for unknown if not found
    def __code(self, index, k) :
        return index[k] if k in index else index['UNK']

    ## --------- encode and pad all sequences of given key (form, lemma, etc) -----------
    def __encode_and_pad(self, data, index, key) :
        enc = [torch.Tensor([self.__code(index,w[key]) for w in s['sent']]) for s in data.sentences()]
        # cut sentences longer than maxlen
        enc = [s[0:self.maxlen] for s in enc]
        # create a tensor full of padding
        X = torch.Tensor([]).new_full((len(enc), self.maxlen), index['PAD'], dtype=torch.int64)
        # fill padding tensor with sentence data
        for i, s in enumerate(enc): X[i, 0:s.size()[0]] = s
        return X

    # [MOD-2.2] generic encode-and-pad for token-derived keys (suffix/prefix
    # of form, etype-of-token) so we can build per-token derived features
    # without changing the dataset pickle format.
    def __encode_and_pad_derived(self, data, index, deriver, padval=None) :
        if padval is None: padval = index.get('PAD', 0)
        enc = [torch.Tensor([self.__code(index, deriver(w)) for w in s['sent']]) for s in data.sentences()]
        enc = [s[0:self.maxlen] for s in enc]
        X = torch.Tensor([]).new_full((len(enc), self.maxlen), padval, dtype=torch.int64)
        for i, s in enumerate(enc): X[i, 0:s.size()[0]] = s
        return X


    ## --------- encode X from given data -----------
    def encode_words(self, data) :

        # encode and pad sentence words
        Xw = self.__encode_and_pad(data, self.word_index, 'form')
        # encode and pad sentence lc_words
        Xlw = self.__encode_and_pad(data, self.lc_word_index, 'lc_form')
        # encode and pad lemmas
        Xl = self.__encode_and_pad(data, self.lemma_index, 'lemma')
        # encode and pad PoS
        Xp = self.__encode_and_pad(data, self.pos_index, 'pos')

        # [MOD-2.2] build the input list according to which extra channels are
        # enabled. We always include [Xlw, Xl, Xp] (the shipped baseline),
        # then optionally Xs (suffix), Xpr (prefix), Xe (etype indicator),
        # and Xform (case-sensitive form).
        inputs = [Xlw, Xl, Xp]
        if self.suflen > 0:
            sl = self.suflen
            sfun = lambda w: w['form'].lower()[-sl:] if len(w['form']) >= sl else w['form'].lower()
            inputs.append(self.__encode_and_pad_derived(data, self.suf_index, sfun))
        if self.preflen > 0:
            pl = self.preflen
            pfun = lambda w: w['form'].lower()[:pl] if len(w['form']) >= pl else w['form'].lower()
            inputs.append(self.__encode_and_pad_derived(data, self.pref_index, pfun))
        if self.use_etype():
            inputs.append(self.__encode_and_pad_derived(data, self.etype_index,
                                                       lambda w: w.get('etype', 'O')))
        if self.params.get('use_form', False):
            inputs.append(Xw)
        # [MOD-2.2] relative position embeddings (distance to DRUG1/DRUG2)
        if self.use_relpos():
            Xd1, Xd2 = self.__encode_relpos(data)
            inputs.append(Xd1)
            inputs.append(Xd2)
        return inputs

    # [MOD-2.2] relative-position helpers
    def __relpos_bucket(self, d) :
        if d <= -10: return "<=-10"
        if d <= -5:  return "-9..-5"
        if d <= -2:  return "-4..-2"
        if d == -1:  return "-1"
        if d == 0:   return "0"
        if d == 1:   return "+1"
        if d <= 4:   return "+2..+4"
        if d <= 9:   return "+5..+9"
        return ">=+10"

    def __encode_relpos(self, data) :
        idx = self.relpos_index
        pad = idx['PAD']
        Xd1 = torch.Tensor([]).new_full((len(data.data), self.maxlen), pad, dtype=torch.int64)
        Xd2 = torch.Tensor([]).new_full((len(data.data), self.maxlen), pad, dtype=torch.int64)
        for si, s in enumerate(data.sentences()):
            toks = s['sent']
            # find DRUG1 / DRUG2 positions
            p1 = next((i for i,t in enumerate(toks) if t['form'] == '<DRUG1>'), None)
            p2 = next((i for i,t in enumerate(toks) if t['form'] == '<DRUG2>'), None)
            for i, t in enumerate(toks):
                if i >= self.maxlen:
                    break
                d1 = (i - p1) if p1 is not None else 0
                d2 = (i - p2) if p2 is not None else 0
                Xd1[si, i] = idx[self.__relpos_bucket(d1)]
                Xd2[si, i] = idx[self.__relpos_bucket(d2)]
        return Xd1, Xd2

    
    ## --------- encode Y from given data ----------- 
    def encode_labels(self, data) :
        # encode and pad sentence labels 
        labels = [[1 if i==self.label_index[s['type']] else 0 for i in range(len(self.label_index))] for s in data.sentences()]
        Y = torch.Tensor(labels)     
        return Y

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)
    ## -------- get lc_word index size ---------
    def get_n_lc_words(self) :
        return len(self.lc_word_index)
    ## -------- get label index size ---------
    def get_n_lemmas(self) :
        return len(self.lemma_index)
    ## -------- get label index size ---------
    def get_n_pos(self) :
        return len(self.pos_index)
    # [MOD-2.2] sizes for new index slots
    def get_n_suffixes(self) :
        return len(self.suf_index)
    def get_n_prefixes(self) :
        return len(self.pref_index)
    def get_n_etypes(self) :
        return len(self.etype_index)
    def use_suffix(self) :
        return self.suflen > 0
    def use_prefix(self) :
        return self.preflen > 0
    def use_etype(self) :
        return bool(self.params.get('use_etype', 0)) and len(self.etype_index) > 1
    def use_form(self) :
        return bool(self.params.get('use_form', False))
    # [MOD-2.2]
    def use_relpos(self) :
        return bool(self.params.get('use_relpos', False))
    def get_n_relpos(self) :
        return len(self.relpos_index)
    ## -------- get label index size ---------
    def get_n_labels(self) :
        return len(self.label_index)
    ## -------- get label index size ---------
    def get_n_features(self) :
        return len(self.features(None))
    ## -------- get index for given word ---------
    def word2idx(self, w) :
        return self.word_index[w]
    ## -------- get index for given lc_word ---------
    def lcword2idx(self, w) :
        return self.lc_word_index[w]
    ## -------- get index for given lemma ---------
    def lemma2idx(self, w) :
        return self.lemma_index[w]
    ## -------- get index for given pos ---------
    def pos2idx(self, w) :
        return self.pos_index[w]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError

