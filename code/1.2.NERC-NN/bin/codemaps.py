import os
import string
import re
import torch

from dataset import *

# folder where this file is located
THISDIR=os.path.abspath(os.path.dirname(__file__))
# go two folders up and locate "resources" folder there
NERDIR=os.path.dirname(THISDIR)
SOLDIR=os.path.dirname(NERDIR)
MAINDIR=os.path.dirname(SOLDIR)
RESOURCESDIR=os.path.join(MAINDIR, "resources")

class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, params) :
        # [MOD-1.2] remember params so __create_indexs can read pref_len etc.
        self.params = params if params is not None else {}
        maxlen = params['max_len'] if 'max_len' in params else None
        suflen = params['suf_len'] if 'suf_len' in params else None
        
        #----------------------
        self.external = {}
        self.externalpart = {}
        with open(os.path.join(RESOURCESDIR,"HSDB.txt"),encoding='utf-8') as h :
            for x in h.readlines() :
                x = x.strip().lower()
                self.external[x] = {"any"}
                wds = x.split()
                if len(wds)>1 :
                   for w in wds:
                       self.externalpart[w] = {"any"}
                                
        with open(os.path.join(RESOURCESDIR,"DrugBank.txt"),encoding='utf-8') as h :
            for x in h.readlines() :
                (n,t) = x.strip().lower().split("|")
                if n in self.external : self.external[n].add(t)
                else: self.external[n] = {t}
                wds = n.split()
                if len(wds)>1 :
                   for w in wds:
                       if w in self.externalpart :
                          self.externalpart[w].add(t)
                       else :
                          self.externalpart[w] = {t}
                                
        #----------------------
                
        if isinstance(data,Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen)

        elif type(data) == str :
            print('Codemaps: ', end='')
            if maxlen is not None or suflen is not None :
                print('Ignoring given params and ', end='')
            print(f'loading index from {data}.idx')
            self.__load(data)

        else:
            print(f'codemaps: Missing max_len and/or suf_len parameters in constructor. params={params}')
            exit()

            
    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen) :

        self.maxlen = maxlen
        self.suflen = suflen
        # [MOD-1.2] prefix length shares the same knob family as suffix; default 3
        self.preflen = int(self.params.get('pref_len', 3)) if hasattr(self, 'params') else 3
        words = set([])
        lc_words = set([])
        sufs = set([])
        prefs = set([])     # [MOD-1.2] prefixes
        lemmas = set([])    # [MOD-1.2] lemmas
        pos_tags = set([])  # [MOD-1.2] spaCy coarse PoS
        labels = set([])

        for _,tokens,lab in data.sentences() :
            for i,t in enumerate(tokens) :
                if t.text.startswith(" "): continue
                words.add(t.text)
                lc_words.add(t.text.lower())
                sufs.add(t.text.lower()[-self.suflen:])
                prefs.add(t.text.lower()[:self.preflen])       # [MOD-1.2]
                lemmas.add(getattr(t, 'lemma_', t.text).lower())  # [MOD-1.2]
                pos_tags.add(getattr(t, 'pos_', 'X'))          # [MOD-1.2]
                labels.add(lab[i])

        self.word_index = {w: i+2 for i,w in enumerate(list(words))}
        self.word_index['PAD'] = 0 # Padding
        self.word_index['UNK'] = 1 # Unknown words

        self.lc_word_index = {w: i+2 for i,w in enumerate(list(lc_words))}
        self.lc_word_index['PAD'] = 0 # Padding
        self.lc_word_index['UNK'] = 1 # Unknown words

        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes

        # [MOD-1.2] new indexes: prefix, lemma, PoS
        self.pref_index = {s: i+2 for i,s in enumerate(list(prefs))}
        self.pref_index['PAD'] = 0
        self.pref_index['UNK'] = 1

        self.lemma_index = {w: i+2 for i,w in enumerate(list(lemmas))}
        self.lemma_index['PAD'] = 0
        self.lemma_index['UNK'] = 1

        self.pos_index = {p: i+2 for i,p in enumerate(list(pos_tags))}
        self.pos_index['PAD'] = 0
        self.pos_index['UNK'] = 1

        self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
        self.label_index['PAD'] = 0 # Padding
        
    ## --------- load indexs -----------
    def __load(self, name) :
        self.maxlen = 0
        self.suflen = 0
        self.preflen = 3              # [MOD-1.2]
        self.word_index = {}
        self.lc_word_index = {}
        self.suf_index = {}
        self.pref_index = {}          # [MOD-1.2]
        self.lemma_index = {}         # [MOD-1.2]
        self.pos_index = {}           # [MOD-1.2]
        self.label_index = {}

        with open(name+".idx") as f :
            for line in f.readlines():
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)
                elif t == 'PREFLEN' : self.preflen = int(k)            # [MOD-1.2]
                elif t == 'WORD': self.word_index[k] = int(i)
                elif t == 'LCWORD': self.lc_word_index[k] = int(i)
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'PREF': self.pref_index[k] = int(i)          # [MOD-1.2]
                elif t == 'LEMMA': self.lemma_index[k] = int(i)        # [MOD-1.2]
                elif t == 'POS': self.pos_index[k] = int(i)            # [MOD-1.2]
                elif t == 'LABEL': self.label_index[k] = int(i)


    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w") as f :
            print ('MAXLEN', self.maxlen, "-", file=f)
            print ('SUFLEN', self.suflen, "-", file=f)
            print ('PREFLEN', self.preflen, "-", file=f)               # [MOD-1.2]
            for key in self.label_index : print('LABEL', key, self.label_index[key], file=f)
            for key in self.word_index : print('WORD', key, self.word_index[key], file=f)
            for key in self.lc_word_index : print('LCWORD', key, self.lc_word_index[key], file=f)
            for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)
            # [MOD-1.2] persist new indexes
            for key in self.pref_index : print('PREF', key, self.pref_index[key], file=f)
            for key in self.lemma_index : print('LEMMA', key, self.lemma_index[key], file=f)
            for key in self.pos_index : print('POS', key, self.pos_index[key], file=f)


    ## --------- Pad tensors for short sentences and cut sentences longer 
    ## --------- than maxlen, so all sentences have the same length.
    ## --------- Return a tensor with all the sentences.
    ## --------- Given tensor_list is assumed to have one tensor per sentence.
    ## --------- Each sentence tensors has :
    ## ---------    1nd dimension = n_words in the sentence
    ## ---------    2nd dimension (if any) = n_feature bits for each word
    def cut_and_pad(self, tensor_list, pad) :
        # check if the tensors are 1d or 2d, and decide shape of output tensor 
        if len(tensor_list[0].shape)==1 : 
           shape = (len(tensor_list), self.maxlen)
        elif len(tensor_list[0].shape)==2 : 
           shape = (len(tensor_list), self.maxlen, tensor_list[0].shape[1])
        # cut sentences longer than maxlen
        tensor_list = [s[0:self.maxlen] for s in tensor_list]
        # create a tensor full of padding with the final desired shape
        padded = torch.Tensor([]).new_full(shape, pad, dtype=torch.int64)        
        # fill padded tensor with given data, leaving padding in unused spaces
        for i,s in enumerate(tensor_list):
           for j,f in enumerate(tensor_list[i]) :
              padded[i,j] = f
        return padded
    
    ## --------- encode X from given data -----------
    def encode_words(self, data) :

        #----- encode sentence words
        enc = [torch.Tensor([self.word_index[w.text] if w.text in self.word_index else self.word_index['UNK'] for w in s]) for _,s,_ in data.sentences()]
        Xw = self.cut_and_pad(enc, self.word_index['PAD'])

        #------ encode sentence lowercase words
        enc = [torch.Tensor([self.lc_word_index[w.text.lower()] if w.text.lower() in self.lc_word_index else self.lc_word_index['UNK'] for w in s]) for _,s,_ in data.sentences()]
        Xlw = self.cut_and_pad(enc, self.lc_word_index['PAD'])

        #------ encode sentence suffixes
        enc = [torch.Tensor([self.suf_index[w.text.lower()[-self.suflen:]] if w.text.lower()[-self.suflen:] in self.suf_index else self.suf_index['UNK'] for w in s]) for _,s,_ in data.sentences()]
        Xs = self.cut_and_pad(enc, self.suf_index['PAD'])

        # [MOD-1.2] encode sentence prefixes
        enc = [torch.Tensor([self.pref_index.get(w.text.lower()[:self.preflen], self.pref_index['UNK']) for w in s]) for _,s,_ in data.sentences()]
        Xp = self.cut_and_pad(enc, self.pref_index['PAD'])

        # [MOD-1.2] encode lemmas (falls back to the word if lemma_ is missing)
        enc = [torch.Tensor([self.lemma_index.get(getattr(w, 'lemma_', w.text).lower(), self.lemma_index['UNK']) for w in s]) for _,s,_ in data.sentences()]
        Xl = self.cut_and_pad(enc, self.lemma_index['PAD'])

        # [MOD-1.2] encode spaCy coarse PoS tags
        enc = [torch.Tensor([self.pos_index.get(getattr(w, 'pos_', 'X'), self.pos_index['UNK']) for w in s]) for _,s,_ in data.sentences()]
        Xpos = self.cut_and_pad(enc, self.pos_index['PAD'])

        #------ encode word features
        enc = [torch.Tensor([self.features(w) for w in s]) for _,s,_ in data.sentences()]
        Xf = self.cut_and_pad(enc, 0)

        # [MOD-1.2] return 7 channels. Order MUST match network.forward signature:
        # (lw, w, s, pref, lemma, pos, f)
        return [Xlw, Xw, Xs, Xp, Xl, Xpos, Xf]

    
    ## --------- encode Y from given data ----------- 
    def encode_labels(self, data) :
        # encode and pad sentence labels
        enc = [torch.Tensor([self.label_index[lab] for lab in l]) for _,_,l in data.sentences()]
        Y = self.cut_and_pad(enc, self.label_index['PAD'])
        return Y

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)
    ## -------- get lc_word index size ---------
    def get_n_lc_words(self) :
        return len(self.lc_word_index)
    ## -------- get suf index size ---------
    def get_n_sufs(self) :
        return len(self.suf_index)
    ## [MOD-1.2] -------- get pref index size ---------
    def get_n_prefs(self) :
        return len(self.pref_index)
    ## [MOD-1.2] -------- get lemma index size ---------
    def get_n_lemmas(self) :
        return len(self.lemma_index)
    ## [MOD-1.2] -------- get pos index size ---------
    def get_n_pos(self) :
        return len(self.pos_index)
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
    ## -------- get index for given suffix --------
    def suff2idx(self, s) :
        return self.suff_index[s]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError

    ## [MOD-1.2] -------- build pretrained embedding matrix ---------
    # Looks up every token of self.lc_word_index in a spaCy model that
    # ships word vectors (default: en_core_web_md, 300d). Returns a
    # float tensor of shape (n_lc_words, emb_dim). OOV tokens get a
    # small random normal vector. PAD gets zeros.
    def build_pretrained_matrix(self, model_name='en_core_web_md') :
        import spacy
        try:
            nlp = spacy.load(model_name)
        except OSError:
            print(f"[MOD-1.2] pretrained model {model_name} not found, "
                  "falling back to en_core_web_md")
            nlp = spacy.load('en_core_web_md')
        dim = nlp.vocab.vectors_length
        if dim == 0:
            raise RuntimeError(f"spaCy model {model_name} has no word vectors; "
                               "use en_core_web_md or en_core_web_lg")
        n = len(self.lc_word_index)
        mat = torch.randn(n, dim) * 0.1           # small random for OOV
        hit = 0
        for w, i in self.lc_word_index.items():
            if w == 'PAD':
                mat[i] = 0.0
                continue
            if w == 'UNK':
                continue
            lex = nlp.vocab[w]
            if lex.has_vector and lex.vector_norm > 0:
                mat[i] = torch.from_numpy(lex.vector.copy())
                hit += 1
        print(f"[MOD-1.2] pretrained vectors: {hit}/{n-2} lc-word matches "
              f"({100.0*hit/max(n-2,1):.1f}%) from {model_name} (dim={dim})")
        return mat, dim

    ## -------- create vector with binary features (used by encode_words)
    def features(self,w) :
        f = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        if w is not None :
            form = w.text
            if form.isupper(): f[0] = 1
            if form.istitle(): f[1] = 1
            if form.isdigit(): f[2] = 1
            if '-' in form:    f[3] = 1
            if re.search('[0-9]',form): f[4] = 1
            if any([c in string.punctuation for c in form]): f[5] = 1

            lcform = w.text.lower()
            if lcform in self.external :
                if 'drug' in self.external[lcform] : f[6] = 1
                if 'group' in self.external[lcform] : f[7] = 1
                if 'brand' in self.external[lcform] : f[8] = 1
                if 'drug_n' in self.external[lcform] : f[9] = 1
                if 'any' in self.external[lcform] : f[10] = 1
            if lcform in self.externalpart :
                if 'drug' in self.externalpart[lcform] : f[11] = 1
                if 'group' in self.externalpart[lcform] : f[12] = 1
                if 'brand' in self.externalpart[lcform] : f[13] = 1
                if 'drug_n' in self.externalpart[lcform] : f[14] = 1
                if 'any' in self.externalpart[lcform] : f[15] = 1
        
        return f





