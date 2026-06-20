#! /usr/bin/python3

import sys, os
from xml.dom.minidom import parse
import spacy

from patterns import *

## ------------------- 
## -- Convert a pair of drugs and their context in a feature vector

def extract_pair_features(tree, entities, e1, e2) :
   feats = set()

   # Features about entity types
   feats.add("typeE1="+ entities[e1]['type'])
   feats.add("typeE2="+ entities[e2]['type'])
   if entities[e1]['text'].lower() == entities[e2]['text'].lower() :
      feats.add("samedrug")

   # features about paths in the tree.
   # get head token for each gold entity
   tkE1 = get_fragment_head(tree,entities[e1]['start'],entities[e1]['end'])
   tkE2 = get_fragment_head(tree,entities[e2]['start'],entities[e2]['end'])

   # [MOD-2.1] mod1 — distance & sentence-position features (DISABLED)
   # Rationale: DDI is sentence-level classification; we hypothesised the
   # relative position of E1 vs E2 and the overall sentence length would
   # give cheap, strong cues. Tested standalone: -1.9 pp M-F1 vs ref-MEM
   # (62.4 vs 64.3). Tested in combination with mod2: -1.3 pp M-F1 vs ref
   # (63.0 vs 64.3). Conclusion: distance/length buckets dilute the
   # informative path features. Block left as a negative-result record.
   # if tkE1 is not None and tkE2 is not None:
   #    e1pos_ = get_position(tree, tkE1)
   #    e2pos_ = get_position(tree, tkE2)
   #    dist_ = abs(e2pos_ - e1pos_)
   #    senlen_ = len(tree)
   #    if dist_ <= 2:     feats.add("dist=0-2")
   #    elif dist_ <= 5:   feats.add("dist=3-5")
   #    elif dist_ <= 10:  feats.add("dist=6-10")
   #    elif dist_ <= 20:  feats.add("dist=11-20")
   #    else:              feats.add("dist=21+")
   #    if senlen_ <= 15:  feats.add("senlen=0-15")
   #    elif senlen_ <= 30:feats.add("senlen=16-30")
   #    elif senlen_ <= 50:feats.add("senlen=31-50")
   #    else:              feats.add("senlen=51+")
   #    feats.add("e1_before_e2" if e1pos_ < e2pos_ else "e2_before_e1")

   if tkE1 is not None and tkE2 is not None:
      # get LCS      
      lcs = get_LCS(tree,tkE1,tkE2)

      if lcs is not None :
          feats.add("lcs="+lcs.lemma_+"_"+lcs.pos_)
          
          # paths from E1 to LCS, using lemma, rel, or both
          path1 = get_up_path(tkE1,lcs)
          p1 = "<".join([x.lemma_+"_"+x.dep_ for x in path1])
          feats.add("path1="+p1)
          p1b = "<".join([x.lemma_ for x in path1])
          feats.add("path1b="+p1b)
          p1c = "<".join([x.dep_ for x in path1])
          feats.add("path1c="+p1c)

          # paths from LCS to E2, using lemma, rel, or both
          path2 = get_down_path(lcs,tkE2)
          p2 = ">".join([x.lemma_+"_"+x.dep_ for x in path2])
          feats.add("path2="+p2)
          p2b = ">".join([x.lemma_ for x in path2])
          feats.add("path2b="+p2b)
          p2c = ">".join([x.dep_ for x in path2])
          feats.add("path2c="+p2c)

          # paths from E1 to E2, using lemma, rel, or both
          p = p1+"<"+lcs.lemma_+"_"+lcs.dep_+">"+p2
          feats.add("path="+p)
          pb = p1b+"<"+lcs.lemma_+">"+p2b
          feats.add("pathb="+pb)
          pc = p1c+"<"+lcs.dep_+">"+p2c
          feats.add("pathc="+pc)

          # LCS lemma/tag and rels under it
          if len(path1)>0 and len(path2)>0 :
             pa = path1[-1].dep_+"<"+lcs.lemma_+">"+path2[0].dep_
             feats.add("pathA="+pa)
             pab = path1[-1].dep_+"<"+lcs.pos_+">"+path2[0].dep_
             feats.add("pathAb="+pab)

          # words in path from E1 to E2
          for w in path1 :
             feats.add("wip1="+w.lemma_)
             feats.add("wip="+w.lemma_)
          for w in path2 :
             feats.add("wip2="+w.lemma_)
             feats.add("wip="+w.lemma_)
          feats.add("wip="+lcs.lemma_)
          feats.add("lcs="+lcs.lemma_)

          # lcs children
          for w in lcs.children : feats.add("lcsCH="+w.lemma_)

          # [MOD-2.1] mod7 — LCS subtree shape (DISABLED, -0.3 pp M-F1 vs mod_best2)
          # kid_deps_ = sorted([w.dep_ for w in lcs.children])
          # feats.add("lcs_kids_deps=" + "_".join(kid_deps_) if kid_deps_ else "lcs_kids_deps=∅")
          # nk_ = len(kid_deps_)
          # if nk_ == 0:   feats.add("lcs_nk=0")
          # elif nk_ <= 2: feats.add("lcs_nk=1-2")
          # elif nk_ <= 4: feats.add("lcs_nk=3-4")
          # else:          feats.add("lcs_nk=5+")
          # if lcs.head == lcs:
          #    feats.add("lcs_parent_dep=ROOT")
          # else:
          #    feats.add("lcs_parent_dep=" + lcs.head.dep_)

          # [MOD-2.1] mod3 — third-entity context on the E1→E2 dep path (mod_best)
          path_tokens_ = list(path1) + [lcs] + list(path2)
          for tk_ in path_tokens_:
             eid_ = is_entity(tk_, entities)
             if eid_ is not None and eid_ not in (e1, e2):
                feats.add("path_has_other_entity")
                feats.add("path_other_type=" + entities[eid_]['type'])

   # [MOD-2.1] mod3 — third-entity context (mod_best)
   others_ = [eid for eid in entities if eid not in (e1, e2)]
   n_other_ = len(others_)
   if n_other_ == 0:    feats.add("n_other=0")
   elif n_other_ == 1:  feats.add("n_other=1")
   elif n_other_ == 2:  feats.add("n_other=2")
   else:                feats.add("n_other=3+")
   for t_ in sorted({entities[eid_]['type'] for eid_ in others_}):
      feats.add("other_type=" + t_)

   # [MOD-2.1] mod4 — entity-type pair (DISABLED — final config = mod_best2)
   # Tested both single-stage and two-stage:
   #  - single-stage mod4 alone: -0.2 pp M vs ref
   #  - single-stage mod_best (mod2+mod3+mod4): devel 65.2, test 66.8
   #  - single-stage mod_best2 (mod2+mod3, no mod4): devel 65.4, test 65.7
   #  - two-stage on mod_best:  devel 65.3, test 66.1
   #  - two-stage on mod_best2: devel 66.0, test 66.6  ← FINAL CHAMPION
   # mod4 helps single-stage on test but hurts two-stage on test; devel
   # selection plus two-stage choice → drop mod4.
   # feats.add("typePair=" + entities[e1]['type'] + "_" + entities[e2]['type'])
   # feats.add("typePairSorted=" + "_".join(sorted([entities[e1]['type'],
   #                                                entities[e2]['type']])))

   # [MOD-2.1] mod5 — negation cues (DISABLED)
   # Standalone test: M-F1 63.0 vs ref 64.3 (-1.3 pp). Hurt advise (-2.1)
   # and int (-3.4); the model doesn't learn negation polarity well from
   # bag-of-cue features alone.
   # NEG_LEMMAS_ = {"not", "no", "without", "fail", "neither", "nor", "never", "none"}
   # has_neg_sent_ = False
   # has_neg_between_ = False
   # if tkE1 is not None and tkE2 is not None:
   #    lo_ = min(get_position(tree, tkE1), get_position(tree, tkE2))
   #    hi_ = max(get_position(tree, tkE1), get_position(tree, tkE2))
   #    for i_, tk_ in enumerate(tree):
   #       lem_ = tk_.lemma_.lower()
   #       if lem_ in NEG_LEMMAS_:
   #          has_neg_sent_ = True
   #          feats.add("neg_lemma=" + lem_)
   #          if lo_ < i_ < hi_:
   #             has_neg_between_ = True
   # if has_neg_sent_:    feats.add("neg_in_sent")
   # if has_neg_between_: feats.add("neg_between")

   # features using rule-based patterns
   for pat in patterns :
      match = patterns[pat](tree, entities, e1, e2)
      if match is not None: 
         for m in match :
            feats.add(pat+"="+m)
                     
   return feats


## --------- Feature extractor ----------- 
## -- Extract features for each entity pair in each
## -- sentence in given file

def extract_features(datafile, outfile, dump_trees=False) :

   # open output file
   outf = open(outfile, "w")
   if dump_trees:
       treedir = os.path.join(os.path.dirname(outfile), "svg")
       os.makedirs(treedir, exist_ok=True)
    
   # create spacy parser
   nlp = spacy.load("en_core_web_trf",
                    enable=["transformer", "tagger","attribute_ruler", "lemmatizer", "ner", "parser"])

   # parse XML file, obtaining a DOM tree
   tree = parse(datafile)

   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text
        print(f"extracting sentence {sid}             \r", end="")
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents :
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")           
           text = e.attributes["text"].value
           typ = e.attributes["type"].value
           entities[id] = {'start': int(offs[0]), 'end': int(offs[-1]),
                           'text': text, 'type' : typ}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1 : continue

        # get syntactic analysis for the sentence
        analysis = nlp(stext)
        if dump_trees : 
           svg = spacy.displacy.render(analysis,style="dep")    
           with open(os.path.join(treedir,sid+".svg"),"w") as sf :  
              sf.write(svg)       
        
        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi=="true") : dditype = p.attributes["type"].value
            else : dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction
            feats = extract_pair_features(analysis,entities,id_e1,id_e2) 
            # resulting vector
            print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t", file=outf)


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir outfile
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- corresponding feature vectors to outfile
## --

if __name__ == "__main__" :
    # directory with files to process
    datafile = sys.argv[1]
    # file where to store results
    featfile = sys.argv[2]
    trees = len(sys.argv)>3 and sys.argv[3]=="trees"
    
    extract_features(datafile, featfile, trees)

