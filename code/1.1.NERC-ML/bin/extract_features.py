#! /usr/bin/python3

import sys, os
import re
from xml.dom.minidom import parse
import spacy

import paths
from dictionaries import Dictionaries

## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML
def get_label(tks, tke, spans) :
    for (spanS,spanE,spanT) in spans :
        if tks==spanS and tke<=spanE+1 : return "B-"+spanT
        elif tks>spanS and tke<=spanE+1 : return "I-"+spanT
    return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_sentence_features(tokens, dicts) :
   # Ablation toggles (requested): keep ADDITION 2/3/4/5 and 6 enabled.
   # Current experiment: ENABLE ADDITION 7 (char n-grams) while keeping ADDITION 6 on.
   # Now also ENABLE ADDITION 8 (multi-token dictionary span matches).
   ENABLE_ADDITION_7 = True
   ENABLE_ADDITION_8 = False
   ENABLE_ADDITION_9 = True

   # -- ADDITION 1 --
   def word_shape(txt) :
      shape = []
      for ch in txt :
         if ch.isupper() : shape.append('A')
         elif ch.islower() : shape.append('a')
         elif ch.isdigit() : shape.append('d')
         else : shape.append(ch)
      return "".join(shape)
   # ----------------

   # -- ADDITION 6 --
   # Text normalization & biomedical regex/pattern flags.
   # These help generalize to unseen tokens (especially drug_n-like strings)
   # that include digits, punctuation, Greek letters, or chemical patterns.
   def norm_digits(txt) :
      return re.sub(r"[0-9]", "0", txt)

   def has_greek(txt) :
      # Common Greek letters in biomedical text (α β γ δ etc.)
      return re.search(r"[\u0370-\u03FF]", txt) is not None

   def is_roman_numeral(txt) :
      return re.fullmatch(r"[IVXLCDM]+", txt) is not None

   def is_chem_like(txt) :
      # Heuristic: token with mixture of letters+digits and some chemical punctuation.
      # Examples: 5-FU, H2O2, IL-2, TNF-α, N-acetylcysteine
      if re.search(r"[A-Za-z]", txt) is None :
         return False
      if re.search(r"[0-9]", txt) is None :
         return False
      return re.search(r"[-+/().,]", txt) is not None

   def has_stereo(txt) :
      # (R) / (S) stereochemistry marker
      return re.fullmatch(r"\([RS]\)", txt) is not None

   def is_dose_unit(txt) :
      # Keep small on purpose; add more if needed.
      return txt in {
         "mg", "g", "mcg", "ug", "µg", "kg",
         "ml", "l", "mm", "cm",
         "%", "mmhg"
      }
   # ----------------

   # -- ADDITION 7 --
   # Character n-grams (c3/c4/c5) for current token.
   # This is a strong feature family for biomedical NER.
   def limited_char_ngrams(txt, n, limit=8) :
      if len(txt) < n :
         return []
      grams = []
      for j in range(0, len(txt) - n + 1) :
         grams.append(txt[j:j+n])
      if len(grams) <= limit :
         return grams
      # keep a few from start + end to cap feature explosion
      keep = grams[:limit//2] + grams[-(limit - limit//2):]
      return keep
   # ----------------

   # -- ADDITION 8 --
   # Multi-token dictionary span matches.
   # Token-level dictionary hits miss multiword drug names (e.g., "acetyl salicylic acid").
   # We precompute span-level marks (Begin/Inside/End) for 2..5 token spans.
   span_marks = {i: [] for i in range(len(tokens))}
   if ENABLE_ADDITION_8 :
      max_span_len = 5
      for start in range(len(tokens)) :
         for span_len in range(2, max_span_len+1) :
            end = start + span_len - 1
            if end >= len(tokens) :
               break
            phrase = " ".join(tokens[k].text.lower() for k in range(start, end+1))
            found, val = dicts.find(phrase, 'external')
            if not found :
               continue
            # begin token
            span_marks[start].append(f"dictSpanBeginLen={span_len}")
            for c in val :
               span_marks[start].append(f"dictSpanBeginType={c}")
            # inside tokens
            for mid in range(start+1, end) :
               span_marks[mid].append(f"dictSpanInsideLen={span_len}")
               for c in val :
                  span_marks[mid].append(f"dictSpanInsideType={c}")
            # end token
            span_marks[end].append(f"dictSpanEndLen={span_len}")
            for c in val :
               span_marks[end].append(f"dictSpanEndType={c}")
   # ----------------

   # for each token, generate list of features and add it to the result
   sentenceFeatures = {}
   for i,tk in enumerate(tokens) :
      tokenFeatures = []
      t = tk.text

      tokenFeatures.append("form="+t)
      tokenFeatures.append("formlower="+t.lower())

      # -- ADDITION 1 --
      tokenFeatures.append("pref2="+t[:2])
      tokenFeatures.append("pref3="+t[:3])
      tokenFeatures.append("pref4="+t[:4])

      tokenFeatures.append("suf2="+t[-2:])
      # ----------------
      tokenFeatures.append("suf3="+t[-3:])
      tokenFeatures.append("suf4="+t[-4:])

      # -- ADDITION 9 --
      if ENABLE_ADDITION_9:
          if len(t) >= 5:
              tokenFeatures.append("suf5="+t[-5:])
              tokenFeatures.append("pref5="+t[:5])
          if len(t) >= 6:
              tokenFeatures.append("suf6="+t[-6:])
              tokenFeatures.append("pref6="+t[:6])
      # ----------------

      # -- ADDITION 1 --
      tokenFeatures.append("shape="+word_shape(t))
      # -- ADDITION 6 --
      t_norm = norm_digits(t.lower())
      tokenFeatures.append("normDigits="+t_norm)
      if has_greek(t) : tokenFeatures.append("hasGreek")
      if is_roman_numeral(t) : tokenFeatures.append("isRoman")
      if is_chem_like(t) : tokenFeatures.append("isChemLike")
      if has_stereo(t) : tokenFeatures.append("hasStereo")
      if is_dose_unit(t.lower()) : tokenFeatures.append("isDoseUnit")
      # ----------------

      tl = len(t)
      if tl <= 3 : tokenFeatures.append("len<=3")
      elif tl <= 6 : tokenFeatures.append("len<=6")
      elif tl <= 10 : tokenFeatures.append("len<=10")
      else : tokenFeatures.append("len>10")
      # ----------------
      if t.isupper() : tokenFeatures.append("isUpper")
      if t.istitle() : tokenFeatures.append("isTitle")
      if t.isdigit() : tokenFeatures.append("isDigit")
      if '-' in t : tokenFeatures.append("hasDash")
      if re.search('[0-9]',t) : tokenFeatures.append("hasDigit")

      # -- ADDITION 1 --
      if '(' in t or ')' in t : tokenFeatures.append("hasParen")
      if '/' in t : tokenFeatures.append("hasSlash")
      if '.' in t : tokenFeatures.append("hasDot")
      if '+' in t : tokenFeatures.append("hasPlus")
      if ',' in t : tokenFeatures.append("hasComma")
      # ----------------

      # -- ADDITION 2 ---
      tokenFeatures.append("pos=" + tk.tag_) # Detailed tag (e.g., NNP)
      tokenFeatures.append("unipos=" + tk.pos_) # Universal tag (e.g., PROPN)
      # -----------------
      # -- ADDITION 3 ---
      tokenFeatures.append("lemma=" + tk.lemma_)
      # -----------------

      found,val = dicts.find(t.lower(), 'external')
      if found:
         # -- ADDITION 1 --
         tokenFeatures.append("inDictFull")
         # ----------------
         for c in val : tokenFeatures.append("external="+c)
      found,val = dicts.find(t.lower(), 'externalpart')
      if found:
          tokenFeatures.append("inDictPart")
          for c in val : tokenFeatures.append("externalpart="+c)

      # -- ADDITION 8 --
      # Add span-level dictionary marks for this token position.
      if ENABLE_ADDITION_8 :
         for m in span_marks.get(i, []) :
            tokenFeatures.append(m)
      # -----------------

      if i>0 :
         tPrev_obj = tokens[i-1]
         tPrev = tPrev_obj.text
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("formlowerPrev="+tPrev.lower())
         # -- ADDITION 1 --
         tokenFeatures.append("pref2Prev="+tPrev[:2])
         tokenFeatures.append("pref3Prev="+tPrev[:3])
         tokenFeatures.append("pref4Prev="+tPrev[:4])
         # -----------------
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
         tokenFeatures.append("suf4Prev="+tPrev[-4:])
         # -- ADDITION 1 --
         tokenFeatures.append("shapePrev="+word_shape(tPrev))
         # -----------------
         if tPrev.isupper() : tokenFeatures.append("isUpperPrev")
         if tPrev.istitle() : tokenFeatures.append("isTitlePrev")
         if tPrev.isdigit() : tokenFeatures.append("isDigitPrev")
         if '-' in tPrev : tokenFeatures.append("hasDashPrev")
         if re.search('[0-9]',tPrev) : tokenFeatures.append("hasDigitPrev")
         # -- ADDITION 1 --
         if '(' in tPrev or ')' in tPrev : tokenFeatures.append("hasParenPrev")
         if '/' in tPrev : tokenFeatures.append("hasSlashPrev")
         if '.' in tPrev : tokenFeatures.append("hasDotPrev")
         if '+' in tPrev : tokenFeatures.append("hasPlusPrev")
         if ',' in tPrev : tokenFeatures.append("hasCommaPrev")
         # -----------------

         # -- ADDITION 4 ---
         # Add POS/lemma for context tokens too (helps boundary decisions).
         tokenFeatures.append("posPrev=" + tPrev_obj.tag_)
         tokenFeatures.append("uniposPrev=" + tPrev_obj.pos_)
         tokenFeatures.append("lemmaPrev=" + tPrev_obj.lemma_)
         # -----------------

         # -- ADDITION 6 --
         tPrev_norm = norm_digits(tPrev.lower())
         tokenFeatures.append("normDigitsPrev="+tPrev_norm)
         if has_greek(tPrev) : tokenFeatures.append("hasGreekPrev")
         if is_roman_numeral(tPrev) : tokenFeatures.append("isRomanPrev")
         if is_chem_like(tPrev) : tokenFeatures.append("isChemLikePrev")
         if has_stereo(tPrev) : tokenFeatures.append("hasStereoPrev")
         if is_dose_unit(tPrev.lower()) : tokenFeatures.append("isDoseUnitPrev")
         # -----------------

         found,val = dicts.find(tPrev.lower(), 'external')
         if found:
             # -- ADDITION 1 --
             tokenFeatures.append("inDictFullPrev")
             # -----------------
             for c in val : tokenFeatures.append("externalPrev="+c)
         found,val = dicts.find(tPrev.lower(), 'externalpart')
         if found:
             # -- ADDITION 1 --
             tokenFeatures.append("inDictPartPrev")
             # -----------------
             for c in val : tokenFeatures.append("externalpartPrev="+c)
      else :
         tokenFeatures.append("BoS")

      # -- ADDITION 5: Window -2 ---
      if i>1 :
         tPrev2_obj = tokens[i-2]
         tPrev2 = tPrev2_obj.text
         tokenFeatures.append("formPrev2="+tPrev2)
         tokenFeatures.append("formlowerPrev2="+tPrev2.lower())
         # -- ADDITION 1 --
         tokenFeatures.append("pref2Prev2="+tPrev2[:2])
         tokenFeatures.append("pref3Prev2="+tPrev2[:3])
         tokenFeatures.append("pref4Prev2="+tPrev2[:4])
         # -----------------
         tokenFeatures.append("suf3Prev2="+tPrev2[-3:])
         tokenFeatures.append("suf4Prev2="+tPrev2[-4:])
         # -- ADDITION 1 --
         tokenFeatures.append("shapePrev2="+word_shape(tPrev2))
         # -----------------
         if tPrev2.isupper() : tokenFeatures.append("isUpperPrev2")
         if tPrev2.istitle() : tokenFeatures.append("isTitlePrev2")
         if tPrev2.isdigit() : tokenFeatures.append("isDigitPrev2")
         if '-' in tPrev2 : tokenFeatures.append("hasDashPrev2")
         if re.search('[0-9]',tPrev2) : tokenFeatures.append("hasDigitPrev2")
         # -- ADDITION 1 --
         if '(' in tPrev2 or ')' in tPrev2 : tokenFeatures.append("hasParenPrev2")
         if '/' in tPrev2 : tokenFeatures.append("hasSlashPrev2")
         if '.' in tPrev2 : tokenFeatures.append("hasDotPrev2")
         if '+' in tPrev2 : tokenFeatures.append("hasPlusPrev2")
         if ',' in tPrev2 : tokenFeatures.append("hasCommaPrev2")
         # -----------------

         # -- ADDITION 4 ---
         # Add POS/lemma for context tokens too (window -2).
         tokenFeatures.append("posPrev2=" + tPrev2_obj.tag_)
         tokenFeatures.append("uniposPrev2=" + tPrev2_obj.pos_)
         tokenFeatures.append("lemmaPrev2=" + tPrev2_obj.lemma_)
         # -----------------

         # -- ADDITION 6 --
         tPrev2_norm = norm_digits(tPrev2.lower())
         tokenFeatures.append("normDigitsPrev2="+tPrev2_norm)
         if has_greek(tPrev2) : tokenFeatures.append("hasGreekPrev2")
         if is_roman_numeral(tPrev2) : tokenFeatures.append("isRomanPrev2")
         if is_chem_like(tPrev2) : tokenFeatures.append("isChemLikePrev2")
         if has_stereo(tPrev2) : tokenFeatures.append("hasStereoPrev2")
         if is_dose_unit(tPrev2.lower()) : tokenFeatures.append("isDoseUnitPrev2")
         # -----------------

         found,val = dicts.find(tPrev2.lower(), 'external')
         if found:
             # -- ADDITION 1 --
             tokenFeatures.append("inDictFullPrev2")
             # -----------------
             for c in val : tokenFeatures.append("externalPrev2="+c)
         found,val = dicts.find(tPrev2.lower(), 'externalpart')
         if found:
             # -- ADDITION 1 --
             tokenFeatures.append("inDictPartPrev2")
             # -----------------
             for c in val : tokenFeatures.append("externalpartPrev2="+c)
      else :
         tokenFeatures.append("BoS2")

      if i<len(tokens)-1 :
         tNext_obj = tokens[i+1]
         tNext = tNext_obj.text
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("formlowerNext="+tNext.lower())
         # -- ADDITION 1 --
         tokenFeatures.append("pref2Next="+tNext[:2])
         tokenFeatures.append("pref3Next="+tNext[:3])
         tokenFeatures.append("pref4Next="+tNext[:4])
         # -----------------
         tokenFeatures.append("suf3Next="+tNext[-3:])
         tokenFeatures.append("suf4Next="+tNext[-4:])
         tokenFeatures.append("shapeNext="+word_shape(tNext))
         if tNext.isupper() : tokenFeatures.append("isUpperNext")
         if tNext.istitle() : tokenFeatures.append("isTitleNext")
         if tNext.isdigit() : tokenFeatures.append("isDigitNext")
         if '-' in tNext : tokenFeatures.append("hasDashNext")
         if re.search('[0-9]',tNext) : tokenFeatures.append("hasDigitNext")
         # -- ADDITION 1 --
         if '(' in tNext or ')' in tNext : tokenFeatures.append("hasParenNext")
         if '/' in tNext : tokenFeatures.append("hasSlashNext")
         if '.' in tNext : tokenFeatures.append("hasDotNext")
         if '+' in tNext : tokenFeatures.append("hasPlusNext")
         if ',' in tNext : tokenFeatures.append("hasCommaNext")
         # -----------------

         # -- ADDITION 4 --
         # Add POS/lemma for context tokens too.
         tokenFeatures.append("posNext=" + tNext_obj.tag_)
         tokenFeatures.append("uniposNext=" + tNext_obj.pos_)
         tokenFeatures.append("lemmaNext=" + tNext_obj.lemma_)
         # ----------------

         # -- ADDITION 6 --
         tNext_norm = norm_digits(tNext.lower())
         tokenFeatures.append("normDigitsNext="+tNext_norm)
         if has_greek(tNext) : tokenFeatures.append("hasGreekNext")
         if is_roman_numeral(tNext) : tokenFeatures.append("isRomanNext")
         if is_chem_like(tNext) : tokenFeatures.append("isChemLikeNext")
         if has_stereo(tNext) : tokenFeatures.append("hasStereoNext")
         if is_dose_unit(tNext.lower()) : tokenFeatures.append("isDoseUnitNext")
         # -----------------

         found,val = dicts.find(tNext.lower(), 'external')
         if found:
            # -- ADDITION 1 --
            tokenFeatures.append("inDictFullNext")
            # -----------------
            for c in val : tokenFeatures.append("externalNext="+c)
         found,val = dicts.find(tNext.lower(), 'externalpart')
         if found:
            # -- ADDITION 1 --
            tokenFeatures.append("inDictPartNext")
            # -----------------
            for c in val : tokenFeatures.append("externalpartNext="+c)
      else:
         tokenFeatures.append("EoS")

   
      # -- ADDITION 5: Window +2 ---
      if i < len(tokens) - 2 :
            tNext2_obj = tokens[i+2]
            tNext2 = tNext2_obj.text
            tokenFeatures.append("formNext2="+tNext2)
            tokenFeatures.append("formlowerNext2="+tNext2.lower())
            # -- ADDITION 1 --
            tokenFeatures.append("pref2Next2="+tNext2[:2])
            tokenFeatures.append("pref3Next2="+tNext2[:3])
            tokenFeatures.append("pref4Next2="+tNext2[:4])
            # -----------------
            tokenFeatures.append("suf3Next2="+tNext2[-3:])
            tokenFeatures.append("suf4Next2="+tNext2[-4:])
            tokenFeatures.append("shapeNext2="+word_shape(tNext2))
            if tNext2.isupper() : tokenFeatures.append("isUpperNext2")
            if tNext2.istitle() : tokenFeatures.append("isTitleNext2")
            if tNext2.isdigit() : tokenFeatures.append("isDigitNext2")
            if '-' in tNext2 : tokenFeatures.append("hasDashNext2")
            if re.search('[0-9]',tNext2) : tokenFeatures.append("hasDigitNext2")
            # -- ADDITION 1 --
            if '(' in tNext2 or ')' in tNext2 : tokenFeatures.append("hasParenNext2")
            if '/' in tNext2 : tokenFeatures.append("hasSlashNext2")
            if '.' in tNext2 : tokenFeatures.append("hasDotNext2")
            if '+' in tNext2 : tokenFeatures.append("hasPlusNext2")
            if ',' in tNext2 : tokenFeatures.append("hasCommaNext2")
            # -----------------

            # -- ADDITION 4 --
            # Add POS/lemma for context tokens too (window +2).
            tokenFeatures.append("posNext2=" + tNext2_obj.tag_)
            tokenFeatures.append("uniposNext2=" + tNext2_obj.pos_)
            tokenFeatures.append("lemmaNext2=" + tNext2_obj.lemma_)
            # ----------------

            # -- ADDITION 6 --
            tNext2_norm = norm_digits(tNext2.lower())
            tokenFeatures.append("normDigitsNext2="+tNext2_norm)
            if has_greek(tNext2) : tokenFeatures.append("hasGreekNext2")
            if is_roman_numeral(tNext2) : tokenFeatures.append("isRomanNext2")
            if is_chem_like(tNext2) : tokenFeatures.append("isChemLikeNext2")
            if has_stereo(tNext2) : tokenFeatures.append("hasStereoNext2")
            if is_dose_unit(tNext2.lower()) : tokenFeatures.append("isDoseUnitNext2")
            # -----------------

            found,val = dicts.find(tNext2.lower(), 'external')
            if found:
               # -- ADDITION 1 --
               tokenFeatures.append("inDictFullNext2")
               # -----------------
               for c in val : tokenFeatures.append("externalNext2="+c)
            found,val = dicts.find(tNext2.lower(), 'externalpart')
            if found:
               # -- ADDITION 1 --
               tokenFeatures.append("inDictPartNext2")
               # -----------------
               for c in val : tokenFeatures.append("externalpartNext2="+c)
      else:
            tokenFeatures.append("EoS2")


      # -- ADDITION 1 --
      if i == 0 : tokenFeatures.append("isFirst")
      if i == 1 : tokenFeatures.append("isSecond")
      if i == len(tokens)-2 : tokenFeatures.append("isSecondLast")
      if i == len(tokens)-1 : tokenFeatures.append("isLast")
      # -----------------
    
      sentenceFeatures[i] = tokenFeatures
    
   return sentenceFeatures

## --------- Feature extractor ----------- 
## -- Extract features for each token in each
## -- sentence in each file of given dir

def extract_features(datafile, outfile) :

   # load dictionaries
   dicts = Dictionaries(os.path.join(paths.RESOURCES, "dictionaries.json"))

   # open output file
   outf = open(outfile, "w")

   # create analyzer. We don't need parser/ner here; disabling them speeds up feature extraction.
   # -- ADDITION 2, 3 --
   # 2: add the tagger
   # 3: add the lemmatizer and attribute ruler
   #
   # NOTE: spaCy language models are separate packages. If the transformer model
   # is not installed, fall back to the small model so experiments can run.
   # Install with (inside your venv):
   #   python -m spacy download en_core_web_trf
   #   python -m spacy download en_core_web_sm
   model_candidates = ["en_core_web_trf", "en_core_web_sm"]
   nlp = None
   last_err = None
   for model_name in model_candidates:
      try:
         nlp = spacy.load(
            model_name,
            enable=["tokenizer", "tagger", "attribute_ruler", "lemmatizer"],
            disable=["parser", "ner"],
         )
         break
      except OSError as e:
         last_err = e
         continue

   if nlp is None:
      print("ERROR: No spaCy English model found.", file=sys.stderr)
      print("Tried: " + ", ".join(model_candidates), file=sys.stderr)
      print("Install one of them with: python -m spacy download <model>", file=sys.stderr)
      raise last_err
   # -----------------

   # parse XML file, obtaining a DOM tree
   tree = parse(datafile)

   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences:
      sid = s.attributes["id"].value  # get sentence id
      print(f"extracting sentence {sid}        \r", end="")
      spans = []
      stext = s.attributes["text"].value  # get sentence text
      entities = s.getElementsByTagName("entity")  # get gold standard entities
      for e in entities:
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start, end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ = e.attributes["type"].value
         spans.append((int(start), int(end), typ))

      # convert the sentence to a list of tokens
      tokens = nlp(stext)
      # extract sentence features
      features = extract_sentence_features(tokens, dicts)

      # print features in format expected by CRF/SVM/MEM trainers
      for i, tk in enumerate(tokens):
         # see if the token is part of an entity
         tks, tke = tk.idx, tk.idx + len(tk.text)
         # get gold standard tag for this token
         tag = get_label(tks, tke, spans)
         # print feature vector for this token
         print(
            sid,
            tk.text,
            tks,
            tke - 1,
            tag,
            "\t".join(features[i]),
            sep="\t",
            file=outf,
         )

      # blank line to separate sentences
      print(file=outf)

   # close output file
   outf.close()

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
    
    extract_features(datafile, featfile)

