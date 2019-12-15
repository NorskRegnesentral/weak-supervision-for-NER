#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tarfile, datetime, string, re, subprocess, multiprocessing, random
import os, traceback, urllib, itertools, json, pickle, time
import numpy as np
import pandas, spacy

"""Various methods for analysing sentences, using a Wrapper around Spacy."""


# List of currency symbols and three-letter codes
CURRENCY_SYMBOLS =  {"$", "¥", "£", "€", "kr", "₽", "R$", "₹", "Rp", "₪", "zł", "Rs", "₺", "RS"}

CURRENCY_CODES = {"USD", "EUR", "CNY", "JPY", "GBP", "NOK", "DKK", "CAD", "RUB", "MXN", "ARS", "BGN", 
                  "BRL", "CHF",  "CLP", "CZK", "INR", "IDR", "ILS", "IRR", "IQD", "KRW", "KZT", "NGN", 
                  "QAR", "SEK", "SYP", "TRY", "UAH", "AED", "AUD", "COP", "MYR", "SGD", "NZD", "THB", 
                  "HUF", "HKD", "ZAR", "PHP", "KES", "EGP", "PKR", "PLN", "XAU", "VND", "GBX"}

NOT_NAMED_ENTITIES = {"EPS", "No", "Nb", "n't", "n’t"}


LEGAL_SUFFIXES = {
    'ltd',     # Limited ~13.000
    'llc',     # limited liability company (UK)
    'ltda',    # limitada (Brazil, Portugal)
    'inc',     # Incorporated ~9700
    'co ltd',  # Company Limited ~9200
    'corp',    # Corporation ~5200
    'sa',      # Spółka Akcyjna (Poland), Société Anonyme (France)  ~3200
    'plc',     # Public Limited Company (Great Britain) ~2100
    'ag',      # Aktiengesellschaft (Germany) ~1000
    'gmbh',    # Gesellschaft mit beschränkter Haftung  (Germany)
    'bhd',     # Berhad (Malaysia) ~900
    'jsc',     # Joint Stock Company (Russia) ~900
    'co',      # Corporation/Company ~900
    'ab',      # Aktiebolag (Sweden) ~800
    'ad',      # Akcionarsko Društvo (Serbia), Aktsionerno Drujestvo (Bulgaria) ~600
    'tbk',     # Terbuka (Indonesia) ~500
    'as',      # Anonim Şirket (Turkey), Aksjeselskap (Norway) ~500
    'pjsc',    # Public Joint Stock Company (Russia, Ukraine) ~400
    'spa',     # Società Per Azioni (Italy) ~300
    'nv',      # Naamloze vennootschap (Netherlands, Belgium) ~230
    'dd',      # Dioničko Društvo (Croatia) ~220
    'a s',     # a/s (Denmark), a.s (Slovakia) ~210
    'oao',     # Открытое акционерное общество (Russia) ~190
    'asa',     # Allmennaksjeselskap (Norway) ~160
    'ojsc',    # Open Joint Stock Company (Russia) ~160
    'lp',      # Limited Partnership (US) ~140
    'llp',     # limited liability partnership
    'oyj',     # julkinen osakeyhtiö (Finland) ~120
    'de cv',   # Capital Variable (Mexico) ~120
    'se',      # Societas Europaea (Germany) ~100
    'kk',      # kabushiki gaisha (Japan)
    'aps',     # Anpartsselskab (Denmark)
    'cv',      # commanditaire vennootschap (Netherlands)
    'sas',     # société par actions simplifiée (France)
    'sro',     # Spoločnosť s ručením obmedzeným (Slovakia)
    'oy',      # Osakeyhtiö (Finland)
    'kg',      # Kommanditgesellschaft (Germany)
    'bv',      # Besloten Vennootschap (Netherlands)
    'sarl',    # société à responsabilité limitée (France)
    'srl',     # Società a responsabilità limitata (Italy)
    'sl'       # 	Sociedad Limitada (Spain) 
} 

form_freq_fd = open("data/form_frequencies.json")
FORM_FREQUENCIES = json.load(form_freq_fd)
form_freq_fd.close()
                
    
############################################
# NLP (with Spacy)
############################################
             
            

class Parser:
    """Wrapper class for parsing text or text records with spacy, and correcting 
    incorrect boundaries for named entities."""
    
    def __init__(self, model, min_length_for_truecasing=30, correct_entities=True):
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        self.nlp = spacy.load(model) 
                
        self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"), first=True)
        self.nlp.add_pipe(_correct_tokenisation, after="sentencizer", name="tokenize")
        self.nlp.add_pipe(truecase, after="tokenize", name="truecasing")
        if "parser" in self.nlp.pipe_names:
            self.nlp.add_pipe(_correct_sentence_boundaries, before='parser')
            self.nlp.add_pipe(_correct_parsing, after='parser')
        if correct_entities:
            self.nlp.add_pipe(_correct_entities, after='ner')
        self.vocab = self.nlp.vocab
        
        self.vocab.strings.add("thousand")
        self.vocab.strings.add("million")
        self.vocab.strings.add("billion")
        
        
    def __call__(self, text):
        """Directly parses the text, and performs some postprocessing"""
        if type(text)==str:
            return self.nlp(text)
        elif type(text)==spacy.tokens.Doc:
            doc = text
            for _, proc in self.nlp.pipeline:
                doc = proc(doc)
            return doc
    
    def pipe(self, generator):
        """Applies the spacy parser on a stream of texts. empty texts yield a None."""
        
        generator1, generator2 = itertools.tee(generator, 2)
        parses = self.nlp.pipe((text for text in generator1 if type(text)==str 
                                and len(text)>0)) #, batch_size=8)
        
        for text in generator2:
            if type(text)==str and len(text) > 0:
                parse = next(parses)
                yield parse
            else:
                yield None

     
    def load(self, doc_bytes):
        """Returns a Spacy Doc object from its byte encoding"""
        return spacy.tokens.Doc(self.vocab).from_bytes(doc_bytes)



def _correct_tokenisation(parsed):
    """Correct the tokenisation provided by Spacy"""
    
    tokens, spaces = [], []
    one_more_pass = False
    has_changed = False
    
    tokens_stack = list(parsed)
    while tokens_stack:
        token = tokens_stack.pop(0)
        # Correct the tokenisation of acronyms such as "E/S", by merging them in a single token
        if token.i < len(parsed)-2 and len(token)<=2 and token.is_upper and token.nbor(1).text=="/":
            seq = parsed[token.i:token.i+3].text
            if re.match(r"\b[A-Z]{1,2}(?:/[A-Z]{1,2})+\b", seq):
                tokens.append(seq)
                spaces.append(bool(token.nbor(2).whitespace_))
                tokens_stack.pop(0)
                tokens_stack.pop(0)
                has_changed = True
                continue
              
        # Ensure that currency codes preceding numbers (such as USD3,400) are tokenised
        match = re.match("(%s)([\d\.\,]+.*)"%"|".join(CURRENCY_CODES), token.text)
        if match:
            tokens += [match.group(1), match.group(2)]
            spaces += [False, bool(token.whitespace_)]
            one_more_pass = True
            has_changed = True
            continue
                
        # Ensure that abbreviations such as K, M, B, mln, bln are tokenised properly
        match = re.match("([\d\.\,]+)(bl?n?\.?|ml?n?\.?|k\.+)$", token.text, re.IGNORECASE)
        if match:
            tokens += [match.group(1), match.group(2)]
            spaces += [False, bool(token.whitespace_)]
            has_changed = True
            continue        
        else:
            tokens.append(token.text)
            spaces.append(bool(token.whitespace_))

    if not has_changed:
        return parsed
    
    # Creates a new document with the tokenised words and space information
    new_doc = spacy.tokens.Doc(parsed.vocab, words=tokens, spaces=spaces)
    
    if one_more_pass:
        return _correct_tokenisation(new_doc)
    else:
        return new_doc
            

def truecase(spacy_doc, min_prob=0.25, skip_doc_longer_than=25):
    """Performs truecasing of the tokens in the spacy document. Based on relative frequencies of
    word forms, tokens that 
    (1) are made of letters, with a first letter in uppercase
    (2) and are not sentence start
    (3) and have a relative frequency below min_prob
    ... will be replaced by its most likely case (such as lowercase). """
    
    # Documents that are longer than a specified amount of tokens are not truecased (since 
    # problematic cases typically happen in titles and not longer paragraphs)
    if len(spacy_doc) > skip_doc_longer_than:
        return spacy_doc
    
    tokens = [tok.text for tok in spacy_doc]
    for tok in spacy_doc:
        if tok.is_alpha and not tok.is_sent_start and tok.text[0].isupper():
            token_lc = tok.text.lower()
            if token_lc in FORM_FREQUENCIES:
                frequencies = FORM_FREQUENCIES[token_lc]
                if frequencies.get(tok.text,0) < min_prob:
                    alternative = sorted(frequencies.keys(), key=lambda x: frequencies[x])[-1]
                    tokens[tok.i] = alternative
    
    # Spacy needs to know whether the token is followed by a space
    spaces = [spacy_doc.text[tok.idx+len(tok)].isspace() for tok in spacy_doc[:-1]] + [False]
        
    # Creates a new document with the tokenised words and space information
    spacy_doc2 = spacy.tokens.Doc(spacy_doc.vocab, words=tokens, spaces=spaces)
        
    # Add attributes from the original document
  #  all_attrs = [spacy.attrs.LEMMA, spacy.attrs.TAG, spacy.attrs.DEP, 
  #               spacy.attrs.HEAD, spacy.attrs.ENT_IOB, spacy.attrs.ENT_TYPE]
  #  all_vals = spacy_doc.to_array(all_attrs)
  #  spacy_doc2 = spacy_doc2.from_array(all_attrs, all_vals)
        
    return spacy_doc2



def _correct_sentence_boundaries(doc):
    """Sets manual sentence boundaries for some corner cases (sentences ending with quotes)
    when using Spacy's sentence segmenter."""

    doc.is_parsed = False
    for token in doc[:-1]:
        if token.text in {'‘', '“'}:
            doc[token.i+1].is_sent_start = False
        elif token.text in {'”', '’'}:
            token.is_sent_start = False
            if token.i > 0 and doc[token.i-1].text!=".":
                doc[token.i+1].is_sent_start = False  
    return doc




def _correct_parsing(parsed):
    """Corrects some frequent parsing errors (especially for tokens that are specific 
    to financial texts)"""
    
    for token in parsed:    
        
        # Ensuring that currency codes are of the right POS/TAG
        if token.text in CURRENCY_CODES:
            token.pos = spacy.symbols.PROPN
            token.tag = parsed.vocab.strings["NNP"]
            
        # Ensuring that abbreviated units (mln for million, etc.) are of NUM/CD tag
        elif re.match("(?:bl?n?\.?|ml?n?\.?|k\.+)$", token.text, re.IGNORECASE):
            token.pos = spacy.symbols.NUM
            token.tag = parsed.vocab.strings["CD"]
            if token.dep_=="compound":
                token.dep = parsed.vocab.strings["nummod"]
        elif (token.text=="won" and token.i > 1 and (token.nbor(-1).text[0].isdigit() or
              token.nbor(-1).text.lower() in {"million", "billion", "mln", "bln", "bn", "thousand",
                                 "m", "k", "b", "m.", "k.", "b.", "mln.", "bln.", "bn."})):
            token.pos = spacy.symbols.PROPN
            token.tag = parsed.vocab.strings["NNP"]
            
    return parsed

        
def _correct_entities(parsed, recursive=True):
    """Correct the named entities in Spacy documents (wrong boundaries or entity type)"""

    new_ents = []
    has_changed = False
    
    # Remove errors (words or phrases that are never named entities)
    existing_ents = [ent for ent in parsed.ents if ent.text not in NOT_NAMED_ENTITIES]

    for ent in existing_ents:
        # If the token after the span is a currency symbol, extend the span on the right side
        if (ent.end < len(parsed) and (parsed[ent.end].lemma_.lower() in (CURRENCY_SYMBOLS | {"euro", "cent", "ruble"})
                                       or parsed[ent.end].text.upper() in CURRENCY_CODES)
            and ((ent.end == len(parsed)-1) or (parsed[ent.end].ent_type==0))):
            new_ents.append((ent.start, ent.end+1, spacy.symbols.MONEY))
            has_changed = True

        # Correct entities that go one token too far and include the preposition to
        if (parsed[ent.end-1].lemma_.lower()=="to" and ent.label==spacy.symbols.MONEY):
            new_ents.append((ent.start, ent.end-1, spacy.symbols.MONEY))
            has_changed=True
            
        # Special case to deal with the south-korean currency "won"
        elif (ent.end < len(parsed) and parsed[ent.end].text.lower()=="won"
            and ((ent.end == len(parsed)-1) or (parsed[ent.end].ent_type==0))
             and ent.label in {spacy.symbols.MONEY, spacy.symbols.CARDINAL}):
            new_ents.append((ent.start, ent.end+1, spacy.symbols.MONEY))
            has_changed = True
            
        # Extend MONEY spans if the following token is "million", "billion", etc.
        elif (ent.end < len(parsed) and parsed[ent.end].lemma_.lower() in {"million", "billion", "mln", "bln", "bn", "thousand", 
                                                                           "m", "k", "b", "m.", "k.", "b.", "mln.", "bln.", "bn."}
            and ent.label in {spacy.symbols.MONEY, spacy.symbols.CARDINAL}):
            new_ents.append((ent.start, ent.end+1, ent.label))
            has_changed = True
            
        # If the token preceding the span is a currency symbol or code, expend the span on the left
        elif (ent.start > 0 and parsed[ent.start-1].ent_type==0 and 
              (parsed[ent.start-1].text in CURRENCY_SYMBOLS or parsed[ent.start-1].text in CURRENCY_CODES)):
            new_ents.append((ent.start-1, ent.end, spacy.symbols.MONEY))
            has_changed = True
            
        # If the token preceding the span is #, assign it to a CARDINAL type
        elif (ent.start > 0 and parsed[ent.start-1].ent_type==0 and parsed[ent.start-1].text=='#'):
            new_ents.append((ent.start-1, ent.end, spacy.symbols.MONEY))
            has_changed = True
            
        # If the first token is a #, assign it to a CARDINAL type
        elif (parsed[ent.start].text=='#'):
            new_ents.append((ent.start, ent.end, spacy.symbols.CARDINAL))    
            has_changed = True
            
        # If the first token is a quartal, remove it from the entities
        elif re.match("Q[1-4]", parsed[ent.start].text):
            has_changed = True

        # If the first token of the span is a currency symbol, make sure its label is MONEY
        elif (len(parsed[ent.start].text) >=3 and parsed[ent.start].text[:3] in CURRENCY_CODES 
              and ent.label != spacy.symbols.MONEY):
            new_ents.append((ent.start, ent.end, spacy.symbols.MONEY))    
            has_changed = True
        
        # If the entity contains "per cent", make sure the entity label is percent, not money
        elif len(ent)>=3 and ent.text.endswith("per cent") and ent.label != spacy.symbols.PERCENT :
            new_ents.append((ent.start, ent.end, spacy.symbols.PERCENT))    
            has_changed = True
        
        # Fix expression with pennies such as 520.0p
        elif parsed[ent.end-1].text[0].isdigit() and ent.text[-1]=='p' and ent.label != spacy.symbols.MONEY:
            new_ents.append((ent.start, ent.end, spacy.symbols.MONEY))    
            has_changed = True
            
        # If the next token is a legal suffix, make sure the label is ORG
        elif (ent.end < len(parsed) and parsed[ent.end].lower_.rstrip(".") in LEGAL_SUFFIXES):
            new_ents.append((ent.start, ent.end+1, spacy.symbols.ORG))    
            has_changed = True   
            
        # If the last token is a legal suffix (and the entity has at least 2 tokens), make sure the label is ORG
        elif parsed[ent.end-1].lower_.rstrip(".") in LEGAL_SUFFIXES and (ent.end > ent.start+1) and ent.label!=spacy.symbols.ORG:
            new_ents.append((ent.start, ent.end, spacy.symbols.ORG))    
            has_changed = True            
        
        # Otherwise, add the entity if it does not overlap with any of the new ones
        elif not new_ents or new_ents[-1][1] < ent.end:
            new_ents.append((ent.start, ent.end, ent.label))
        
    # Loop on the tokens to find occurrences of currency symbols followed by numeric values
    # (which remain often undetected in the current entities)
    for token in parsed:
        if (token.text in CURRENCY_CODES|CURRENCY_SYMBOLS and token.ent_type!=spacy.symbols.MONEY 
            and token.i < len(parsed)-1 and (parsed[token.i+1].text[0].isdigit() or 
                                             parsed[token.i+1].text in CURRENCY_SYMBOLS)):
            entity_end = token.i+2
            for i in range(token.i+2, len(parsed)):
                if any([i>=start and i <end for start,end, _ in new_ents]):
                    entity_end = i+1
                else:
                    break
            new_ents.append((token.i,entity_end, spacy.symbols.MONEY))
            has_changed = True

    new_ents = sorted(new_ents, key=lambda p: p[0])

    # We need to deal with overlapping named entities by merging them
    merge_loop = True
    while merge_loop:
        merge_loop = False
        new_ents2 = list(new_ents)
        for i, (ent_start, ent_end, ent_label) in enumerate(new_ents2):
            for j, (ent2_start, ent2_end, ent2_label) in enumerate(new_ents2[i+1:i+5]):
                if ent_end>ent2_start or (ent_end==ent2_start and ent_label==ent2_label):
                    del new_ents[i+j]
                    # If one label is MONEY, assume the merge is MONEY as well
                    if ent_label==spacy.symbols.MONEY or ent2_label==spacy.symbols.MONEY:
                        new_ents[i] = (ent_start, ent2_end, spacy.symbols.MONEY)
                    # Otherwise, take the label of the longest sequence
                    elif ent2_end-ent2_start >= ent_end-ent_start:
                        new_ents[i] = (ent_start, ent2_end, ent2_label)
                    else:
                        new_ents[i] = (ent_start, ent2_end, ent_label)
                    merge_loop = True
                    has_changed = True
                    break
            if merge_loop:
                break
                

    # If something has changed, create new spans and run the method once more
    if has_changed:
        new_spans = tuple(spacy.tokens.Span(parsed, start, end, symbol) 
                          for (start, end, symbol) in new_ents)
        parsed.ents = new_spans
        if recursive:  
            return _correct_entities(parsed, False)
    return parsed


