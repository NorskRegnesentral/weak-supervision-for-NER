  
import pandas, re, spacy

"""This module contains data structures useful for other modules, in particular tries (for searching for occurrences
of large terminology lists in texts) and interval trees (for representing annotations of text spans)."""

class Trie: 
    """Implementation of a trie for searching for occurrences of terms in a text."""
    
    def __init__(self): 
        self.start = Node() 
        self.value_mapping = {}   # to avoid storing many (identical) strings as values, we use a mapping
        self.index_mapping = {}   # (reverse dictionary of value_mapping)
        self.len = 0

        
    def longest_prefix(self, key, case_sensitive=True): 
        """Search for the longest prefix. The key must be a list of tokens. The method
        returns the prefix length (in number of covered tokens) and the corresponding value. """
        current = self.start 
        value = None
        prefix_length = 0 
        for i, c in enumerate(key):
            if current.children is None:
                break
            elif c in current.children: 
                current = current.children[c] 
                if current.value: 
                    value = current.value 
                    prefix_length = i+1
            elif not case_sensitive:
                found_alternative = False
                for alternative in {c.title(), c.lower(), c.upper()}:
                    if alternative in current.children:
                        current = current.children[alternative] 
                        if current.value: 
                            value = current.value 
                            prefix_length = i+1
                        found_alternative = True
                        break
                if not found_alternative:
                    break
            else:
                break
        value = self.index_mapping[value] if value is not None else None
        return prefix_length, value 

    def __contains__(self, key):
        return self[key]!=None
    
    def __getitem__(self, key): 
        current = self.start 
        for i, c in enumerate(key): 
            if current.children is not None and c in current.children: 
                current = current.children[c] 
            else: 
                return None 
        return self.index_mapping.get(current.value, None)
 
    
    def __setitem__(self, key, value): 
        current = self.start 
        for c in key: 
            if current.children is None:
                new_node = Node() 
                current.children = {c:new_node}
                current = new_node 
            elif c not in current.children: 
                new_node = Node() 
                current.children[c] = new_node 
                current = new_node 
            else: 
                current = current.children[c] 
        if value in self.value_mapping:
            value_index = self.value_mapping[value]
        else:
            value_index = len(self.value_mapping) + 1
            self.value_mapping[value] = value_index
            self.index_mapping[value_index] = value
        current.value = value_index
        self.len += 1
        
    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self._iter_from_node(self.start) 
        
    def _iter_from_node(self, n):
        if n.value is not None:
            yield (), n.value
        if n.children is not None:
            for child_key, child_value in n.children.items():
                for subval_key, subval_value in self._iter_from_node(child_value):
                    yield (child_key, *subval_key), subval_value
    
    def __repr__(self):
        return list(self).__repr__()
    

class Node: 
    """Representation of a trie node"""
    __slots__ = ('children', 'value')
    def __init__(self): 
        self.children = None
        self.value = None 
        

        
def tokenise_fast(text):
    """Fast tokenisation of a string (designed to be roughly similar to Spacy's)"""
    tokens = text.split(" ")
    
    tokens2 = []
    for token in tokens:
        
        # Special case: handle hyphenised tokens like Jean-Pierre
        if "-" in token:
            subtokens = token.split("-")
            for j, sub_token in enumerate(subtokens):
                tokens2.append(sub_token)
                if j < len(subtokens)-1:
                    tokens2.append("-")
                    
        # Special case: handle tokens like 3G, where Spacy tokenisation is unpredictable
        elif re.match("\d+[A-Za-z]+", token):
            if not hasattr(tokenise_fast, "nlp"):
                tokenise_fast.nlp = spacy.load("en", disable=["tagger","parser","ner"])
            for tok in tokenise_fast.nlp(token):
                tokens2.append(tok.text)
        else:
            tokens2.append(token)
    tokens = tokens2
    
    i = 0 
    while i < len(tokens): 
        
        # Special case: handle genitives
        if tokens[i].endswith("'s"): 
            tokens[i] = tokens[i].rstrip("s").rstrip("'") 
            tokens.insert(i+1, "'s") 
            i += 2 
        else: 
            i += 1
            
    tokens = [tok for tok in tokens if len(tok)>0]
    return tokens


####################################
# CODE BELOW IS OUTDATED!
####################################