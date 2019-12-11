import hmmlearn.hmm, hmmlearn
import numpy as np
from numba import jit, njit, prange
import pickle, gc, itertools
import annotations

LABELS = ['CARDINAL', "COMPANY", 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 
          'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

POSITIONED_LABELS = ["O"] + ["%s-%s"%(bilu, label) for label in LABELS for bilu in "BILU"]

LABEL_INDICES = {label:i for i, label in enumerate(POSITIONED_LABELS)}

SOURCE_NAMES = ['BTC', 'BTC+c', 'SEC', 'SEC+c', 'company_type_detector',  'compound_detector', 
                'conll2003', 'conll2003+c', 'core_web_md', 'core_web_md+c', 'crunchbase_cased', 'crunchbase_uncased', 'date_detector', 
                'doc_history', 'doc_majority_cased', 'doc_majority_uncased', 'full_name_detector', 'geo_cased', 'geo_uncased', 
                'infrequent_compound_detector', 'infrequent_nnp_detector', 'infrequent_proper2_detector', 'infrequent_proper_detector',
                'legal_detector', 'misc_detector', 'money_detector',
                'multitoken_crunchbase_cased', 'multitoken_crunchbase_uncased', 'multitoken_geo_cased', 'multitoken_geo_uncased', 
                'multitoken_product_cased',  'multitoken_product_uncased', 'multitoken_wiki_cased', 'multitoken_wiki_small_cased',
                'multitoken_wiki_small_uncased', 'multitoken_wiki_uncased', 'nnp_detector', 'number_detector', 'product_cased', 
                'product_uncased', 'proper2_detector', 'proper_detector', 'snips', 'time_detector', 'wiki_cased', 'wiki_small_cased', 
                'wiki_small_uncased', 'wiki_uncased']

NUMBER_NERS = ["CARDINAL", "DATE", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "TIME"]

SOURCE_PRIORS = {'BTC': {l:(0.4,0.4) if l in ["COMPANY", "ORG", "PERSON", "GPE", "LOC"] else (0.3,0.3) for l in LABELS if l not in NUMBER_NERS},
                 'BTC+c': {l:(0.5,0.5) if l in ["COMPANY", "ORG", "PERSON", "GPE", "LOC", "MONEY"] else (0.4,0.4) for l in LABELS},
                 'SEC': {l:(0.1,0.1) if l in ["COMPANY", "ORG", "PERSON", "GPE", "LOC"] else (0.05,0.05) for l in LABELS if l not in NUMBER_NERS},
                 'SEC+c': {l:(0.1,0.1) if l in ["COMPANY", "ORG", "PERSON", "GPE", "LOC", "MONEY"] else (0.05,0.05) for l in LABELS},
                 'company_type_detector': {'COMPANY': (0.9999, 0.4)},
                 'compound_detector': {l:(0.7,0.8) if l not in NUMBER_NERS else (0.01, 0.01) for l in LABELS},
                 'conll2003': {l:(0.7,0.7) if l in ["COMPANY", "ORG", "PERSON", "GPE", "LOC"] else (0.4,0.4) for l in LABELS if l not in NUMBER_NERS},
                 'conll2003+c': {l:(0.7,0.7) if l in ["COMPANY", "ORG", "PERSON", "GPE", "LOC"] else (0.4,0.4) for l in LABELS},
                 "core_web_md": {l:(0.9,0.9) for l in LABELS},
                 "core_web_md+c": {l:(0.95,0.95) for l in LABELS},
                 "crunchbase_cased": {l:(0.7,0.6) for l in ["PERSON", "ORG", "COMPANY"]},
                 "crunchbase_uncased": {l:(0.6,0.7) for l in ["PERSON", "ORG", "COMPANY"]},
                 'date_detector': {'DATE': (0.9, 0.9)},
                 'doc_history': {l:(0.99, 0.4) for l in ["PERSON", "COMPANY"]},
                 'doc_majority_cased':{l:(0.98, 0.4) for l in LABELS},
                 'doc_majority_uncased':{l:(0.95, 0.5) for l in LABELS},
                 'full_name_detector': {'PERSON': (0.9999, 0.4)},
                 "geo_cased":{l:(0.8,0.8) for l in ["GPE", "LOC"]},
                 "geo_uncased":{l:(0.8,0.8) for l in ["GPE", "LOC"]},
                 'infrequent_compound_detector': {l:(0.7,0.8) if l not in NUMBER_NERS else (0.01, 0.01) for l in LABELS},
                 'infrequent_nnp_detector':  {l:(0.7,0.8) if l not in NUMBER_NERS else (0.01, 0.01) for l in LABELS},
                 'infrequent_proper2_detector':  {l:(0.7,0.8) if l not in NUMBER_NERS else (0.01, 0.01) for l in LABELS},
                 'infrequent_proper_detector':  {l:(0.7,0.8) if l not in NUMBER_NERS else (0.01, 0.01) for l in LABELS},
                 'legal_detector': {"LAW":(0.8,0.8)},
                 'misc_detector': {l:(0.7,0.7) for l in ["NORP", "EVENT", "FAC", "GPE", "LANGUAGE"]},
                 'money_detector': {'MONEY': (0.9, 0.9)},
                 'multitoken_crunchbase_cased': {l:(0.8, 0.6) for l in ["PERSON", "ORG", "COMPANY"]},
                 'multitoken_crunchbase_uncased': {l:(0.7, 0.7) for l in ["PERSON", "ORG", "COMPANY"]},
                 'multitoken_geo_cased': {l:(0.8, 0.6) for l in ["GPE", "LOC"]},
                 'multitoken_geo_uncased': {l:(0.7, 0.7) for l in ["GPE", "LOC"]},
                 'multitoken_product_cased': {"PRODUCT":(0.8, 0.6)},
                 'multitoken_product_uncased': {"PRODUCT":(0.7, 0.7)},
                 'multitoken_wiki_cased': {l:(0.8, 0.6) for l in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
                 'multitoken_wiki_small_cased': {l:(0.8, 0.6) for l in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
                 'multitoken_wiki_small_uncased': {l:(0.7, 0.7) for l in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
                 'multitoken_wiki_uncased': {l:(0.7, 0.7) for l in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
                 'nnp_detector':  {l:(0.8,0.8) if l not in NUMBER_NERS else (0.01, 0.01) for l in LABELS},
                 "number_detector":{l:(0.9,0.9) for l in ["CARDINAL", "ORDINAL", "QUANTITY", "PERCENT"]},
                 'product_cased': {"PRODUCT":(0.7, 0.6)},
                 'product_uncased': {"PRODUCT":(0.6, 0.7)},
                 'proper2_detector':  {l:(0.6,0.8) if l not in NUMBER_NERS else (0.01, 0.01) for l in LABELS},
                 'proper_detector':  {l:(0.6,0.8) if l not in NUMBER_NERS else (0.01, 0.01) for l in LABELS},
                 "snips":{l:(0.8,0.8) for l in ["DATE", "TIME", "PERCENT", "CARDINAL", "ORDINAL", "MONEY"]},
                 'time_detector': {'TIME': (0.9, 0.9)},
                 'wiki_cased': {l:(0.6, 0.5) for l in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
                 'wiki_small_cased': {l:(0.7, 0.6) for l in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
                 'wiki_small_uncased': {l:(0.6, 0.7) for l in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]},
                 'wiki_uncased': {l:(0.5, 0.6) for l in ["PERSON", "GPE", "LOC", "ORG", "COMPANY", "PRODUCT"]}}

# In some rare cases (due to specialisations of corrections of labels), we also need to add some other labels
for source in ["BTC", "BTC+c", "SEC", "SEC+c", "conll2003", "conll2003+c"]:  
    SOURCE_PRIORS[source].update({l:(0.8,0.01) for l in NUMBER_NERS})

OUT_RECALL = 0.9
OUT_PRECISION = 0.8
  

############################################
# BASE CLASS TO UNIFY ANNOTATIONS
############################################


class UnifiedAnnotator(annotations.BaseAnnotator):
    """Base class for annotators that seek to 'unify' the annotations from other
    supervision sources. Must implement two class: 'train' and 'label'"""
    
    def __init__(self, sources_to_keep=None, source_name="HMM"):
        annotations.BaseAnnotator.__init__(self)
        if sources_to_keep is None:
            self.source_indices_to_keep = {i for i in range(len(SOURCE_NAMES))}
        else:
            print("Using", sources_to_keep, "as supervision sources")
            self.source_indices_to_keep = {SOURCE_NAMES.index(s) for s in sources_to_keep} 
        self.source_name = source_name
        
    def train(self, docbin_file, cutoff=None):
        """Trains the parameters of the annotator based on the annotation in the provided
        docbin file"""
        
        raise NotImplementedError()


    def label(self, doc):
        """Returns two lists, one list with the predicted label for each token in the document,
        and one with the associated probabilities according to the model. """
        
        raise NotImplementedError()
        
        
    def annotate(self, doc):
        """Annotates the document with a new layer of annotation with the model predictions"""

        doc.user_data["annotations"][self.source_name] = {}

        predicted, confidences = self.label(doc)  
        i = 0
        while i < len(predicted):
            if predicted[i]!= "O":
                if predicted[i].startswith("U-") or predicted[i].startswith("I-") or predicted[i].startswith("L-"):
                    conf = round(confidences[i],3)
                    doc.user_data["annotations"][self.source_name][(i, i+1)] = ((predicted[i][2:], conf),)
                    i += 1 
                elif predicted[i].startswith("B-"):
                    start = i
                    label = predicted[i][2:]
                    i += 1
                    while i < (len(doc)-1) and predicted[i] != "O" and predicted[i].startswith("I-"):
                        i += 1
                    if i < len(doc) and predicted[i].startswith("L-"):
                        conf = round(confidences[start:i+1].max().mean(),3)
                        doc.user_data["annotations"][self.source_name][(start, i+1)] = ((label, conf),)
                    i += 1
            else:
                i += 1 
        return doc 
        

    def extract_sequence(self, doc):
        """Convert the annotations of a spacy document into an array of observations of shape 
        (nb_sources, nb_biluo_labels)"""

        doc = self.specialise_annotations(doc)
        sequence = np.zeros((len(doc), len(SOURCE_NAMES), len(POSITIONED_LABELS)), dtype=np.float32)
        for i, source in enumerate(SOURCE_NAMES):
            sequence[:,i,0] = 1.0
            if source not in doc.user_data["annotations"] or i not in self.source_indices_to_keep:
                continue
            for (start,end), vals in doc.user_data["annotations"][source].items():
                for label, conf in vals:
                    if label in {"MISC", "ENT"}:
                        continue
                    elif start >= len(doc):
                        print("wrong boundary")
                        continue
                    elif end > len(doc):
                        print("wrong boundary2")
                        end = len(doc)
                    sequence[start:end, i, 0] = 0.0
                    if end-start==1:
                        sequence[start, i, LABEL_INDICES["U-%s"%label]] = conf
                    else:
                        sequence[start,i, LABEL_INDICES["B-%s"%label]] = conf
                        sequence[start+1:end-1,i, LABEL_INDICES["I-%s"%label]] = conf                           
                        sequence[end-1,i, LABEL_INDICES["L-%s"%label]] = conf                                  

        return sequence


    def specialise_annotations(self, doc):
        """Replace generic ENT or MISC values with the most likely labels from other annotators"""

        to_add = []
        
        annotated = doc.user_data.get("annotations", [])
        for source in annotated:

            other_sources = [s for s in annotated if "HMM" not in s 
                             and s!="gold" and s in SOURCE_NAMES
                             and s!=source and "proper" not in s and "nnp_" not in s 
                             and "SEC"  not in s 
                             and "compound" not in s and "BTC" not in s
                             and SOURCE_NAMES.index(s) in self.source_indices_to_keep]

            current_spans = dict(annotated[source])
            for (start,end), vals in current_spans.items():
                for label, conf in vals:
                    if label in {"ENT", "MISC"}:

                        label_counts = {}
                        for other_source in other_sources:
                            overlaps = annotations.get_overlaps(start, end, annotated, [other_source])
                            for (start2, end2, vals2) in overlaps:
                                for label2, conf2 in vals2:
                                    if label2 not in {"ENT", "MISC"}:  

                                        # simple heuristic for the confidence of the label specialisation
                                        conf2 = conf2 if (start2==start and end2==end) else 0.3*conf2
                                        conf2 = conf2*SOURCE_PRIORS[other_source][label2][0]
                                        label_counts[label2] = label_counts.get(label2, 0) + conf*conf2
                        vals = tuple((l,SOURCE_PRIORS[source][l][0]*conf2/sum(label_counts.values())) 
                                     for l, conf2 in label_counts.items())
                        to_add.append((source, start, end, vals))

        for source,start,end, vals in to_add:
            doc.user_data["annotations"][source][(start,end)] = vals

        return doc

                                    
    def save(self, filename):
        fd = open(filename, "wb")
        pickle.dump(self, fd)
        fd.close()
            
    @classmethod
    def load(cls, pickle_file):
        print("Loading", pickle_file)
        fd = open(pickle_file, "rb")
        ua = pickle.load(fd)
        fd.close()
        return ua
    

############################################
# HMM ANNOTATOR
############################################

    
class HMMAnnotator(hmmlearn.hmm._BaseHMM, UnifiedAnnotator):
  
    def __init__(self, sources_to_keep=None, source_name="HMM", informative_priors=True):
        hmmlearn.hmm._BaseHMM.__init__(self, len(POSITIONED_LABELS), verbose=True, n_iter=10)
        UnifiedAnnotator.__init__(self, sources_to_keep= sources_to_keep, source_name=source_name)
        self.informative_priors = informative_priors
        
  
    def train(self, docbin_file, cutoff=None):
        """Train the HMM annotator based on the docbin file"""
               
        spacy_docs = annotations.docbin_reader(docbin_file, cutoff=cutoff)
        X_stream = (self.extract_sequence(doc) for doc in spacy_docs)
        streams = itertools.tee(X_stream, 3)
        self._initialise_startprob(streams[0])
        self._initialise_transmat(streams[1])
        self._initialise_emissions(streams[2])
        self._check()

        self.monitor_._reset()
        for iter in range(self.n_iter):
            print("Starting iteration", (iter+1))
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            
            nb_docs = 0
            for doc in annotations.docbin_reader(docbin_file, cutoff=cutoff):
                X = self.extract_sequence(doc)
                framelogprob = self._compute_log_likelihood(X)
                if framelogprob.max(axis=1).min() < -100000:
                    print("problem found!")
                    return framelogprob

                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_sufficient_statistics(
                    stats, X, framelogprob, posteriors, fwdlattice,
                    bwdlattice)
                nb_docs += 1
                
                if nb_docs % 1000 == 0:
                    print("Number of processed documents:", nb_docs)
            print("Finished E-step with %i documents"%nb_docs)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)

            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break

        return self
        
    def label(self, doc):
        """Makes a list of predicted labels (using Viterbi) for each token, along with
        the associated probability according to the HMM model."""
        
        if not hasattr(self, "emission_probs"):
            raise RuntimeError("Model is not yet trained")
            
        doc.user_data["annotations"][self.source_name] = {}

        sequence = self.extract_sequence(doc)
        framelogprob = self._compute_log_likelihood(sequence)
        logprob, predicted = self._do_viterbi_pass(framelogprob)
        _check_outputs(predicted)
        
        labels = [POSITIONED_LABELS[x] for x in predicted]
            
        predicted_proba = np.exp(framelogprob)
        predicted_proba = predicted_proba / predicted_proba.sum(axis=1)[:,np.newaxis]
        
        confidences = np.array([probs[x] for (probs, x) in zip(predicted_proba, predicted)])
        return labels, confidences

    
    def _initialise_startprob(self, X_stream):
    
        print("Constructing start distribution prior...")

        init_counts = np.zeros((len(POSITIONED_LABELS),))
        
        if self.informative_priors:
            source_with_best_coverage = sorted(SOURCE_NAMES, key=lambda x: len(SOURCE_PRIORS[x]))[-1]
            print("Using source", source_with_best_coverage, "to estimate start probability priors")
            source_index = SOURCE_NAMES.index(source_with_best_coverage)
            for X in X_stream:
                init_counts[X[0,source_index].argmax()] += 1

        for i, label in enumerate(POSITIONED_LABELS):
            if i==0 or label.startswith("B-") or label.startswith("U-"):
                init_counts[i] += 1
                
        self.startprob_prior = init_counts +1            
        self.startprob_ = np.random.dirichlet(init_counts + 1E-10)

        
    def _initialise_transmat(self, X_stream):
    
        print("Constructing transition matrix prior...")
        trans_counts = np.zeros((len(POSITIONED_LABELS), len(POSITIONED_LABELS)))

        if self.informative_priors:
            source_with_best_coverage = sorted(SOURCE_NAMES, key=lambda x: len(SOURCE_PRIORS[x]))[-1]
            source_index = SOURCE_NAMES.index(source_with_best_coverage)
            for X in X_stream:
                for k in range(0, len(X)-1):
                    trans_counts[X[k,source_index].argmax(), X[k+1,source_index].argmax()] += 1
        
        for i, label in enumerate(POSITIONED_LABELS):
            if label.startswith("B-") or label.startswith("I-"):
                trans_counts[i, POSITIONED_LABELS.index("I-"+label[2:])] += 1
                trans_counts[i, POSITIONED_LABELS.index("L-"+label[2:])] += 1
            elif i==0 or label.startswith("U-") or label.startswith("L-"):
                for j, label2 in enumerate(POSITIONED_LABELS):
                    if j==0 or label2.startswith("B-") or label2.startswith("U-"):
                        trans_counts[i,j] += 1

        self.transmat_prior = trans_counts + 1
        self.transmat_ = np.vstack([np.random.dirichlet(trans_counts2  + 1E-10) 
                                    for trans_counts2 in trans_counts])
        
        
    def _initialise_emissions(self, X_stream, strength=1000):
        
        print("Constructing emission probabilities...")
        
        obs_counts = np.zeros((len(SOURCE_NAMES), len(POSITIONED_LABELS)), dtype=np.float64)
        if self.informative_priors:
            for X in X_stream:
                obs_counts += X.sum(axis=0)            
        for source_index, source in enumerate(SOURCE_NAMES):
            obs_counts[source_index, 0]+= 1
            for pos_index, pos_label in enumerate(POSITIONED_LABELS[1:]):
                if pos_label[2:] in SOURCE_PRIORS[source]:
                    obs_counts[source_index,pos_index] += 1
                    
        obs_probs = obs_counts / obs_counts.sum(axis=1)[:,np.newaxis]
        
        matrix = np.zeros((len(SOURCE_NAMES), len(POSITIONED_LABELS), len(POSITIONED_LABELS)))
        
        label_indices = {label:i for i, label in enumerate(POSITIONED_LABELS)}

        for source_index, source in enumerate(SOURCE_NAMES):

            for pos_index, pos_label in enumerate(POSITIONED_LABELS):

                # Simple case: set P(O=x|Y=x) to be the recall
                recall = 0
                if pos_index == 0 or not self.informative_priors:
                    recall = OUT_RECALL                  
                elif pos_label[2:] in SOURCE_PRIORS[source]:
                    _, recall = SOURCE_PRIORS[source][pos_label[2:]]  
                matrix[source_index, pos_index, pos_index] = recall
                    
                for pos_index2, pos_label2 in enumerate(POSITIONED_LABELS):
                    if pos_index2 == pos_index:
                        continue
                    elif pos_index2== 0 or not self.informative_priors:
                        precision = OUT_PRECISION
                    elif pos_label2[2:] in SOURCE_PRIORS[source]:
                        precision, _ = SOURCE_PRIORS[source][pos_label2[2:]]
                    else:
                        precision = 1.0
                    
                    
                    # Otherwise, we set the probability to be inversely proportional to the precision 
                    # and the (unconditional) probability of the observation
                    error_prob = (1-recall) * (1-precision) * (0.001 + obs_probs[source_index, pos_index2])
                    
                    # We increase the probability for boundary errors (i.e. I-ORG -> B-ORG)
                    if self.informative_priors and pos_index > 0 and pos_index2 > 0 and pos_label[2:]==pos_label2[2:]:
                        error_prob *= 5                        

                    # We increase the probability for errors with same boundary (i.e. I-ORG -> I-GPE)
                    if self.informative_priors and pos_index > 0 and pos_index2 > 0 and pos_label[0]==pos_label2[0]:
                        error_prob *= 2                        

                    matrix[source_index, pos_index, pos_index2] = error_prob
                
                error_indices = [i for i in range(len(POSITIONED_LABELS)) if i!=pos_index]
                error_sum = matrix[source_index, pos_index, error_indices].sum()
                matrix[source_index, pos_index, error_indices] /= (error_sum/(1-recall))
        
        self.emission_priors = matrix * strength
        self.emission_probs = matrix
    
    
    def generate_sample_from_state(self, state, random_state=None):
        result = np.zeros((len(SOURCE_NAMES), len(POSITIONED_LABELS)), dtype=bool)
        for i in range(len(SOURCE_NAMES)):
            choice = np.random.choice(self.emission_probs.shape[2], 
                                      p=self.emission_probs[i,state])
            result[i,choice] = True
        return result
    
    def _compute_log_likelihood(self, X):
        
        logsum = np.zeros((len(X), len(POSITIONED_LABELS)))
        emission_logs = np.log(self.emission_probs + 1E-10)
        for source_index in range(len(SOURCE_NAMES)):
            if source_index not in self.source_indices_to_keep:
                continue
            probs = np.dot(X[:,source_index,:], self.emission_probs[source_index,:,:].T)
            logsum += np.ma.log(probs).filled(-np.inf)
            
        # We also add a constraint that the probability of a state is zero is no labelling functions observes it
        X_all_obs = X.sum(axis=1).astype(bool)
        logsum = np.where(X_all_obs, logsum, -np.inf)
        
        return logsum

    def _initialize_sufficient_statistics(self):
        stats = super(HMMAnnotator, self)._initialize_sufficient_statistics()
        stats['obs'] = np.zeros(self.emission_probs.shape)
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(HMMAnnotator, self)._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
        if 'e' in self.params:
            sum_posteriors(stats["obs"], X, posteriors)
 
    def _do_mstep(self, stats):
        super(HMMAnnotator, self)._do_mstep(stats)
        if 'e' in self.params:
            
            emission_counts = self.emission_priors + stats['obs']
            emission_probs = emission_counts / (emission_counts + 1E-100).sum(axis=2)[:,:, np.newaxis]
            self.emission_probs = np.where(self.emission_probs > 0, emission_probs, 0)

           

    
@njit(parallel=True)
def sum_posteriors(stats, X, posteriors):
    for i in prange(X.shape[0]):
        for source_index in range(X.shape[1]):
            for j in range(X.shape[2]):
                obs = X[i,source_index, j]
                if obs>0:
                    stats[source_index,:,j] += (obs*posteriors[i])
 
    
                 
def _check_outputs(predictions):
    """Checks whether the output is consistent"""
        
    prev_bilu_label = "O"
    for i in range(len(predictions)):
        bilu_label = POSITIONED_LABELS[predictions[i]]
        if prev_bilu_label[0] in {"L", "U", "O"} and bilu_label[0] in {"I", "L"}:
            print("inconsistent start of NER at pos %i:"%i, bilu_label, "after", prev_bilu_label)
        elif prev_bilu_label[0] in {"B", "I"}:
            if bilu_label[0] not in {"I", "L"} or bilu_label[2:]!=prev_bilu_label[2:]:
                print("inconsistent continuation of NER at pos %i:"%i, bilu_label, "after", prev_bilu_label)
        prev_bilu_label = bilu_label
   

############################################
# BASELINE ANNOTATORS
############################################

     
class MajorityVoter(UnifiedAnnotator):
    """Simple majority voter, taking the most common label of the sources. 
    The parameter nb_sources_threshold specifies the minimum number of sources
    emitting a label != "O" to consider the token to be part of a named entity.
    """
    
    def __init__(self, sources_to_keep=None, 
                 source_name="majority_voter", 
                 nb_sources_threshold=10):
        
        UnifiedAnnotator.__init__(self, sources_to_keep, source_name)
        
        self.sources_nb_threshold = nb_sources_threshold
    
    def label(self, doc):
        sequence = self.extract_sequence(doc)

        labels = []
        confidences = []
        for i in range(len(doc)):
            counts = np.bincount(sequence[i].argmax(axis=1))
            if counts[1:].sum() >= self.sources_nb_threshold:
                labels.append(POSITIONED_LABELS[counts[1:].argmax()+1])
                confidences.append(counts[1:].max() / counts[1:].sum())
            else:
                labels.append("O")
                confidences.append(counts[0] / counts.sum())
            
        return labels, np.array(confidences)

    
class SnorkelModel(UnifiedAnnotator):
    """Snorkel-based model. The model first extracts a list of candidate spans 
    from a few trustworthy sources, and then relies on the full set of sources
    for the classification"""
    
    def __init__(self, sources_to_keep=None, source_name="snorkel", 
                 candidate_sources=["proper2_detector", "nnp_detector", "compound_detector"]):
        UnifiedAnnotator.__init__(self, sources_to_keep, source_name)
        self.candidate_sources = candidate_sources
    
    def train(self, docbin_file):
        """Trains the Snorkel model on the provided corpus"""
        
        import snorkel.labeling
        all_obs = []
        for doc in annotations.docbin_reader(docbin_file):
            doc = self.specialise_annotations(doc)
            spans, obs = self._get_inputs(doc)
            all_obs.append(obs)
            if len(all_obs) > 5:
                break
        all_obs = np.vstack(all_obs)
        self.label_model = snorkel.labeling.LabelModel(len(LABELS) + 1)
        self.label_model.fit(all_obs)
        
    def _get_inputs(self, doc):
        """Returns the list of spans and the associated labels for each source (-1 to abtain)"""
        
        spans = sorted(annotations.get_spans(doc, self.candidate_sources))
        span_indices = {span:i for i, span in enumerate(spans)}
        obs = np.full((len(spans), max(self.source_indices_to_keep)+1), -1)
        for source_index in sorted(self.source_indices_to_keep):
            source = SOURCE_NAMES[source_index]
            if source in doc.user_data["annotations"]:
                for (start,end), vals in doc.user_data["annotations"][source].items():
                    if (start,end) in span_indices and len(vals) > 0:
                        span_index = span_indices[(start,end)]
                        vals = sorted(vals, key=lambda x: x[1])
                        obs[span_index, source_index] = 1 + LABELS.index(vals[-1][0])
                        
        return spans, obs
        
        
    def annotate(self, doc):
        """Annotates the document with the Snorkel output"""
        
        doc.user_data["annotations"][self.source_name] = {}
        doc = self.specialise_annotations(doc)
        spans, obs = self._get_inputs(doc)
        predict_probs = self.label_model.predict_proba(obs)
        for (start,end), probs_for_span in zip(spans, predict_probs):
            label_index = probs_for_span.argmax()
            if label_index > 0:
                label = LABELS[label_index-1]
                prob = probs_for_span.max()
                doc.user_data["annotations"][self.source_name][(start,end)] = ((label, prob),)
        return doc
    
    
            
