# Weak supervision for NER

Source code associated with the paper "Named Entity Recognition without Labelled Data: a Weak Supervision Approach" accepted to ACL 2020.

## Requirements:

You should first make sure that the following Python packages are installed:
- `spacy` (version >= 2.2)
- `hmmlearn`
- `snips-nlu-parsers`
- `pandas`
- `numba`
- `scikit-learn`

You should also install the `en_core_web_sm` and `en_core_web_md` models in Spacy.

To run the neural models in `ner.py`, you need also need `pytorch`, `cupy`, `keras` and `tensorflow` installed. 

To run the baselines, you will also need to have `snorkel` installed.

Finally, you also need to download the following files and add them to the `data` directory:
- [`conll2003_spacy.tar.gz`](https://github.com/anonymous-NLP/weak-supervision-for-NER/releases/download/acl2020/conll2003_spacy.tar.gz) (unpack the archive in the same directory)
- [`BTC_spacy.tar.gz`](https://github.com/NorskRegnesentral/weak-supervision-for-NER/releases/download/acl2020/BTC_spacy.tar.gz) (same)
- [`SEC_spacy.tar.gz`](https://github.com/NorskRegnesentral/weak-supervision-for-NER/releases/download/acl2020/SEC_spacy.tar.gz) (same)
- [`wikidata.json`](https://github.com/NorskRegnesentral/weak-supervision-for-NER/releases/download/acl2020/wikidata.json)
- [`wikidata_small.json`](https://github.com/NorskRegnesentral/weak-supervision-for-NER/releases/download/acl2020/wikidata_small.json)
- [`crunchbase.json`](https://github.com/NorskRegnesentral/weak-supervision-for-NER/releases/download/acl2020/crunchbase.json)
- [`conll2003.docbin`](https://github.com/NorskRegnesentral/weak-supervision-for-NER/releases/download/acl2020/conll2003.docbin)

## Quick start

You should first convert your corpus to Spacy `DocBin` format.

Then, to run all labelling functions on your corpus, you can simply:

```
import annotations
annotator = annotations.FullAnnotator().add_all()
annotator.annotate_docbin('path_to_your_docbin_corpus')
```

You can then estimate an HMM model that aggregates all sources:

```
import labelling
hmm = labelling.HMMAnnotator()
hmm.train('path_to_your_docbin_corpus')
```

And run it on your corpus to get the aggregated labels:
```
hmm.annotate_docbin('path_to_your_docbin_corpus')
```

## Step-by-step instructions

More detailed instructions with a step-by-step example are available in the Jupyter Notebook `Weak Supervision.ipynb`. Don't forget to run it using Jupyter to get the visualisation for the NER annotations.

