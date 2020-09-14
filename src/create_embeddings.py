#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes, Sérgio Matos

https://github.com/ruiantunes/biocreative-vi-track-5-chemprot


BioCreative VI - Track 5 (CHEMPROT).

This script creates part-of-speech (POS), and dependency parsing (DEP)
embeddings from the full ChemProt dataset. The `gensim` framework [1]_
is used to create `KeyedVectors` models.


References
----------
.. [1] https://radimrehurek.com/gensim/models/word2vec.html

"""

# built-in modules (sys.builtin_module_names)
from itertools import product

# third-party modules
from gensim.models import Word2Vec
import os

# own modules
from support import DATA
from support import LABEL2INDEX
from support import load_data_from_zips
from utils import create_directory


# input arguments

# training groups
GROUPS = ['training', 'development', 'test_gs']

# vector sizes
SIZES = [20, 50, 100]

# window sizes
WINDOWS = [3, 5, 10]

# iterations
ITERS = [10, 50, 100]

# other contants

# output directory
OUT = 'word2vec'
create_directory(OUT)

ZIPS = [
    os.path.join(
        DATA,
        'chemprot_{}'.format(group),
        'support',
        'processed_corpus.zip',
    ) for group in GROUPS
]

# load dataset
dataset = load_data_from_zips(zips=ZIPS, label2index=LABEL2INDEX)

# get sentences (list of lists of tokens)
sentences = {k: [] for k in ('pos', 'dep')}
for features in dataset['data']:
    for wrd, pos, dep in features:
        sentences['pos'].append(pos)
        sentences['dep'].append(dep)

for key, size, window, it in product(sentences.keys(), SIZES, WINDOWS, ITERS):
    fp = '{}-size{}-window{}-iter{}.kv'.format(key, size, window, it)
    print('Generating KeyedVectors model...\n{}\n'.format(repr(fp)))
    w2v = Word2Vec(
        sentences=sentences[key],
        size=size,
        window=window,
        min_count=0,
        workers=4,
        sg=0,
        hs=0,
        iter=it,
        sorted_vocab=1,
    )
    w2v.wv.save(os.path.join(OUT, fp))
