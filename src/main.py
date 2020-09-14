#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes, Sérgio Matos

https://github.com/ruiantunes/biocreative-vi-track-5-chemprot


BioCreative VI - Track 5 (CHEMPROT). Multiclass classification problem.

This script follows a supervised learning strategy using deep learning
algorithms in the BioCreative VI - Track 5 (CHEMPROT task). Its goal is
to extract Chemical-Protein (CHEMPROT) Relations (CPRs) from PubMed
abstracts (title and abstracts in English) from scientific papers
published between 2005 and 2014 (see CHEMPROT guidelines PDF:
"Annotation manual of CHEMPROT interactions between CEM and GPRO" [1]_).


References
----------
.. [1] http://www.biocreative.org/tasks/biocreative-vi/track-5/
.. [2] https://github.com/fchollet/keras/issues/2607
.. [3] https://github.com/fchollet/keras/blob/53e541f7bf55de036f4f5641bd2947b96dd8c4c3/keras/metrics.py
.. [4] https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
.. [5] https://www.kaggle.com/lystdo/lb-0-18-lstm-with-glove-and-magic-features
.. [6] https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
.. [7] https://stackoverflow.com/questions/39263002/calling-fit-multiple-times-in-keras
.. [8] https://stackoverflow.com/questions/34673396/what-does-the-standard-keras-model-output-mean-what-is-epoch-and-loss-in-keras#37979686
.. [9] https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

"""

# read script file
with open('main.py') as f:
    main = f.read()

# input arguments

# seed number
# parse command line arguments to obtain SEED integer value
import sys
import os
args = sys.argv[1:]
n = len(args)
if n >= 1:
    SEED = int(args[0])
else:
    SEED = 0

# DNN model to use (choose between 'lstm' or 'cnn')
MODEL = 'lstm'

#EXTERNAL_GROUPS = ['biogrid']
EXTERNAL_GROUPS = []

#TRAINING_GROUPS = ['training', 'development']
TRAINING_GROUPS = ['training']

#TEST_GROUPS = ['development', 'test_gs']
TEST_GROUPS = ['development']

# use Shortest Dependency Path (features)
SDP = True

# use Left/Right text chemical/gene (features)
LR = False

# use words as features
WRD = True

# use Part-of-Speech (features)
POS = False

# use dependency incoming edges (features)
DEP = False

if 1:
    # word2vec size [100, 300]
    W2V_SIZE = 100

    # word2vec window [5, 20, 50]
    W2V_WINDOW = 50

    # gensim Word2Vec model file path
    W2V_PUBMED_FPATH = os.path.join(
        'word2vec',
        'pubmed_full_umlsstop_word2vec_model_{}_{}'.format(
            W2V_SIZE, W2V_WINDOW),
    )
else:
    W2V_SIZE = 200
    W2V_WINDOW = 20
    W2V_PUBMED_FPATH = os.path.join(
        'word2vec',
        'BioWordVec_PubMed_MIMICIII_d200.vec.bin',
    )

# Pos2Vec word2vec model filepath (None to use random embeddings)
#P2V_FPATH = None
P2V_FPATH = os.path.join(
    'word2vec',
    'pos-size20-window3-iter100.kv',
)

# pos2vec size (only applicable when using random embeddings)
P2V_SIZE = 20

# Dep2Vec word2vec model filepath (None to use random embeddings)
#D2V_FPATH = None
D2V_FPATH = os.path.join(
    'word2vec',
    'dep-size20-window3-iter100.kv',
)

# dependency2vec size (only applicable when using random embeddings)
D2V_SIZE = 20

assert MODEL in ('lstm', 'cnn'), 'Invalid model!'
assert len(TRAINING_GROUPS) > 0
assert len(TEST_GROUPS) > 0
assert SDP or LR, 'SDP and LR are False!'
assert WRD or POS or DEP, 'WRD, POS, and DEP are False!'


# built-in modules (sys.builtin_module_names)
import time

# third-party modules
import datetime
from gensim.models import Word2Vec
#from keras import regularizers
#from keras import backend as K
#from keras.callbacks import EarlyStopping
#from keras.callbacks import ModelCheckpoint
#from keras.layers import Activation
#from keras.layers import AveragePooling1D
from keras.layers import Bidirectional
from keras.layers import concatenate
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
#from keras.layers import Flatten
from keras.layers import GaussianNoise
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input
from keras.layers import LSTM
#from keras.layers import MaxPooling1D
#from keras.layers.normalization import BatchNormalization
#from keras.models import load_model
from keras.models import Model
#from keras.preprocessing.text import text_to_word_sequence
#from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
#from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random as rn
#from scipy.linalg import norm
#from sklearn.metrics import make_scorer
#from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# own modules
from mfuncs import load_keyedvectors
from mfuncs import load_word2vec
from mfuncs import normalized_rand
from mfuncs import normalized_sum
from mfuncs import tokenize
from mfuncs import tokseqs2intseqs
from mfuncs import to_uncategorical
from support import CPR_EVAL_GROUPS
from support import DATA
from support import INDEX2LABEL
from support import LABEL2INDEX
from support import chemprot_eval
from support import chemprot_eval_arrays
#from support import get_pmids
from support import load_data_from_zips
from utils import create_directory
from utils import Printer

# to remove tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# to ensure determinism/reproducibility [6]
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(SEED)
rn.seed(SEED)
tf.set_random_seed(SEED)


# other constants

# the directory for saving the output files (logs, ...)
OUT = 'out'
create_directory(OUT)

# output files: logs, predictions, probabilities, best model, history
#               PNG image, main python script (this script)
FN = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S-%f'))
LOGS_FPATH = os.path.join(OUT, FN + '-logs.txt')
PREDICTIONS_FPATH = os.path.join(OUT, FN + '-predictions.tsv')
PROBABILITIES_FPATH = os.path.join(OUT, FN + '-probabilities.tsv')
BEST_MODEL_FPATH = os.path.join(OUT, FN + '-best-model.h5')
HISTORY_FPATH = os.path.join(OUT, FN + '-history.png')
MAIN_FPATH = os.path.join(OUT, FN + '-main.py')

with open(MAIN_FPATH, 'w') as f:
    _ = f.write(main)

# external datasets
EXTERNAL_ZIPS = [
    os.path.join(
        DATA,
        'chemprot_{}'.format(group),
        'support',
        'processed_corpus.zip',
    ) for group in EXTERNAL_GROUPS
]

# training datasets
TRAINING_ZIPS = [
    os.path.join(
        DATA,
        'chemprot_{}'.format(group),
        'support',
        'processed_corpus.zip',
    ) for group in TRAINING_GROUPS
]

# test datasets
TEST_ZIPS = [
    os.path.join(
        DATA,
        'chemprot_{}'.format(group),
        'support',
        'processed_corpus.zip',
    ) for group in TEST_GROUPS
]

# test gold standard relations
TEST_GS_FPATHS = [
    os.path.join(
        DATA,
        'chemprot_{}'.format(group),
        'chemprot_{}_gold_standard.tsv'.format(group),
    )
    for group in TEST_GROUPS
]

# number of unique classes
NUM_CLASSES = len(INDEX2LABEL)

# load datasets BUT ONLY for creating the dataset vocabulary (this is
# important because the TEES tokenizer produces very different results
# compared to the gensim `simple_preprocess` tokenizer, which was used
# to create the W2V PubMed-based models)
EXTERNAL_DATA = load_data_from_zips(
    zips=EXTERNAL_ZIPS,
    label2index=LABEL2INDEX,
)
TRAINING_DATA = load_data_from_zips(
    zips=TRAINING_ZIPS,
    label2index=LABEL2INDEX,
)
TEST_DATA = load_data_from_zips(
    zips=TEST_ZIPS,
    label2index=LABEL2INDEX,
)

# dataset vocabulary
DATASET_VOCABULARY = set()
data = EXTERNAL_DATA['data'] + TRAINING_DATA['data'] + TEST_DATA['data']
for features in data:
    for f in features:
        # take advantage of this dataset vocabulary building and check
        # that indeed there are not empty strings (sanity check)
        for s in f:
            assert '' not in s, 'Empty strings not expected!'
        DATASET_VOCABULARY.update(f[0])

# free memory (delete temporary dataset)
del EXTERNAL_DATA
del TRAINING_DATA
del TEST_DATA

# load w2v PubMed-based model (normalized vectors)
W2V_PUBMED = load_word2vec(W2V_PUBMED_FPATH)
key = next(iter(W2V_PUBMED))
W2V_DTYPE = W2V_PUBMED[key].dtype
W2V_PUBMED_VOCABULARY = set(W2V_PUBMED)
# w2v model (only with the non-zero-embedding words from the dataset)
W2V = dict()
# create word embeddings only from the dataset vocabulary
for word in DATASET_VOCABULARY:
    tokens = tokenize(word, W2V_PUBMED_VOCABULARY)
    if len(tokens) > 0:
        W2V[word] = normalized_sum(
            embeddings=[W2V_PUBMED[t] for t in tokens],
            size=W2V_SIZE,
            dtype=W2V_DTYPE,
        )

# free memory (delete word2vec PubMed-based model)
del W2V_PUBMED
del W2V_PUBMED_VOCABULARY

# W2V dataset-vocabulary (vocabulary only from the dataset)
W2V_VOCABULARY = set(W2V)
UNIQUE_WRD = sorted(W2V_VOCABULARY)
NUM_WRD = len(W2V_VOCABULARY)
W2V_TRAINABLE = False

# filter words (pos/dep) by the W2V vocabulary
WORDS = W2V_VOCABULARY

# load training dataset(s)
EXTERNAL_DATA = load_data_from_zips(
    zips=EXTERNAL_ZIPS,
    label2index=LABEL2INDEX,
    shuffle=True,
    random_state=SEED,
    words=WORDS,
)

# load training dataset(s)
TRAINING_DATA = load_data_from_zips(
    zips=TRAINING_ZIPS,
    label2index=LABEL2INDEX,
    shuffle=True,
    random_state=SEED,
    words=WORDS,
)

# load test dataset(s)
TEST_DATA = load_data_from_zips(
    zips=TEST_ZIPS,
    label2index=LABEL2INDEX,
    shuffle=True,
    random_state=SEED,
    words=WORDS,
)

data = EXTERNAL_DATA['data'] + TRAINING_DATA['data'] + TEST_DATA['data']

# calculate unique POS and unique dependencies from the datasets
UNIQUE_POS = set()
UNIQUE_DEP = set()
for features in data:
    for f in features:
        UNIQUE_POS.update(f[1])
        UNIQUE_DEP.update(f[2])

UNIQUE_POS = sorted(UNIQUE_POS)
NUM_POS = len(UNIQUE_POS)
UNIQUE_DEP = sorted(UNIQUE_DEP)
NUM_DEP = len(UNIQUE_DEP)

if P2V_FPATH is None:
    # use Pos2Vec word2vec random embeddings
    P2V = {
        pos: normalized_rand(P2V_SIZE, dtype=W2V_DTYPE) for pos in UNIQUE_POS
    }
    P2V_TRAINABLE = True
else:
    # use pre-trained Pos2Vec word2vec embeddings
    P2V = load_keyedvectors(P2V_FPATH)
    # find embeddings size
    key = next(iter(P2V))
    P2V_SIZE = P2V[key].size
    # discard unused words
    P2V = {w: v for w, v in P2V.items() if w in UNIQUE_POS}
    # add unknown words (full zeros: non-informative)
    for w in UNIQUE_POS:
        if w not in P2V:
            P2V[w] = np.zeros(P2V_SIZE, dtype=W2V_DTYPE)
    # pre-trained embeddings (fixed)
    P2V_TRAINABLE = False


if D2V_FPATH is None:
    # use Dep2Vec word2vec random embeddings
    D2V = {
        dep: normalized_rand(D2V_SIZE, dtype=W2V_DTYPE) for dep in UNIQUE_DEP
    }
    # attention: the empty incoming edge is represented by the `#none`
    #            tag and it was randomly initialized on purpose (maybe
    #            it is relevant to know that a word does not have
    #            incoming edges)
    D2V_TRAINABLE = True
else:
    # use pre-trained Dep2Vec word2vec embeddings
    D2V = load_keyedvectors(D2V_FPATH)
    # find embeddings size
    key = next(iter(D2V))
    D2V_SIZE = D2V[key].size
    # discard unused words
    D2V = {w: v for w, v in D2V.items() if w in UNIQUE_DEP}
    # add unknown words (full zeros: non-informative)
    for w in UNIQUE_DEP:
        if w not in D2V:
            D2V[w] = np.zeros(D2V_SIZE, dtype=W2V_DTYPE)
    # pre-trained embeddings (fixed)
    D2V_TRAINABLE = False

# wrd/pos/dep vectorizers (index 0 is used for padding)
WRD2INT = {w: i for i, w in enumerate(UNIQUE_WRD, start=1)}
POS2INT = {p: i for i, p in enumerate(UNIQUE_POS, start=1)}
DEP2INT = {d: i for i, d in enumerate(UNIQUE_DEP, start=1)}

# embedding matrixes:
# +1 is added because the index 0 is used for padding (non-informative)

# W2V embedding matrix
W2V_MATRIX = np.zeros((NUM_WRD + 1, W2V_SIZE), dtype=W2V_DTYPE)
for index, word in enumerate(UNIQUE_WRD, start=1):
    W2V_MATRIX[index] = W2V[word]

# P2V embedding matrix
P2V_MATRIX = np.zeros((NUM_POS + 1, P2V_SIZE), dtype=W2V_DTYPE)
for index, pos in enumerate(UNIQUE_POS, start=1):
    P2V_MATRIX[index] = P2V[pos]

# D2V embedding matrix
D2V_MATRIX = np.zeros((NUM_DEP + 1, D2V_SIZE), dtype=W2V_DTYPE)
for index, dep in enumerate(UNIQUE_DEP, start=1):
    D2V_MATRIX[index] = D2V[dep]

# SDP and L/R CHEM/GENE maximum lengths
if 0:
    # automatically choose maximum values (requires more processing)
    SDP_MAXLEN = -1
    LR_MAXLEN = -1
    for features in data:
        sdp, lch, rch, lge, rge = features
        SDP_MAXLEN = max(SDP_MAXLEN, len(sdp[0]))
        LR_MAXLEN = max(
            LR_MAXLEN,
            len(lch[0]),
            len(rch[0]),
            len(lge[0]),
            len(rge[0]),
        )
else:
    # manually chosen
    SDP_MAXLEN = 10
    LR_MAXLEN = 20

# to "plot" a histogram of SDP and LR lengths
# to help to manually choose values for SDP_MAXLEN and LR_MAXLEN doing
# a compromise between system performance (time of execution) and system
# quality (ability to correctly predict)
if 0:
    n = len(data)
    sdp_lens = np.zeros(shape=n, dtype='uint32')
    lch_lens = np.zeros(shape=n, dtype='uint32')
    rch_lens = np.zeros(shape=n, dtype='uint32')
    lge_lens = np.zeros(shape=n, dtype='uint32')
    rge_lens = np.zeros(shape=n, dtype='uint32')
    for i, features in enumerate(data):
        sdp, lch, rch, lge, rge = features
        sdp_lens[i] = len(sdp[0])
        lch_lens[i] = len(lch[0])
        rch_lens[i] = len(rch[0])
        lge_lens[i] = len(lge[0])
        rge_lens[i] = len(rge[0])
    lens = [sdp_lens, lch_lens, rch_lens, lge_lens, rge_lens]
    seqs = ['SDP', 'LCH', 'RCH', 'LGE', 'RGE']
    for s, l in zip(seqs, lens):
        print('{} lengths'.format(s))
        for v in set(l):
            percentage = (np.sum(l <= v) / n) * 100
            print('up to {:2d}: {:6.2f}%'.format(v, percentage))
        print()

# Keras-related constants

# batch size [16, 32, 64, 128, 256, 512]
BATCH_SIZE = 128

# patience [10, 20, 30, 50]
PATIENCE = 30

# epochs [100, 200, 300, 500, 1000]
EPOCHS = 500

# validation split [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
VALIDATION_SPLIT = 0.3


def get_features(data):
    n_features = 5
    x = [list() for i in range(n_features*3)]
    # go through all data
    for features in data:
        # sanity check: all data must have the same number of features
        assert len(features) == n_features
        # sdp, lch, rch, lge, rge
        for i, f in enumerate(features):
            # wrd, pos, dep
            for j, s in enumerate(f):
                x[i*3+j].append(s)
    return x


def convert_features(x):
    TOK2INT = [WRD2INT, POS2INT, DEP2INT]
    MAXLEN = [SDP_MAXLEN] * 3 + [LR_MAXLEN] * 12
    PADDING = ['post']*3 + ['pre']*3 + ['post']*3 + ['pre']*3 + ['post']*3
    TRUNCATING = PADDING
    n_features = 5 * 3
    assert len(x) == n_features
    y = [list() for i in range(n_features)]
    for i, tokseqs in enumerate(x):
        y[i] = tokseqs2intseqs(
            tokseqs=tokseqs,
            tok2int=TOK2INT[i%3],
            maxlen=MAXLEN[i],
            padding=PADDING[i],
            truncating=TRUNCATING[i],
        )
    return y


def build_model(
    dropout_rate=0.4,
    gaussiannoise_stddev=0.01,
    conv1d_filters=64,
    conv1d_kernel_sizes=[3, 4, 5],
    lstm_units=128,
    lstm_dropout=0.4,
    lstm_recurrent_dropout=0.4,
):
    # for reproducibility
    np.random.seed(SEED)
    rn.seed(SEED)
    tf.set_random_seed(SEED)
    # Input
    sdp_wrd_input = Input(shape=(SDP_MAXLEN,), dtype='int32')
    sdp_pos_input = Input(shape=(SDP_MAXLEN,), dtype='int32')
    sdp_dep_input = Input(shape=(SDP_MAXLEN,), dtype='int32')
    lch_wrd_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    lch_pos_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    lch_dep_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    rch_wrd_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    rch_pos_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    rch_dep_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    lge_wrd_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    lge_pos_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    lge_dep_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    rge_wrd_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    rge_pos_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    rge_dep_input = Input(shape=(LR_MAXLEN,), dtype='int32')
    # Embedding
    wrd_emb = Embedding(input_dim=NUM_WRD+1, output_dim=W2V_SIZE, weights=[W2V_MATRIX], trainable=W2V_TRAINABLE)
    pos_emb = Embedding(input_dim=NUM_POS+1, output_dim=P2V_SIZE, weights=[P2V_MATRIX], trainable=P2V_TRAINABLE)
    dep_emb = Embedding(input_dim=NUM_DEP+1, output_dim=D2V_SIZE, weights=[D2V_MATRIX], trainable=D2V_TRAINABLE)
    # convert integer indexes to embeddings
    sdp_wrd = wrd_emb(sdp_wrd_input)
    sdp_pos = pos_emb(sdp_pos_input)
    sdp_dep = dep_emb(sdp_dep_input)
    lch_wrd = wrd_emb(lch_wrd_input)
    lch_pos = pos_emb(lch_pos_input)
    lch_dep = dep_emb(lch_dep_input)
    rch_wrd = wrd_emb(rch_wrd_input)
    rch_pos = pos_emb(rch_pos_input)
    rch_dep = dep_emb(rch_dep_input)
    lge_wrd = wrd_emb(lge_wrd_input)
    lge_pos = pos_emb(lge_pos_input)
    lge_dep = dep_emb(lge_dep_input)
    rge_wrd = wrd_emb(rge_wrd_input)
    rge_pos = pos_emb(rge_pos_input)
    rge_dep = dep_emb(rge_dep_input)
    # concatenate
    if WRD and POS and DEP:
        sdp = concatenate([sdp_wrd, sdp_pos, sdp_dep])
        lr = concatenate([
            lch_wrd, lch_pos, lch_dep,
            rch_wrd, rch_pos, rch_dep,
            lge_wrd, lge_pos, lge_dep,
            rge_wrd, rge_pos, rge_dep,
        ])
    elif WRD and POS:
        sdp = concatenate([sdp_wrd, sdp_pos])
        lr = concatenate([
            lch_wrd, lch_pos,
            rch_wrd, rch_pos,
            lge_wrd, lge_pos,
            rge_wrd, rge_pos,
        ])
    elif WRD and DEP:
        sdp = concatenate([sdp_wrd, sdp_dep])
        lr = concatenate([
            lch_wrd, lch_dep,
            rch_wrd, rch_dep,
            lge_wrd, lge_dep,
            rge_wrd, rge_dep,
        ])
    elif POS and DEP:
        sdp = concatenate([sdp_pos, sdp_dep])
        lr = concatenate([
            lch_pos, lch_dep,
            rch_pos, rch_dep,
            lge_pos, lge_dep,
            rge_pos, rge_dep,
        ])
    elif WRD:
        sdp = sdp_wrd
        lr = concatenate([
            lch_wrd,
            rch_wrd,
            lge_wrd,
            rge_wrd,
        ])
    elif POS:
        sdp = sdp_pos
        lr = concatenate([
            lch_pos,
            rch_pos,
            lge_pos,
            rge_pos,
        ])
    elif DEP:
        sdp = sdp_dep
        lr = concatenate([
            lch_dep,
            rch_dep,
            lge_dep,
            rge_dep,
        ])
    else:
        assert False, 'WRD, POS, and DEP are False!'
    # GaussianNoise
    sdp = GaussianNoise(stddev=gaussiannoise_stddev)(sdp)
    lr = GaussianNoise(stddev=gaussiannoise_stddev)(lr)
    if MODEL == 'cnn':
        sdps = list()
        lrs = list()
        for kernel_size in conv1d_kernel_sizes:
            # Conv1D
            c = Conv1D(filters=conv1d_filters, kernel_size=kernel_size, activation='relu')(sdp)
            # GlobalMaxPooling1D
            g = GlobalMaxPooling1D()(c)
            sdps.append(g)
            # Conv1D
            c = Conv1D(filters=conv1d_filters, kernel_size=kernel_size, activation='relu')(lr)
            # GlobalMaxPooling1D
            g = GlobalMaxPooling1D()(c)
            lrs.append(g)
        if len(conv1d_kernel_sizes) > 1:
            # concatenate
            sdp = concatenate(sdps)
            lr = concatenate(lrs)
        else:
            sdp = sdps[0]
            lr = lrs[0]
    elif MODEL == 'lstm':
        # LSTM, Bidirectional
        sdp = Bidirectional(LSTM(units=lstm_units, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))(sdp)
        lr = Bidirectional(LSTM(units=lstm_units, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))(lr)
    else:
        assert False, 'Invalid model!'
    # concatenate
    if SDP and LR:
        merged = concatenate([sdp, lr])
    elif SDP:
        merged = sdp
    elif LR:
        merged = lr
    else:
        assert False, 'SDP and LR are False!'
    # Dropout
    merged = Dropout(rate=dropout_rate, seed=0)(merged)
    # Dense
    preds = Dense(units=NUM_CLASSES, activation='softmax')(merged)
    # Model
    model = Model(
        inputs=[
            sdp_wrd_input, sdp_pos_input, sdp_dep_input,
            lch_wrd_input, lch_pos_input, lch_dep_input,
            rch_wrd_input, rch_pos_input, rch_dep_input,
            lge_wrd_input, lge_pos_input, lge_dep_input,
            rge_wrd_input, rge_pos_input, rge_dep_input,
        ],
        outputs=preds,
    )
    # for reproducibility
    np.random.seed(SEED)
    rn.seed(SEED)
    tf.set_random_seed(SEED)
    # Model.compile
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['acc'],
    )
    return model


# main code
printer = Printer(filepath=LOGS_FPATH)
D = printer.date
P = printer.print
D('Start!')
P(
    '--------------------------------\n'
    '--- BioCreative VI -------------\n'
    '------- Track 5 (CHEMPROT) -----\n'
    '--------------------------------\n'
)
P(
    'See the Keras Model consulting the \'build_model\' function '
    'in the \'-main.py\' saved script.\n'
)
P('main input arguments\n')
P(
    '\tSEED\n'
    '\t\t{}\n'.format(SEED)
)
P(
    '\tMODEL\n'
    '\t\t{}\n'.format(MODEL)
)
P(
    '\tEXTERNAL_GROUPS\n'
    '\t\t{}\n'.format(EXTERNAL_GROUPS)
)
P(
    '\tTRAINING_GROUPS\n'
    '\t\t{}\n'.format(TRAINING_GROUPS)
)
P(
    '\tTEST_GROUPS\n'
    '\t\t{}\n'.format(TEST_GROUPS)
)
P(
    '\tSDP\n'
    '\t\t{}\n'.format(SDP)
)
P(
    '\tLR\n'
    '\t\t{}\n'.format(LR)
)
P(
    '\tWRD\n'
    '\t\t{}\n'.format(WRD)
)
P(
    '\tPOS\n'
    '\t\t{}\n'.format(POS)
)
P(
    '\tDEP\n'
    '\t\t{}\n'.format(DEP)
)
P(
    '\tW2V_SIZE\n'
    '\t\t{}\n'.format(W2V_SIZE)
)
P(
    '\tW2V_WINDOW\n'
    '\t\t{}\n'.format(W2V_WINDOW)
)
P(
    '\tW2V_PUBMED_FPATH\n'
    '\t\t{}\n'.format(W2V_PUBMED_FPATH)
)
P(
    '\tP2V_FPATH\n'
    '\t\t{}\n'.format(P2V_FPATH)
)
P(
    '\tP2V_SIZE\n'
    '\t\t{}\n'.format(P2V_SIZE)
)
P(
    '\tD2V_FPATH\n'
    '\t\t{}\n'.format(D2V_FPATH)
)
P(
    '\tD2V_SIZE\n'
    '\t\t{}\n'.format(D2V_SIZE)
)
P('output directory\n')
P(
    '\tOUT\n'
    '\t\t{}\n'.format(OUT)
)
P('output files\n')
P(
    '\tLOGS_FPATH\n'
    '\t\t{}\n'.format(LOGS_FPATH)
)
P(
    '\tPREDICTIONS_FPATH\n'
    '\t\t{}\n'.format(PREDICTIONS_FPATH)
)
P(
    '\tPROBABILITIES_FPATH\n'
    '\t\t{}\n'.format(PROBABILITIES_FPATH)
)
P(
    '\tBEST_MODEL_FPATH\n'
    '\t\t{}\n'.format(BEST_MODEL_FPATH)
)
P(
    '\tHISTORY_FPATH\n'
    '\t\t{}\n'.format(HISTORY_FPATH)
)
P(
    '\tMAIN_FPATH\n'
    '\t\t{}\n'.format(MAIN_FPATH)
)
P('other variables\n')
P(
    '\tEXTERNAL_ZIPS\n'
    '\t\t{}\n'.format(EXTERNAL_ZIPS)
)
P(
    '\tTRAINING_ZIPS\n'
    '\t\t{}\n'.format(TRAINING_ZIPS)
)
P(
    '\tTEST_ZIPS\n'
    '\t\t{}\n'.format(TEST_ZIPS)
)
P(
    '\tTEST_GS_FPATHS\n'
    '\t\t{}\n'.format(TEST_GS_FPATHS)
)
P(
    '\tNUM_CLASSES\n'
    '\t\t{}\n'.format(NUM_CLASSES)
)
P(
    '\tW2V_DTYPE\n'
    '\t\t{}\n'.format(W2V_DTYPE)
)
P(
    '\tUNIQUE_WRD[:20]\n'
    '\t\t{}\n'.format(UNIQUE_WRD[:20])
)
P(
    '\tNUM_WRD\n'
    '\t\t{}\n'.format(NUM_WRD)
)
P(
    '\tUNIQUE_POS\n'
    '\t\t{}\n'.format(UNIQUE_POS)
)
P(
    '\tNUM_POS\n'
    '\t\t{}\n'.format(NUM_POS)
)
P(
    '\tUNIQUE_DEP\n'
    '\t\t{}\n'.format(UNIQUE_DEP)
)
P(
    '\tNUM_DEP\n'
    '\t\t{}\n'.format(NUM_DEP)
)
P(
    '\tW2V_TRAINABLE\n'
    '\t\t{}\n'.format(W2V_TRAINABLE)
)
P(
    '\tP2V_TRAINABLE\n'
    '\t\t{}\n'.format(P2V_TRAINABLE)
)
P(
    '\tD2V_TRAINABLE\n'
    '\t\t{}\n'.format(D2V_TRAINABLE)
)
P(
    '\tW2V_MATRIX.shape\n'
    '\t\t{}\n'.format(W2V_MATRIX.shape)
)
P(
    '\tP2V_MATRIX.shape\n'
    '\t\t{}\n'.format(P2V_MATRIX.shape)
)
P(
    '\tD2V_MATRIX.shape\n'
    '\t\t{}\n'.format(D2V_MATRIX.shape)
)
P(
    '\tSDP_MAXLEN\n'
    '\t\t{}\n'.format(SDP_MAXLEN)
)
P(
    '\tLR_MAXLEN\n'
    '\t\t{}\n'.format(LR_MAXLEN)
)
P('keras-related constants\n')
P(
    '\tBATCH_SIZE\n'
    '\t\t{}\n'.format(BATCH_SIZE)
)
P(
    '\tPATIENCE\n'
    '\t\t{}\n'.format(PATIENCE)
)
P(
    '\tEPOCHS\n'
    '\t\t{}\n'.format(EPOCHS)
)
P(
    '\tVALIDATION_SPLIT\n'
    '\t\t{}\n'.format(VALIDATION_SPLIT)
)
P('sanity check\n')
P(
    '\tlen(EXTERNAL_DATA[\'data\'])\n'
    '\t\t{}\n'.format(len(EXTERNAL_DATA['data']))
)
P(
    '\tlen(set(EXTERNAL_DATA[\'target\']))\n'
    '\t\t{}\n'.format(len(set(EXTERNAL_DATA['target'])))
)
P(
    '\tlen(TRAINING_DATA[\'data\'])\n'
    '\t\t{}\n'.format(len(TRAINING_DATA['data']))
)
P(
    '\tlen(set(TRAINING_DATA[\'target\']))\n'
    '\t\t{}\n'.format(len(set(TRAINING_DATA['target'])))
)
P(
    '\tlen(TEST_DATA[\'data\'])\n'
    '\t\t{}\n'.format(len(TEST_DATA['data']))
)
P(
    '\tlen(set(TEST_DATA[\'target\']))\n'
    '\t\t{}\n'.format(len(set(TEST_DATA['target'])))
)

# wrd/pos/dep features (lists of lists of lists of strings)
# lists of samples:
#   each sample is a list of five features:
#     SDP, LCH, RCH, LGE, RGE
#     each feature has three kinds of information:
#       WRD, POS, DEP
#       each type of information is represented by a list of strings
external = get_features(EXTERNAL_DATA['data'])
train = get_features(TRAINING_DATA['data'])
test = get_features(TEST_DATA['data'])

# wrd/pos/dep features (lists of numpy arrays)
x_external = convert_features(external)
x_train = convert_features(train)
x_test = convert_features(test)

# y_external, y_train and y_test (labels)
y_external_int = np.array(EXTERNAL_DATA['target'], dtype='int32')
y_train_int = np.array(TRAINING_DATA['target'], dtype='int32')
y_test_int = np.array(TEST_DATA['target'], dtype='int32')

# convert integer labels to binary vectors (one-hot encoding)
y_external = to_categorical(
    y=y_external_int,
    num_classes=NUM_CLASSES,
)
y_train = to_categorical(
    y=y_train_int,
    num_classes=NUM_CLASSES,
)
y_test = to_categorical(
    y=y_test_int,
    num_classes=NUM_CLASSES,
)

P('len(x_external)')
P('\t{}\n'.format(len(x_external)))

P('len(x_train)')
P('\t{}\n'.format(len(x_train)))

P('len(x_test)')
P('\t{}\n'.format(len(x_test)))

for i, v in enumerate(x_external):
    P('x_external[{}].shape'.format(i))
    P('\t{}\n'.format(v.shape))

for i, v in enumerate(x_train):
    P('x_train[{}].shape'.format(i))
    P('\t{}\n'.format(v.shape))

for i, v in enumerate(x_test):
    P('x_test[{}].shape'.format(i))
    P('\t{}\n'.format(v.shape))

P('y_external.shape')
P('\t{}\n'.format(y_external.shape))

P('y_train.shape')
P('\t{}\n'.format(y_train.shape))

P('y_test.shape')
P('\t{}\n'.format(y_test.shape))

# number of samples to split
n = int(VALIDATION_SPLIT * len(y_train))
# split training data into [partial_trainining, validation]
x_partial_train = [v[n:] for v in x_train]
y_partial_train = y_train[n:]
y_partial_train_int = y_train_int[n:]
x_val = [v[:n] for v in x_train]
y_val = y_train[:n]
y_val_int = y_train_int[:n]

# join "external" and "partial_train" into "total_train"
x_total_train = [
    np.concatenate((x_external[i], x_partial_train[i]), axis=0)
    for i in range(len(x_external))
]
y_total_train = np.concatenate((y_external, y_partial_train), axis=0)
y_total_train_int = np.concatenate((y_external_int, y_partial_train_int),
    axis=0)

P(
    'Splitting "train" into "partial_train" and "val".\n'
    'Joining "external" and "partial_train" into "total_train".\n'
)

P('len(x_total_train)')
P('\t{}\n'.format(len(x_total_train)))

P('len(x_val)')
P('\t{}\n'.format(len(x_val)))

P('len(x_test)')
P('\t{}\n'.format(len(x_test)))

for i, v in enumerate(x_total_train):
    P('x_total_train[{}].shape'.format(i))
    P('\t{}\n'.format(v.shape))

for i, v in enumerate(x_val):
    P('x_val[{}].shape'.format(i))
    P('\t{}\n'.format(v.shape))

for i, v in enumerate(x_test):
    P('x_test[{}].shape'.format(i))
    P('\t{}\n'.format(v.shape))

P('y_total_train.shape')
P('\t{}\n'.format(y_total_train.shape))

P('y_val.shape')
P('\t{}\n'.format(y_val.shape))

P('y_test.shape')
P('\t{}\n'.format(y_test.shape))

# class weight, sample weight
class_weight = normalize(
    [
        compute_class_weight(
            class_weight='balanced',
            classes=np.array(range(NUM_CLASSES)),
            y=y_total_train_int,
        )
    ],
    norm='l1',
)[0]

P('list(class_weight)')
P('\t{}\n'.format(list(class_weight)))

sample_weight_total_train = np.array(
    [class_weight[v] for v in y_total_train_int]
)


def multiply_cpr0(predictions, factor=1.0):
    r"""
    Increase/reduce the CPR:0 probability.

    Increasing or reducing the probability of the negative class (CPR:0)
    is useful to modify the prediction threshold. This can be useful to
    regulate the number of positive predictions, changing the F-score.

    Parameters
    ----------
    predictions : numpy.ndarray (2D)
        Array with probabilities.
    factor : float
        The first column is multiplied by this coefficient.

    Returns
    -------
    predictions : numpy.ndarray (2D)
        Array with readjusted probabilities.

    Example
    -------
    >>> predictions = np.array([[0.2, 0.3, 0.5], [0.1, 0.5, 0.4]])
    >>> multiply_cpr0(predictions, factor=2)
    array([[0.33333333, 0.25      , 0.41666667],
           [0.18181818, 0.45454545, 0.36363636]])
    >>>

    """
    if len(predictions) > 0:
        copy = np.array(predictions)
        copy[:,0] *= factor
        norm_copy = normalize(X=copy, norm='l1')
        return norm_copy
    else:
        return np.array(predictions)


def find_best_cpr0_factor(y_int_true, y_pred, method=2, verbose=False):
    r"""
    Find best CPR:0 factor to multiply in order to maximize F1-score.

    This adds some overhead (long execution time), but a better
    threshold is selected to improve F1-score.

    Parameters
    ----------
    y_int_true : numpy.ndarray (1D)
        Vector with integer labels.
    y_pred : numpy.ndarray (2D)
        Matrix with probabilities.
    method : 1 or 2
        Two different methods for searching for the best factor (the one
        that maximizes the F1-score metric). The second method is much
        faster.
    verbose : bool
        To print information. Default: `False`.

    Returns
    -------
    best_factor : float
        The CPR:0 factor that maximizes F1-score.

    """
    if verbose:
        t0 = time.time()
    if method == 1:
        # method 1 (exhaustive)
        factor_step = 0.1
        p = 0
        patience = 30
        best_f1 = 0.0
        factor = 0.0
        best_factor = factor
        while (p < patience):
            current_f1 = chemprot_eval_arrays(
                y_true=y_int_true,
                y_pred=to_uncategorical(multiply_cpr0(y_pred, factor)),
            )['f-score']
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_factor = factor
                p = 0
            else:
                p += 1
            if verbose:
                s = 'factor={:.4f}, f-score={:.4f}'
                if p == 0:
                    s += ' <--- best!'
                print(s.format(factor, current_f1))
            factor += factor_step
    elif method == 2:
        # method 2 (greedy)
        start = 0
        stop = 50
        num = 4
        values = np.linspace(start=start, stop=stop, num=num)
        f1s = np.zeros(values.size)
        p = 0
        patience = 4
        best_f1 = 0.0
        while (p < patience):
            if verbose:
                print('Searching between {}.'.format(repr(values)))
            # calculate F1-scores for the 4 values
            for i, factor in enumerate(values):
                f1s[i] = chemprot_eval_arrays(
                    y_true=y_int_true,
                    y_pred=to_uncategorical(multiply_cpr0(y_pred, factor)),
                )['f-score']
            # find 1st and 2nd maximum indexes
            max2idx, max1idx = np.argsort(f1s)[-2:]
            # 1st and 2nd maximum values
            max1 = f1s[max1idx]
            max2 = f1s[max2idx]
            # the best factor
            if max1 > best_f1:
                best_f1 = max1
                best_factor = values[max1idx]
                p = 0
            else:
                p += 1
            if verbose:
                print('Best two factors: [{}, {}].'.format(values[max1idx],
                    values[max2idx]))
                print('Respective two F1-scores: [{}, {}].\n'.format(max1,
                    max2))
            # find the new range for generating the new 4 values
            diff = values[1] - values[0]
            start = values[max1idx] - diff
            stop = values[max1idx] + diff
            # generating the new 4 values
            values = np.linspace(start=start, stop=stop, num=num)
    else:
        assert False, 'Invalid method!'
    if verbose:
        print()
        print('Best factor: {}'.format(best_factor))
        print('Best F1-score: {}'.format(best_f1))
        et = time.time() - t0
        print('Elapsed time: {:.4f} (s)'.format(et))
    return best_factor


# fit model
# "successive calls to fit will incrementally train the model" [7]

# reproducibility
np.random.seed(SEED)
rn.seed(SEED)
tf.set_random_seed(SEED)

# build Keras model
model = build_model()
P('model.summary()')
model.summary(print_fn=lambda x: P('\t' + x))
P()

# initialize variables
epoch = 0
patience = 0
best_epoch = 0
best_f1_score = 0.0
HISTORY = {
    'fit': {
        'loss': [],
        'acc': [],
    },
    'training': {
        'annotations': [],
        'predictions': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'precision': [],
        'recall': [],
        'f-score': [],
    },
    'validation': {
        'annotations': [],
        'predictions': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'precision': [],
        'recall': [],
        'f-score': [],
    },
    'best_factor': [],
}
while (epoch < EPOCHS) and (patience < PATIENCE):
    # fit
    history = model.fit(
        x=x_total_train,
        y=y_total_train,
        batch_size=BATCH_SIZE,
        epochs=1,
        verbose=1,
        shuffle=True,
        sample_weight=sample_weight_total_train,
    )
    # training predictions
    y_total_train_pred = model.predict(x=x_total_train)
    # given the "total_train" data: find the best CPR:0 factor to
    # multiply (that is, the one that maximizes the F1-score)
    factor = find_best_cpr0_factor(
        y_int_true=y_total_train_int,
        y_pred=y_total_train_pred,
        method=2,
        verbose=False,
    )
    # readjust training predictions
    y_total_train_pred = multiply_cpr0(y_total_train_pred, factor)
    y_total_train_pred_int = to_uncategorical(y_total_train_pred)
    # validation predictions (and readjust)
    y_val_pred = model.predict(x=x_val)
    y_val_pred = multiply_cpr0(y_val_pred, factor)
    y_val_pred_int = to_uncategorical(y_val_pred)
    # evaluate predictions
    train_results = chemprot_eval_arrays(
        y_total_train_int,
        y_total_train_pred_int,
    )
    val_results = chemprot_eval_arrays(
        y_val_int,
        y_val_pred_int,
    )
    # current validation F1-score
    current_f1_score = val_results['f-score']
    # compare with best F1-score
    if current_f1_score >= best_f1_score:
        # update patience
        patience = 0
        # update best epoch
        best_epoch = epoch + 1
        # update best F1-score
        best_f1_score = current_f1_score
        # update the CPR:0 factor for the best F1-score
        best_factor = factor
        # save best model
        model.save_weights(BEST_MODEL_FPATH)
    else:
        patience += 1
    # update epoch
    epoch += 1
    # update history
    HISTORY['fit']['loss'].append(history.history['loss'][0])
    HISTORY['fit']['acc'].append(history.history['acc'][0])
    for k, v in train_results.items():
        HISTORY['training'][k].append(v)
    for k, v in val_results.items():
        HISTORY['validation'][k].append(v)
    HISTORY['best_factor'].append(factor)
    # print valuable information
    P(
        'epoch={:d}/{:d}, patience={:d}/{:d}, val_results={}, '
        'factor={}\n'.format(
            epoch, EPOCHS, patience, PATIENCE, val_results, factor)
    )
P('HISTORY')
P('\t{}\n'.format(HISTORY))

P(
    'Loading best model (epoch={:d}/{:d}, (val)f-score={:.4f}, '
    'factor={:.4f})...\n'.format(
        best_epoch, EPOCHS, best_f1_score, best_factor)
)

# load best model
model.load_weights(BEST_MODEL_FPATH)

# delete best model?
#os.remove(BEST_MODEL_PATH)

# predict test set (and readjust)
y_test_pred = model.predict(x=x_test)
y_test_pred = multiply_cpr0(y_test_pred, best_factor)
# integer predictions
y_test_pred_int = to_uncategorical(y_test_pred)

P(
    'ATTENTION: (sanity check) the following "on-the-fly" resuls may\n'
    '           slightly differ from the final results because the same pair\n'
    '           can have multiple annotations, and in this case we only\n'
    '           consider one annotation (a single label was attributed for\n'
    '           each sample)!\n'
)
P('chemprot_eval_arrays(y_test_int, y_test_pred_int)')
P('\t{}\n'.format(chemprot_eval_arrays(y_test_int, y_test_pred_int)))

# find CPR groups, and probabilites for each CPR group
# (useful for creating the predictions and probabilities TSV files)
relations = dict()
for i, info in enumerate(TEST_DATA['info']):
    pred = y_test_pred[i]
    pred_int = y_test_pred_int[i]
    pmid = info['pmid']
    pair = (info['a1'], info['a2'])
    if pmid not in relations:
        relations[pmid] = dict()
    if pair not in relations[pmid]:
        relations[pmid][pair] = dict()
    # prediction
    relations[pmid][pair]['cpr'] = INDEX2LABEL[pred_int]
    # probabilities
    relations[pmid][pair]['probabilities'] = [str(p) for p in pred]

# write (sorted) predictions and probabilities TSV files
with open(PREDICTIONS_FPATH, 'w') as f, open(PROBABILITIES_FPATH, 'w') as g:
    for pmid in sorted(relations):
        pairs = relations[pmid].keys()
        pairs = sorted(pairs, key=lambda x: x[1])
        pairs = sorted(pairs, key=lambda x: len(x[1]))
        pairs = sorted(pairs, key=lambda x: x[0])
        pairs = sorted(pairs, key=lambda x: len(x[0]))
        for pair in pairs:
            cpr = relations[pmid][pair]['cpr']
            probabilities = relations[pmid][pair]['probabilities']
            if cpr in CPR_EVAL_GROUPS:
                _ = f.write('\t'.join([pmid, cpr, *pair]) + '\n')
            _ = g.write('\t'.join([pmid, *pair, *probabilities]) + '\n')

# evaluation using the predictions TSV file
results = chemprot_eval(TEST_GS_FPATHS, [PREDICTIONS_FPATH])

P('chemprot_eval(TEST_GS_FPATHS, [PREDICTIONS_FPATH])')
P('\t{}\n'.format(results))

P('Total annotations: {}'.format(results['annotations']))
P('Total predictions: {}'.format(results['predictions']))
P('TP: {}'.format(results['TP']))
P('FN: {}'.format(results['FN']))
P('FP: {}'.format(results['FP']))
P('Precision: {}'.format(results['precision']))
P('Recall: {}'.format(results['recall']))
P('F-score: {}\n'.format(results['f-score']))

# save a useful PNG image with a summary of the model training
n_epochs = len(HISTORY['fit']['loss'])
epochs = np.arange(1, n_epochs+1)
loss = HISTORY['fit']['loss']
train_f_score = HISTORY['training']['f-score']
val_f_score = HISTORY['validation']['f-score']

fig, ax = plt.subplots()

_ = ax.plot(epochs, loss, 'b--.', label='Training loss')
_ = ax.plot(epochs, train_f_score, 'r--.', label='Training F-score')
_ = ax.plot(epochs, val_f_score, 'g--.', label='Validation F-score')
_ = ax.legend(loc='upper left')

_ = plt.xlabel('epochs')
_ = plt.title('Model training history')

#plt.show()
fig.set_size_inches(32, 18)
fig.savefig(HISTORY_FPATH, dpi=200)

D('End!')
