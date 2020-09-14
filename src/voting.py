#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes, Sérgio Matos

https://github.com/ruiantunes/biocreative-vi-track-5-chemprot


BioCreative VI - Track 5 (CHEMPROT). Voting classifier.

This script has the goal to merge the outputs of several classifiers
into a unique output file. The following voting scheme is proposed:
- The probabilities of the output classifiers are averaged. The label
  with the highest probability is chosen.

"""

import datetime
import numpy as np
import os

from support import CPR_EVAL_GROUPS
from support import DATA
from support import INDEX2LABEL
from support import chemprot_eval
from support import create_directory
from utils import Printer

# input arguments

# insert directory to average the probabilities
INPUT_DPATH = os.path.join(
    'out',
    'subdirectory',
)

# insert dataset group to be evaluated ['development', 'test_gs']
EVAL_GROUP = 'development'

GOLD_STANDARD_FPATH = os.path.join(
    DATA,
    'chemprot_{}'.format(EVAL_GROUP),
    'chemprot_{}_gold_standard.tsv'.format(EVAL_GROUP),
)

# string termination of the files that contain probabilities
TERMINATION = '-probabilities.tsv'

# predictions input files
PREDICTIONS2MERGE_FPATHS = [
    os.path.join(INPUT_DPATH, f)
    for f in os.listdir(INPUT_DPATH) if f.endswith(TERMINATION)
]

OUT = os.path.join(INPUT_DPATH, 'voting')
create_directory(OUT)

FN = os.path.join(
    OUT,
    datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S-%f'),
)

# prediction output file
LOGS_FPATH = FN + '-voting-logs.txt'
PREDICTIONS_FPATH = FN + '-voting-predictions.tsv'


def get_relations(fpaths):
    relations = dict()
    for fp in fpaths:
        with open(fp) as f:
            for line in f:
                pmid, a1, a2, *probs = line.strip().split('\t')
                rel = (pmid, a1, a2)
                if rel not in relations:
                    relations[rel] = dict()
                    relations[rel]['p'] = list()
                relations[rel]['p'].append([float(p) for p in probs])
    # calculate the average of the probabilities
    for rel in relations:
        relations[rel]['a'] = np.mean(relations[rel]['p'], axis=0)
        i = np.argmax(relations[rel]['a'])
        cpr = INDEX2LABEL[i]
        relations[rel]['cpr'] = cpr
    return relations


def get_predictions(relations):
    predictions = list()
    for rel in relations:
        pmid, a1, a2 = rel
        cpr = relations[rel]['cpr']
        predictions.append([pmid, cpr, a1, a2])
    predictions.sort(key=lambda x: x[3])
    predictions.sort(key=lambda x: len(x[3]))
    predictions.sort(key=lambda x: x[2])
    predictions.sort(key=lambda x: len(x[2]))
    predictions.sort(key=lambda x: x[1])
    predictions.sort(key=lambda x: x[0])
    return predictions


# get relations
relations = get_relations(PREDICTIONS2MERGE_FPATHS)

# get predictions
predictions = get_predictions(relations)

with open(PREDICTIONS_FPATH, 'w') as f:
    for p in predictions:
        if p[1] in CPR_EVAL_GROUPS:
            _ = f.write('\t'.join(p) + '\n')

# printing
printer = Printer(filepath=LOGS_FPATH)
P = printer.print

P('INPUT_DPATH')
P('\t{}\n'.format(INPUT_DPATH))
P('EVAL_GROUP')
P('\t{}\n'.format(EVAL_GROUP))
P('GOLD_STANDARD_FPATH')
P('\t{}\n'.format(GOLD_STANDARD_FPATH))
P('TERMINATION')
P('\t{}\n'.format(TERMINATION))
P('PREDICTIONS2MERGE_FPATHS')
P('\t{}\n'.format(PREDICTIONS2MERGE_FPATHS))
P('LOGS_FPATH')
P('\t{}\n'.format(LOGS_FPATH))
P('PREDICTIONS_FPATH')
P('\t{}\n'.format(PREDICTIONS_FPATH))

results = chemprot_eval([GOLD_STANDARD_FPATH], [PREDICTIONS_FPATH])

P('Total annotations: {}'.format(results['annotations']))
P('Total predictions: {}'.format(results['predictions']))
P('TP: {}'.format(results['TP']))
P('FN: {}'.format(results['FN']))
P('FP: {}'.format(results['FP']))
P('Precision: {}'.format(results['precision']))
P('Recall: {}'.format(results['recall']))
P('F-score: {}'.format(results['f-score']))
