#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes, Sérgio Matos

https://github.com/ruiantunes/biocreative-vi-track-5-chemprot


Usage
-----                        #1
$ python3    confusion.py    pred.tsv

Arguments:
1. File path to Predicted relations. Mandatory.


Summary
-------
The goal of this script is to calculate a confusion matrix given the
Predicted relations of a ChemProt dataset group ('training',
'development', or 'test_gs'). The Gold standard relations are assumed to
be available using the "support.py" script. Make sure you have
downloaded the "./data/" directory.

An extensive TSV file is written containing a table (confusion matrix)
and detailed information about the relations of the confusion matrix.
This is to facilitate a rapid visualization of which instances the model
fails to predict.

The dataset group being evaluated is automatically found by checking the
PMIDs from the file with the Predicted relations. If there are PMIDs
from multiple dataset groups, the program is aborted, since the
Predicted relations are expected to be from only one dataset group
('training', 'development', or 'test_gs'). Unknown PMIDs are ignored.

The file name of the output file is automatically defined containing the
current date followed by "-confusion-logs.tsv". (However, overwrite is
not permitted, causing the program to stop when there is already a file
with the same name. This should be very unlikely.)

The specific dataset group is loaded to print additional information
about the confusion matrix: the entities, the sentence, the shortest
dependency path, the number of entities and the (Gold standard)
relations in the sentence.

It is assumed the Predicted relations contain, at maximum, one CPR
group for each relation triple (PMID, Arg1, Arg2).

Read carefully the comments through the script to have a clear
understanding.


List of abbreviations
---------------------
ChemProt ... chemical-protein
CPR ........ chemical-protein relation
FN ......... false negative
FP ......... false positive
ID ......... identifier
MEDLINE .... MEDLARS Online
MEDLARS .... Medical Literature Analysis and Retrieval System
NLP ........ natural language processing
PMID ....... PubMed ID
PubMed ..... Public MEDLINE
TP ......... true positive
TSV ........ tab-separated values


References
----------
[1] https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative
[2] https://developers.google.com/machine-learning/glossary/#confusion_matrix
[3] https://en.wikipedia.org/wiki/Confusion_matrix
[4] https://en.wikipedia.org/wiki/Sensitivity_and_specificity

"""


# Parse input arguments.
import os
import sys


args = sys.argv[1:]
n = len(args)
if (n == 1):
    pred_filepath = args[0]
else:
    print(__doc__, end='')
    exit()


from datetime import datetime
import numpy as np

from support import DATA
from support import CPR_EVAL_GROUPS
from support import LABEL2INDEX

from support import f1_score
from support import precision
from support import recall

from support import to_int
from support import its2tns

from support import get_pmids
from support import get_pmid2text
from support import get_pmid2entities
from support import get_pmid2gold_standard

from support import get_pmid2sentences
from support import get_pmid2sentence_entities
from support import load_data_from_zips


def load_relations(filepath):
    # Read file.
    f = open(filepath, mode='r', encoding='utf-8')
    lines = f.read().splitlines()
    f.close()
    # Save relations into a dictionary.
    relations = dict()
    n_relations = 0
    pmids = set()
    for line in lines:
        pmid, cpr, arg1, arg2 = line.split('\t')
        # Sanity check. Are the files consistent?
        assert pmid == pmid.strip()
        assert cpr == cpr.strip()
        assert arg1 == arg1.strip()
        assert arg2 == arg2.strip()
        pmids.add(pmid)
        r = (pmid, arg1, arg2)
        if r not in relations:
            relations[r] = set()
        assert cpr not in relations[r], (
            '\nThe following line, containing a repeated relation, '
            'caused this program to stop.\n{}'.format(repr(line))
        )
        relations[r].add(cpr)
        n_relations += 1
    return relations, n_relations, pmids


# Define the TSV output filepath.
out_filepath = '{}-confusion-logs.tsv'.format(
    datetime.now().strftime('%Y-%m-%d-%H%M%S-%f')
)

if os.path.exists(out_filepath):
    print('{} already exists. Program aborted.'.format(
        repr(out_filepath))
    )
    exit(1)

# Load Predicted relations.
predictions, n_predictions, predictions_pmids = \
    load_relations(pred_filepath)

# The calculus of the confusion matrix is assuming that the system only
# predicts, at maximum, a CPR for each relation. Because of that, it is
# wise to assert that there are not more than one CPR for each relation.
for r, cprs in predictions.items():
    assert len(cprs) == 1, (
        '\nThe following Predicted relation does not contain exactly '
        'one CPR.\n{} {}'.format(repr(r), repr(cprs))
    )

# Dataset groups.
groups = ['training', 'development', 'test_gs']

# All known PMIDs.
known_pmids = set()
for group in groups:
    pmids = get_pmids(group)
    known_pmids = known_pmids.union(pmids)

# Remove unknown PMIDs from the Predicted relations. This is useful
# because the official test set contained additional PMIDs just to avoid
# manual annotation. And these should not be included for evaluation to
# not add False Positives.

filtered_predictions = dict()
filtered_n_predictions = 0
filtered_predictions_pmids = set()
unknown_pmids = set()
n_unknown_relations = 0
for rel, cprs in predictions.items():
    pmid, arg1, arg2 = rel
    if pmid in known_pmids:
        filtered_predictions[rel] = cprs
        filtered_n_predictions += 1
        filtered_predictions_pmids.add(pmid)
    else:
        unknown_pmids.add(pmid)
        n_unknown_relations += 1

predictions = filtered_predictions
n_predictions = filtered_n_predictions
predictions_pmids = filtered_predictions_pmids

# Find what is the correct dataset group.
group_found = False
for group in groups[::-1]:
    is_this_group = True
    group_pmids = get_pmids(group)
    # Verify that the PMIDs from the Predicted relations belong to the
    # PMIDs of this dataset group.
    for pmid in predictions_pmids:
        if pmid not in group_pmids:
            is_this_group = False
            break
    if is_this_group:
        group_found = group
        break

assert group_found, (
    'Dataset group not found. '
    'Probably, the input file (Predicted relations) is inconsistent.'
)

group = group_found

# Each relation triplet (PMID, Arg1, Arg2) will contain additional
# information for easier interpretation of the results:
# PMID, Arg1, Arg2,
# Entity1, Entity2,
# Sentence number, Sentence indexes, Sentence,
# Shortest dependency path (words),
# Shortest dependency path (part-of-speeches),
# Shortest dependency path (dependencies),
# Number of entities in the sentence,
# Number of Gold standard (evaluated) relations in the sentence.

pmids = get_pmids(group)
pmid2text = get_pmid2text(group)
pmid2entities = get_pmid2entities(group)
pmid2gold_standard = get_pmid2gold_standard(group)

# Get Gold standard relations in the same format of the Predicted
# relations. As loaded by the "load_relations" function.
gold_standard = dict()
n_gold_standard = 0
gold_standard_pmids = set()
for pmid, gs in pmid2gold_standard.items():
    for rel in gs:
        cpr = rel['cpr']
        # Sanity check. It should only have evaluated (Gold standard)
        # relations.
        assert cpr in CPR_EVAL_GROUPS, (
            'Unexpected CPR group is not Gold standard.\n'
            '{} {}'.format(repr(pmid), repr(cpr))
        )
        arg1 = 'Arg1:' + rel['arg1']['tn']
        arg2 = 'Arg2:' + rel['arg2']['tn']
        gold_standard_pmids.add(pmid)
        r = (pmid, arg1, arg2)
        if r not in gold_standard:
            gold_standard[r] = set()
        assert cpr not in gold_standard[r], (
            '\nThe following repeated Gold standard relation '
            'caused this program to stop.\n{}'.format(repr(rel))
        )
        gold_standard[r].add(cpr)
        n_gold_standard += 1

pmid2sentences = get_pmid2sentences(group)
pmid2sentence_entities = get_pmid2sentence_entities(group)

zips = [
    os.path.join(
        DATA,
        'chemprot_{}'.format(group),
        'support',
        'processed_corpus.zip',
    )
]

data = load_data_from_zips(zips=zips, label2index=LABEL2INDEX)

# Calculate the number of Gold standard relations for each sentence
# (from the official dataset group).
# Note: The "pmid2sentence_relations" variable contain all the PMIDs.
# However, there are PMIDs that do not have any (evaluated) relation.
pmid2sentence_relations = dict()
for pmid in pmids:
    pmid2sentence_relations[pmid] = dict()
    n_sentences = len(pmid2sentences[pmid])
    pmid2sentence_relations[pmid] = [list() for _ in range(n_sentences)]

for i in data['info']:
    cpr = i['cpr']
    # Only add gold standard (evaluated) relations.
    if cpr in CPR_EVAL_GROUPS:
        pmid = i['pmid']
        sn = i['sn']
        a1 = i['a1']
        a2 = i['a2']
        k = (pmid, a1, a2)
        assert k in gold_standard, (
            'Unexpected relation (from the ZIP file) is not in the '
            'Gold standard relations.\n{}'.format(k)
        )
        assert cpr in gold_standard[k], (
            'Unexpected relation (from the ZIP file) does not contain '
            'a specific CPR.\n{} not in {}'.format(repr(cpr), k)
        )
        r = (a1, a2, cpr)
        assert r not in pmid2sentence_relations[pmid][sn], (
            '\nUnexpected repeated relation from the ZIP file.'
            '\nPMID: {}'
            '\n{}'.format(pmid, repr(r))
        )
        pmid2sentence_relations[pmid][sn].append(r)

# Get all the relevant information for each relation triplet
# (PMID, Arg1, Arg2).
info = dict()

for i, d in zip(data['info'], data['data']):
    pmid = i['pmid']
    arg1 = i['a1']
    arg2 = i['a2']
    #
    text = pmid2text[pmid]
    entities = pmid2entities[pmid]
    sentences = pmid2sentences[pmid]
    sentence_entities = pmid2sentence_entities[pmid]
    sentence_relations = pmid2sentence_relations[pmid]
    #
    its = (arg1, arg2)
    tn1, tn2 = its2tns((arg1, arg2))
    e1 = entities[to_int(tn1)-1]
    e2 = entities[to_int(tn2)-1]
    r = (pmid, arg1, arg2)
    # Sentence number.
    sn = i['sn']
    # Sentence indexes.
    si = sentences[sn]
    # Sentence.
    s = text[si[0]:si[1]]
    # Shortest dependency path (words).
    sdp_wrd = ' '.join(d[0][0])
    # Shortest dependency path (part-of-speeches).
    sdp_pos = ' '.join(d[0][1])
    # Shortest dependency path (dependencies).
    sdp_dep = ' '.join(d[0][2])
    # Number of entities in the sentence.
    ne = len(sentence_entities[sn])
    # Number of relations in the sentence.
    nr = len(sentence_relations[sn])
    if ((r in gold_standard) or (r in predictions)) and (r not in info):
        info[r] = dict()
        info[r]['e1'] = e1
        info[r]['e2'] = e2
        info[r]['sn'] = sn
        info[r]['si'] = si
        info[r]['s'] = s
        info[r]['sdp_wrd'] = sdp_wrd
        info[r]['sdp_pos'] = sdp_pos
        info[r]['sdp_dep'] = sdp_dep
        info[r]['ne'] = ne
        info[r]['nr'] = nr

# Find the maximum number of entities and relations in a sentence. This
# is necessary to create the diagrams (tables) with the number of
# entities (per sentence) as being the X-axis, and the number of
# relations (per sentence) as being the Y-axis. Three main values are
# calculated: the number of True Positives (TPs), the number of False
# Positives (FPs), and the number of False Negatives (FNs). These are
# used to calculate other values such as Precision, Recall, and
# F1-score.
max_ne = 0
max_nr = 0
for k, v in info.items():
    max_ne = max(max_ne, v['ne'])
    max_nr = max(max_nr, v['nr'])

# Pretty table.
TABLE = """
         \t        \tGold_standard
         \t        \tNegative\tCPR:3\tCPR:4\tCPR:5\tCPR:6\tCPR:9\tSum
Predicted\tNegative\t        \txxxxx\txxxxx\txxxxx\txxxxx\txxxxx\txxxxx
         \tCPR:3   \txxxxx   \txxxxx\txxxxx\txxxxx\txxxxx\txxxxx\txxxxx
         \tCPR:4   \txxxxx   \txxxxx\txxxxx\txxxxx\txxxxx\txxxxx\txxxxx
         \tCPR:5   \txxxxx   \txxxxx\txxxxx\txxxxx\txxxxx\txxxxx\txxxxx
         \tCPR:6   \txxxxx   \txxxxx\txxxxx\txxxxx\txxxxx\txxxxx\txxxxx
         \tCPR:9   \txxxxx   \txxxxx\txxxxx\txxxxx\txxxxx\txxxxx\txxxxx
         \tSum     \txxxxx   \txxxxx\txxxxx\txxxxx\txxxxx\txxxxx
""".replace(' ', '').replace('_', ' ').replace('xxxxx', '{:d}')[1:]

# Debug table.
#print(TABLE.expandtabs(tabsize=12))

INDEX2LABEL = {
    #0: 'CPR:0',
    0: 'Negative',
    1: 'CPR:3',
    2: 'CPR:4',
    3: 'CPR:5',
    4: 'CPR:6',
    5: 'CPR:9',
}

# The output file will contain the relations for each case of the
# confusion matrix. There as 35 distinct cases. The case when the
# Predicted is Negative and the Gold standard is Negative, is ignored,
# because the provided file (by the organizers) with the Gold standard
# relations do not contain that information. That is, one of the reasons
# why the evaluation metric is F1-score, and not for example, accuracy.
# There is no accurate information of how many Negative relations exist.
# One has to estimate it using their Natural Language Processing (NLP)
# systems, which can vary because of using different methods. Sentence
# splitting changes everything. Also, are we considering cross-sentence
# relations, cross-paragraph relations, or only relations in the same
# sentence? All these choices change the number of total relations, thus
# altering the "Gold standard" number of Negative relations. Because of
# these reasons, the number of correct Predicted relations as Negative
# was not investigated. An exhaustive list containing every possible
# case of the confusion matrix follows.
#  1. [0][1] Predicted is Negative, and Gold standard is    CPR:3.
#  2. [0][2] Predicted is Negative, and Gold standard is    CPR:4.
#  3. [0][3] Predicted is Negative, and Gold standard is    CPR:5.
#  4. [0][4] Predicted is Negative, and Gold standard is    CPR:6.
#  5. [0][5] Predicted is Negative, and Gold standard is    CPR:9.
#  6. [1][0] Predicted is    CPR:3, and Gold standard is Negative.
#  7. [1][1] Predicted is    CPR:3, and Gold standard is    CPR:3.
#  8. [1][2] Predicted is    CPR:3, and Gold standard is    CPR:4.
#  9. [1][3] Predicted is    CPR:3, and Gold standard is    CPR:5.
# 10. [1][4] Predicted is    CPR:3, and Gold standard is    CPR:6.
# 11. [1][5] Predicted is    CPR:3, and Gold standard is    CPR:9.
# 12. [2][0] Predicted is    CPR:4, and Gold standard is Negative.
# 13. [2][1] Predicted is    CPR:4, and Gold standard is    CPR:3.
# 14. [2][2] Predicted is    CPR:4, and Gold standard is    CPR:4.
# 15. [2][3] Predicted is    CPR:4, and Gold standard is    CPR:5.
# 16. [2][4] Predicted is    CPR:4, and Gold standard is    CPR:6.
# 17. [2][5] Predicted is    CPR:4, and Gold standard is    CPR:9.
# 18. [3][0] Predicted is    CPR:5, and Gold standard is Negative.
# 19. [3][1] Predicted is    CPR:5, and Gold standard is    CPR:3.
# 20. [3][2] Predicted is    CPR:5, and Gold standard is    CPR:4.
# 21. [3][3] Predicted is    CPR:5, and Gold standard is    CPR:5.
# 22. [3][4] Predicted is    CPR:5, and Gold standard is    CPR:6.
# 23. [3][5] Predicted is    CPR:5, and Gold standard is    CPR:9.
# 24. [4][0] Predicted is    CPR:6, and Gold standard is Negative.
# 25. [4][1] Predicted is    CPR:6, and Gold standard is    CPR:3.
# 26. [4][2] Predicted is    CPR:6, and Gold standard is    CPR:4.
# 27. [4][3] Predicted is    CPR:6, and Gold standard is    CPR:5.
# 28. [4][4] Predicted is    CPR:6, and Gold standard is    CPR:6.
# 29. [4][5] Predicted is    CPR:6, and Gold standard is    CPR:9.
# 30. [5][0] Predicted is    CPR:9, and Gold standard is Negative.
# 31. [5][1] Predicted is    CPR:9, and Gold standard is    CPR:3.
# 32. [5][2] Predicted is    CPR:9, and Gold standard is    CPR:4.
# 33. [5][3] Predicted is    CPR:9, and Gold standard is    CPR:5.
# 34. [5][4] Predicted is    CPR:9, and Gold standard is    CPR:6.
# 35. [5][5] Predicted is    CPR:9, and Gold standard is    CPR:9.
# Attention: Each Gold standard relation can contain multiple relation
# chemical-protein relation (CPR) groups. However, for simplicity we
# only considered one Predicted CPR group for each relation triplet
# (PMID, Arg1, Arg2).

# ATTENTION! Read this to have a clear understanding of the produced
# confusion matrix. In order to correctly extract the same number of
# TPs, FNs, and FPs (as in the official evaluation), it is necessary to
# define some guidelines (assumptions we previously made when
# implementing our system). First, it is necessary to have in mind that
# our system only predicts one CPR for each chemical-protein pair.
# Second, when giving the machine learning model examples to train we
# only gave one CPR for each chemical-protein, even if that
# chemical-protein pair contained multiple CPR groups. This was followed
# to not teach the machine learning contradictory examples. The CPR we
# decided to always give to the machine learning model was, by default,
# the lower CPR (the priority order is CPR:3 < CPR:4 < CPR:5 < CPR:6 <
# CPR:9). This is explained with more detail in the documentation of the
# "entity_pair_in_relations" function declared in the "support.py" file.
# Following these assumptions, we set the following rules to calculate
# our confusion matrix:
# 1. When a relation (PMID, Arg1, Arg2) has, at maximum, a CPR group in
#    the Gold standard, then "there is no problem". Some examples
#    illustrate this better.
#    1.1. Gold standard is Negative and Predicted is Negative.
#         In this case, we "do nothing". That is, we do not know a
#         priori that this is even a relation.
#    1.2. Gold standard is Negative and Predicted is CPR:3.
#         In this case, it is a False Positive.
#    1.3. Gold standard is CPR:3 and Predicted is Negative.
#         In this case, it is a False Negative.
#    1.4. Gold standard is CPR:3 and Predicted is CPR:3.
#         In this case, it is a True Positive.
#    1.5. Gold standard is CPR:3 and Predicted is CPR:4.
#         The problems arise in these "special" cases. How should we
#         calculate? Is there a False Negative (the system failed to
#         predict CPR:3 as a relation, and a False Positive (the system
#         failed to predict CPR:4 as Negative)? Following the official
#         evaluation script, the answer is Yes. The results obtained by
#         the official evaluation script, indicate that the evaluation
#         system just checks if the relation is predicted or not. It
#         simply calculates the number of True Positives, False
#         Negatives and False Positives.
# 2. When a relation (PMID, Arg1, Arg2) has more than one CPR group in
#    the Gold standard, then "there is" a problem. Some examples
#    illustrate this better.
#    2.1. Gold standard is CPR:3 and CPR:4. Predicted is Negative.
#         Should we count twice as False Negative? According to the
#         official evaluation script, Yes. And it makes sense, there are
#         indeed two False Negatives.
#    2.2. Gold standard is CPR:3 and CPR:4. Predicted is CPR:3.
#         And, what about now? Is there a True Positive, and a False
#         Negative? According to the official evaluation script, Yes.
#    2.3. Gold standard is CPR:3 and CPR:4. Predicted is CPR:4.
#         Again, following the official evaluation script there is a
#         False Negative (CPR:3) and a True Positive (CPR:4).
#    2.4. Gold standard is CPR:3 and CPR:4. Predicted is CPR:5.
#         There are two False Negatives, and one False Positive.
# 3. With these several possibilities of Gold standard and Predicted
#    values this is how we chose to proceed:
#    3.1. In the "simplest" scenario where the Gold standard relation
#         contains, at maximum, a CPR group, we cannot just check if the
#         Predicted value corresponds to a True Positive, False
#         Negative, or False Positive. This would be incomplete (in some
#         cases) causing a wrong sum. Detailed explanation using
#         examples follows (sorry for the redundancy, this is for the
#         sake of clarification).
#         3.1.1. Gold standard is Negative and Predicted is Negative.
#                This is just a True Negative.
#         3.1.2. Gold standard is Negative and Predicted is CPR:3.
#                This is a False Positive.
#         3.1.3. Gold standard is CPR:3 and Predicted is Negative.
#                This is a False Negative.
#         3.1.4. Gold standard is CPR:3 and Predicted is CPR:3.
#                This is a True Positive.
#         3.1.5. Gold standard is CPR:3 and Predicted is CPR:4.
#                To keep consistency with the official evaluation script
#                we chose to add two values in the confusion matrix when
#                this "special" case occurs. In the CPR:4 row
#                (Predicted value), we add a value of one in the
#                Negative and CPR:3 columns (Gold standard values). The
#                value added in the Negative column means that this is a
#                False Positive (which is). And the second value added
#                in the CPR:3 column indicates that this is a False
#                Negative (which is) relation that should be predicted
#                as CPR:3. In my opinion this is a bit confusing, but it
#                is the best near-to-correct way of doing it. Curiously,
#                this is considered simultaneously a False Positive and
#                a False Negative.
#    3.2. In the other scenario where the Gold standard relation
#         contains two or more CPR groups, we proceed as detailed above.
#         3.2.1. If there are no Predicted CPR groups. In the Negative
#                row (Predicted) we add a value of one in the
#                corresponding Gold standard columns (CPR groups).
#         3.2.2. If the Predicted CPR group is not in the Gold standard
#                CPR groups, then we add a single False Positive (in the
#                respective row, and in the Negative column), and
#                add the False Negatives corresponding to the Gold
#                standard CPR groups. That is, in the respective row
#                (predicted value), we add a value of one in the CPR
#                groups (columns) that were not predicted.
#         3.2.3. If the Predicted CPR group is in the Gold standard CPR
#                groups, then we add a single True Positive, and the
#                remaining False Negatives in the respective columns of
#                the same row (predicted value).

# Results (confusion matrix).
# First key (row): Predicted.
# Second key (column): Gold standard.
# Note that the last row and and the last column is the total sum.
results = np.zeros((7, 7), dtype='int64')

# Three main values (TPs, FPs, and FNs) with respect to the number of
# entities (per sentence) and the number of relations (per sentence).
ne_nr = [
    [
        {'tp': 0, 'fp': 0, 'fn': 0}
        for _ in range(max_nr + 1)
    ]
    for _ in range(max_ne + 1)
]

# List of relations in the confusion matrix. To be saved into the output
# file.
relations = [[list() for col in range(6)] for row in range(6)]

for p_rel, p_cprs in predictions.items():
    # Detailed information about the relation.
    info_rel = p_rel + (
        info[p_rel]['e1'],
        info[p_rel]['e2'],
        info[p_rel]['sn'],
        info[p_rel]['si'],
        info[p_rel]['s'],
        info[p_rel]['sdp_wrd'],
        info[p_rel]['sdp_pos'],
        info[p_rel]['sdp_dep'],
        info[p_rel]['ne'],
        info[p_rel]['nr'],
    )
    # Number of entities in the sentence.
    ne = info[p_rel]['ne']
    # Number of relations in the sentence.
    nr = info[p_rel]['nr']
    # It is expected that the predictions have only one Predicted
    # relation group (CPR) for each chemical-protein pair. Just one more
    # check.
    assert len(p_cprs) == 1
    p_cpr = next(iter(p_cprs))
    p_idx = LABEL2INDEX[p_cpr]
    if p_rel in gold_standard:
        g_cprs = gold_standard[p_rel]
        g_idxs = [LABEL2INDEX[g_cpr] for g_cpr in g_cprs]
        if p_idx in g_idxs:
            # Add False Negatives (and a True Positive somewhere).
            for g_idx in g_idxs:
                results[p_idx][g_idx] += 1
                relations[p_idx][g_idx].append(info_rel)
                if p_idx == g_idx:
                    ne_nr[ne][nr]['tp'] += 1
                else:
                    ne_nr[ne][nr]['fn'] += 1
        else:
            # Add False Negatives.
            for g_idx in g_idxs:
                results[p_idx][g_idx] += 1
                relations[p_idx][g_idx].append(info_rel)
                ne_nr[ne][nr]['fn'] += 1
            # Add False Positive.
            results[p_idx][0] += 1
            relations[p_idx][0].append(info_rel)
            ne_nr[ne][nr]['fp'] += 1
    else:
        # Add False Positive.
        results[p_idx][0] += 1
        relations[p_idx][0].append(info_rel)
        ne_nr[ne][nr]['fp'] += 1

for g_rel, g_cprs in gold_standard.items():
    # Detailed information about the relation.
    info_rel = g_rel + (
        info[g_rel]['e1'],
        info[g_rel]['e2'],
        info[g_rel]['sn'],
        info[g_rel]['si'],
        info[g_rel]['s'],
        info[g_rel]['sdp_wrd'],
        info[g_rel]['sdp_pos'],
        info[g_rel]['sdp_dep'],
        info[g_rel]['ne'],
        info[g_rel]['nr'],
    )
    # Number of entities in the sentence.
    ne = info[g_rel]['ne']
    # Number of relations in the sentence.
    nr = info[g_rel]['nr']
    # Assert that there is at least one CPR per relation.
    assert len(g_cprs) >= 1
    g_idxs = [LABEL2INDEX[g_cpr] for g_cpr in g_cprs]
    if g_rel not in predictions:
        for g_idx in g_idxs:
            # Add False Negatives.
            results[0][g_idx] += 1
            relations[0][g_idx].append(info_rel)
            ne_nr[ne][nr]['fn'] += 1

# Calculate the following remaining values.
# TP + FP
# TP + FN
# TP + FP + FN
# Precision
# Recall
# F1-score
# Remember that the X-axis is the number of entities per sentence, and
# the Y-axis is the number of relations per sentence.
for i in range(max_ne + 1):
    for j in range(max_nr + 1):
        tp = ne_nr[i][j]['tp']
        fp = ne_nr[i][j]['fp']
        fn = ne_nr[i][j]['fn']
        ne_nr[i][j]['tp+fp'] = tp + fp
        ne_nr[i][j]['tp+fn'] = tp + fn
        ne_nr[i][j]['tp+fp+fn'] = tp + fp + fn
        ne_nr[i][j]['p'] = precision(tp, fp)
        ne_nr[i][j]['r'] = recall(tp, fn)
        ne_nr[i][j]['f1'] = f1_score(tp, fp, fn)

# Sum rows and columns of the confusion matrix.
results[-1,:-1] = np.sum(results[:-1,:-1], axis=0)
results[:-1,-1] = np.sum(results[:-1,:-1], axis=1)

# Calculate TPs, FNs, and FPs.
tp = np.sum(np.diag(results))

# Lower triangle of the matrix.
lower_triangle = np.tril(results[1:-1,1:-1], k=-1)

# Upper triangle of the matrix.
upper_triangle = np.triu(results[1:-1,1:-1], k=1)

fn = (
    np.sum(results[0,:-1]) +
    np.sum(lower_triangle) +
    np.sum(upper_triangle)
)

fp = np.sum(results[:-1,0])

p = precision(tp, fp)
r = recall(tp, fn)
f = f1_score(tp, fp, fn)

# Convert results to list to print pretty table.
results_list = results.flatten()[1:-1]
RESULTS = TABLE.format(*results_list)

# Print the most relevant information in the terminal.
print(
    'Ignored {:d} relations from {:d} unknown PMIDs.\n'
    ''.format(n_unknown_relations, len(unknown_pmids))
)

print(RESULTS.expandtabs(10))

print('Total Gold standard relations: {:d}'.format(n_gold_standard))
print('Total Predicted relations:     {:d}\n'.format(n_predictions))

print('TP: {:4d}'.format(tp))
print('FN: {:4d}'.format(fn))
print('FP: {:4d}\n'.format(fp))

print('Precision: {:.8f}'.format(p))
print('Recall:    {:.8f}'.format(r))
print('F1-score:  {:.8f}'.format(f))

# Write the output file.
# Another (double) check.
if os.path.exists(out_filepath):
    print('{} already exists. Program aborted.'.format(
        repr(out_filepath))
    )
    exit(1)

fout = open(out_filepath, mode='w', buffering=1, encoding='utf-8')
# Information about ignored PMIDs and respective relations.
_ = fout.write(
    'Ignored {:d} relations from {:d} unknown PMIDs.\n\n'
    ''.format(n_unknown_relations, len(unknown_pmids))
)
# Write confusion matrix.
_ = fout.write(RESULTS + '\n')
# Write TPs, FNs, FPs, precision, recall, f1-score.
_ = fout.write(
    'Total Gold standard relations\t{:d}\n'.format(n_gold_standard)
)
_ = fout.write(
    'Total Predicted relations\t{:d}\n\n'.format(n_predictions)
)
_ = fout.write('True Positives\t{:d}\n'.format(tp))
_ = fout.write('False Negatives\t{:d}\n'.format(fn))
_ = fout.write('False Positives\t{:d}\n\n'.format(fp))
_ = fout.write('Precision\t{:.8f}\n'.format(p))
_ = fout.write('Recall\t{:.8f}\n'.format(r))
_ = fout.write('F1-score\t{:.8f}\n\n'.format(f))

# Write special tables. X-axis is the number of entities per sentence.
# Y-axis is the number of relations per sentence.
heading = {
    'tp': 'Number of True Positives',
    'fp': 'Number of False Positives',
    'fn': 'Number of False Negatives',
    'tp+fp': 'Number of True Positives and False Positives',
    'tp+fn': 'Number of True Positives and False Negatives',
    'tp+fp+fn': 'Number of True Positives, False Positives, and False '
        'Negatives',
    'p': 'Precision',
    'r': 'Recall',
    'f1': 'F1-score',
}
# Special tables.
for key, title in heading.items():
    # Heading title.
    _ = fout.write('\t{}\n'.format(title))
    # X-axis.
    _ = fout.write(
        '\tNumber of Gold standard entities (per sentence)\n'
        'Number of Gold standard evaluated relations (per sentence)'
    )
    for i in range(max_ne + 1):
        _ = fout.write('\t{}'.format(i))
    # Y-axis.
    _ = fout.write('\n')
    for j in range(max_nr + 1):
        _ = fout.write('{}'.format(j))
        for i in range(max_ne + 1):
            _ = fout.write('\t{}'.format(ne_nr[i][j][key]))
        _ = fout.write('\n')
    _ = fout.write('\n')

# Write detailed relations (each case of the confusion matrix).
add_new_line = False
for i, row in enumerate(relations):
    for j, col in enumerate(row):
        if (i == 0) and (j == 0):
            continue
        if add_new_line:
            _ = fout.write('\n')
        else:
            add_new_line = True
        _ = fout.write(
            'Predicted is {}. Gold standard is {}.'
            ''.format(INDEX2LABEL[i], INDEX2LABEL[j])
        )
        if len(col) == 0:
            s = '\tNo relations.\n'
        elif len(col) == 1:
            s = '\tTotal of 1 relation.\n'
        else:
            s = '\tTotal of {:d} relations.\n'.format(len(col))
        _ = fout.write(s)
        for rel in col:
            _ = fout.write(
                '{}\n'.format('\t'.join([str(e) for e in rel]))
            )

fout.close()
