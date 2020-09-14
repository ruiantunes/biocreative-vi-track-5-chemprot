#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes, Sérgio Matos

https://github.com/ruiantunes/biocreative-vi-track-5-chemprot


BioCreative VI - Track 5 (CHEMPROT). Support tasks.

This script provides functions to create support data for the
BioCreative VI - Track 5 (CHEMPROT). Some of these generated data are
stored in the CHEMPROT dataset root directory 'data/', while other data
are stored in the 'support/' folder of each group of the dataset
('training', 'development', and 'test_gs'):

    data/
    |-- chemprot_development/
    |   `-- support/
    |-- chemprot_test_gs/
    |   `-- support/
    `-- chemprot_training/
        `-- support/

It is required that the CHEMPROT dataset has the following tree (files
provided by the BioCreative):

    data/
    |-- chemprot_development/
    |   |-- chemprot_development_abstracts.tsv
    |   |-- chemprot_development_entities.tsv
    |   |-- chemprot_development_gold_standard.tsv
    |   `-- chemprot_development_relations.tsv
    |-- chemprot_test_gs/
    |   |-- chemprot_test_gs_abstracts.tsv
    |   |-- chemprot_test_gs_entities.tsv
    |   |-- chemprot_test_gs_gold_standard.tsv
    |   `-- chemprot_test_gs_relations.tsv
    `-- chemprot_training/
        |-- chemprot_training_abstracts.tsv
        |-- chemprot_training_entities.tsv
        |-- chemprot_training_gold_standard.tsv
        `-- chemprot_training_relations.tsv

This script provides the following options (functions):
    1.  Create 'consistency.tsv' (check CHEMPROT dataset consistency)
    2.  Create '.../support/corpus/' (required for executing the TEES
        program)
    3.  Create '.../support/sentences.tsv' (required that the TEES
        program has been executed)
    4.  Create 'statistics.tsv' (option 3 required)
    5.  Create '.../support/sentence_entities.tsv' (option 3 required)
    6.  Create '.../support/entity_pairs.tsv' (option 5 required)
    7.  Evaluate entity pairs (option 6 required)
    8.  Create '.../support/processed_corpus/' scikit-learn compatible
        processed corpus (option 6 required, approximate running time of
        1 minute in a CPU Intel i3-4160T and Memory 7895MiB)

Running each option (function) will overwrite previously created files.

The option 3 requires the TEES output XML file compressed in gzip format
'_heads.xml.gz' inside the 'tees/' folder of each group of the
dataset:

    data/
    |-- chemprot_development/
    |   `-- tees/
    |       `-- chemprot_development_heads.xml.gz
    |-- chemprot_test_gs/
    |   `-- tees/
    |       `-- chemprot_test_gs_heads.xml.gz
    `-- chemprot_training/
        `-- tees/
            `-- chemprot_training_heads.xml.gz

Nevertheless, the TEES program needs the option 2 to be previously
executed. TEES usage example (python2, java 9 is required):
    $ python2 preprocess.py \
    > -i .../data/chemprot_test_gs/support/corpus/ \
    > -o .../data/chemprot_test_gs/tees/chemprot_test_gs_heads.xml \
    > --steps LOAD,GENIA_SPLITTER,BLLIP_BIO,STANFORD_CONVERT,\
    > SPLIT_NAMES,FIND_HEADS,SAVE

Usage example of compression to gzip format (keeping input files):
    $ gzip --keep chemprot_test_gs_heads.xml

Other variables are hard-coded in the script. Please consider reading
the code for a better understanding.


Routine listings
----------------
#functions-auxiliary
to_int
to_cpr
to_tn
tns2its
its2tns
overlap
where_overlap
inside
#functions-chemprot
get_file
get_pmids
get_pmid2text
get_pmid2entities
get_pmid2relations
get_pmid2gold_standard
#functions-tees
get_pmid2tees
#functions-support
get_pmid2sentences
get_pmid2sentence_entities
get_pmid2entity_pairs
entity_pair_in_relations
get_processed_sample
load_features_from_sample
load_data_from_zips
chemprot_eval
chemprot_eval_arrays
#functions-evaluation
precision
recall
f1_score
#functions-networkx
calculate_path_weight
#functions-main
check_dataset
create_consistency
create_corpus
create_sentences
create_statistics
create_sentence_entities
create_entity_pairs
evaluate_entity_pairs
create_processed_corpus


References
----------
.. [1] http://www.biocreative.org/tasks/biocreative-vi/track-5/
.. [2] https://github.com/jbjorne/TEES/
.. [3] https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

"""

# built-in modules (sys.builtin_module_names)
from _collections import defaultdict

# third-party modules
import gzip
import networkx as nx
import os
import random
from xml.etree import ElementTree as ET
import zipfile

# own modules
from utils import create_directory
from utils import progress


CPR_GROUPS = ['CPR:{}'.format(i) for i in range(11)]
CPR_EVAL_GROUPS = ['CPR:{}'.format(i) for i in (3, 4, 5, 6, 9)]

INDEX2LABEL = {
    0: 'CPR:0',
    1: 'CPR:3',
    2: 'CPR:4',
    3: 'CPR:5',
    4: 'CPR:6',
    5: 'CPR:9',
}
LABEL2INDEX = {
    'CPR:0': 0,
    'CPR:1': 0,
    'CPR:2': 0,
    'CPR:3': 1,
    'CPR:4': 2,
    'CPR:5': 3,
    'CPR:6': 4,
    'CPR:7': 0,
    'CPR:8': 0,
    'CPR:9': 5,
    'CPR:10': 0,
}

GROUPS = ['training', 'development', 'test_gs', 'biogrid']
EXTERNAL_GROUPS = ['biogrid']
TRAINING_GROUPS = ['training', 'development']
TEST_GROUPS = ['development', 'test_gs']

DATA = 'data'


# #functions-auxiliary
def to_int(s):
    r"""
    Force integer.

    Converts the input `s` (`str` expected) to the respective `int`
    number `n`. Only the last digit characters are considered. If `s`
    is not `str` then the function tries to convert the input value to
    `int`.

    Parameters
    ----------
    s : str
        Input to convert to `int`. Only the last digit characters are
        converted.

    Returns
    -------
    n : int
        `int` number.

    Example
    -------
    >>> from support import to_int
    >>> to_int('CPR:3')
    3
    >>> to_int('T25')
    25
    >>> to_int('Arg1:T25')
    25
    >>> to_int('112')
    112
    >>> to_int(8)
    8
    >>>

    """
    if isinstance(s, str):
        i = len(s)
        while s[i-1:].isdigit():
            if i == 0:
                break
            i -= 1
        return int(s[i:])
    return int(s)


def to_cpr(cpr):
    r"""
    Converts to a Chemical-Protein Relation (CPR) group.

    Converts the input `cpr` to a `str` representing the CPR group. The
    input `cpr` can be `str` or `int`.

    Parameters
    ----------
    cpr : str, int
        CPR group (e.g. 'CPR:3', 3).

    Returns
    -------
    cpr : str
        CPR group.

    Example
    -------
    >>> from support import to_cpr
    >>> to_cpr('CPR:3')
    'CPR:3'
    >>> to_cpr(3)
    'CPR:3'
    >>> to_cpr('CPR3')
    'CPR:3'
    >>>

    """
    return 'CPR:{}'.format(to_int(cpr))


def to_tn(tn):
    r"""
    Converts to a Term Number (TN).

    Converts the input `tn` to a `str` representing the term number. The
    input `tn` can be `str` or `int`.

    Parameters
    ----------
    tn : str, int
        Term number (e.g. 'T31', 31).

    Returns
    -------
    tn : str
        Term number (identifier).

    Example
    -------
    >>> from support import to_tn
    >>> to_tn('T54')
    'T54'
    >>> to_tn(54)
    'T54'
    >>>

    """
    return 'T{}'.format(to_int(tn))


def tns2its(tns):
    r"""
    Term Numbers (TNs) to Interactor Terms (ITs).

    This function converts a `tuple` `tns` containing two term numbers
    to a `tuple` `its` containing two interactor terms. 'Arg1:' and
    'Arg2:' are concatenated to the respective term numbers (sorted in
    ascending order).

    Parameters
    ----------
    tns : tuple
        A `tuple` with two term numbers.

    Returns
    -------
    its : tuple
        A `tuple` with two interactor terms (ascending order).

    Example
    -------
    >>> from support import tns2its
    >>> tns2its((27, 15))
    ('Arg1:T15', 'Arg2:T27')
    >>> tns2its((27, 'T15'))
    ('Arg1:T15', 'Arg2:T27')
    >>> tns2its(('T27', 15))
    ('Arg1:T15', 'Arg2:T27')
    >>> tns2its(('T27', 'T15'))
    ('Arg1:T15', 'Arg2:T27')
    >>> tns2its(('Arg1:T27', 'Arg2:T15'))
    ('Arg1:T15', 'Arg2:T27')
    >>>

    """
    tns = [to_int(tns[0]), to_int(tns[1])]
    tns.sort()
    return ('Arg1:' + to_tn(tns[0]), 'Arg2:' + to_tn(tns[1]))


def its2tns(its):
    r"""
    Interactor Terms (ITs) to Term Numbers (TNs).

    This function converts a `tuple` `its` containing two interactor
    terms to a `tuple` `tns` containing two term numbers. 'Arg1' and
    'Arg2' are expected to be in ascending order, otherwise an
    `AssertionError` is raised.

    Parameters
    ----------
    its : tuple
        A `tuple` with two interactor terms (expected to be in ascending
        order).

    Returns
    -------
    tns : tuple
        A `tuple` with two term numbers.

    Raises
    ------
    AssertionError
        If the interactor terms are invalid (badly-written), or are not
        in ascending order.

    Example
    -------
    >>> from support import its2tns
    >>> its2tns(('Arg1:T15', 'Arg2:T27'))
    ('T15', 'T27')
    >>> its2tns(('Arg1:T27', 'Arg2:T15'))
    ...
    AssertionError
    >>>

    """
    assert (its[0][:6] == 'Arg1:T') and (its[1][:6] == 'Arg2:T')
    tns = its[0].split(':')[1], its[1].split(':')[1]
    assert to_int(tns[0]) < to_int(tns[1])
    return tns


def overlap(i1, i2):
    r"""
    Two 2D-intervals (in this case, sentences) overlap?

    This function returns `True` if `i1` overlaps `i2`, otherwise
    `False`.

    Parameters
    ----------
    i1 : tuple
        A `tuple` with two `int` indexes (start character offset, end
        character offset).
    i2 : tuple
        A `tuple` with two `int` indexes (start character offset, end
        character offset).

    Returns
    -------
    b : bool
        If `i1` overlaps `i2` returns `True`, otherwise `False`.

    Example
    -------
    >>> from support import overlap
    >>> overlap((0, 3), (2, 5))
    True
    >>> overlap((2, 5), (0, 3))
    True
    >>> overlap((0, 6), (2, 4))
    True
    >>> overlap((2, 4), (0, 6))
    True
    >>> overlap((0, 3), (3, 6))
    False
    >>> overlap((3, 6), (0, 3))
    False
    >>>

    """
    s1, e1 = i1
    s2, e2 = i2
    if (e1 <= s2) or (e2 <= s1):
        return False
    return True


def where_overlap(l, i):
    r"""
    What are the indexes of overlap?

    This function return the indexes of the 2D-intervals `l` that are
    overlapped by the 2D-interval `i`.

    Parameters
    ----------
    l : list
        A `list` with N elements, where each element is a 2D-interval
        `tuple` representing the start and end character offsets.
    i : tuple
        A 2D-interval `tuple` with the start and end character offsets.

    Returns
    -------
    o : list
        A `list` containing the indexes of overlap.

    Example
    -------
    >>> from support import where_overlap
    >>> l = [(0, 8), (176, 184), (242, 251)]
    >>> where_overlap(l, (8, 100))
    []
    >>> where_overlap(l, (5, 250))
    [0, 1, 2]
    >>> where_overlap(l, (183, 242))
    [1]
    >>>

    """
    return [e for e, elem in enumerate(l) if overlap(elem, i)]


def inside(i1, i2):
    r"""
    Is an interval entirely inside another?

    This function returns `True` if `i1` is inside `i2`, or `i2` is
    inside `i1`, otherwise `False`.

    Parameters
    ----------
    i1 : tuple
        A `tuple` with two `int` indexes (start character offset, end
        character offset).
    i2 : tuple
        A `tuple` with two `int` indexes (start character offset, end
        character offset).

    Returns
    -------
    b : bool
        If `i1` is inside `i2`, or `i2` is inside `i1` returns `True`,
        otherwise `False`.

    Example
    -------
    >>> from support import inside
    >>> inside((0, 3), (2, 5))
    False
    >>> inside((2, 5), (0, 3))
    False
    >>> inside((0, 6), (2, 4))
    True
    >>> inside((2, 4), (0, 6))
    True
    >>> inside((0, 3), (3, 6))
    False
    >>> inside((3, 6), (0, 3))
    False
    >>>

    """
    s1, e1 = i1
    s2, e2 = i2
    if ((s1 >= s2) and (e1 <= e2)) or ((s2 >= s1) and (e2 <= e1)):
        return True
    return False


# #functions-chemprot
def get_file(group, file):
    r"""
    Returns the content of a BioCreative/support file.

    Returns a `list` where each element is a `list` containing the
    individual elements of each line. The `file` is read from the
    respective `group` of the CHEMPROT dataset. Note: the individual
    elements are stripped since the evaluation column in the file
    'relations' has an additional unnecessary space (e.g. 'Y ').

    Parameters
    ----------
    group : str
        CHEMPROT dataset group ('training', 'development', or
        'test_gs').
    file : str
        There are two types of files: (1) the official ones provided
        from the BioCreative, and (2) the ones generated for support
        purposes.
        The BioCreative files are: 'abstracts', 'entities', 'relations',
        and 'gold_standard'.
        The generated support files are: 'sentences',
        'sentence_entities', and 'entity_pairs' (inside the 'support/'
        folder).

    Returns
    -------
    lines : list
        The `list` of lines (tab-separated values). Each line is a
        `list` with the individual elements.

    Example
    -------
    >>> from support import get_file
    >>> group = 'training'
    >>> abstracts = get_file(group, file='abstracts')
    >>> abstracts[0][0]
    '16357751'
    >>> entities = get_file(group, file='entities')
    >>> entities[0]
    ['11319232', 'T1', 'CHEMICAL', '242', '251', 'acyl-CoAs']
    >>> relations = get_file(group, file='relations')
    >>> relations[0]
    ['10047461', 'CPR:3', 'Y', 'ACTIVATOR', 'Arg1:T13', 'Arg2:T57']
    >>> gold_standard = get_file(group, file='gold_standard')
    >>> gold_standard[0]
    ['10047461', 'CPR:3', 'Arg1:T13', 'Arg2:T55']
    >>> sentences = get_file(group, file='sentences')
    >>> sentences[0]
    ['10047461', '0', '150']
    >>>

    """
    if file in ('abstracts', 'entities', 'relations', 'gold_standard'):
        fp = os.path.join(
            DATA,
            'chemprot_' + group,
            'chemprot_' + group + '_' + file + '.tsv',
        )
    else:
        fp = os.path.join(
            DATA,
            'chemprot_' + group,
            'support',
            file + '.tsv',
        )
    with open(fp, encoding='utf-8') as f:
        return [
            [element.strip() for element in tab_separated]
            for tab_separated in (line.split('\t') for line in f)
        ]


def get_pmids(group):
    r"""
    Get unique PMIDs from a specific group of the dataset.

    This function returns a `set` containing the unique PMIDS of a
    specific group of the dataset. The `get_file` function is
    internally used.

    Parameters
    ----------
    group : str
        CHEMPROT dataset group ('training', 'development', or
        'test_gs').

    Returns
    -------
    pmids : set
        A `set` containing the unique PMIDs.

    Example
    -------
    >>> from support import get_pmids
    >>> pmids_training = get_pmids('training')
    >>> pmids_development = get_pmids('development')
    >>> pmids_test_gs = get_pmids('test_gs')
    >>> len(pmids_training)
    1020
    >>> len(pmids_development)
    612
    >>> len(pmids_test_gs)
    800
    >>> groups = ['training', 'development', 'test_gs']
    >>> pmids = {v for g in groups for v in get_pmids(g)}
    >>> len(pmids)
    2432
    >>>

    """
    return {pmid for pmid, _, _ in get_file(group=group, file='abstracts')}


def get_pmid2text(group):
    r"""
    Mapping PMID to text (title and abstract separated by a newline).

    This function returns a `dict` mapping a PMID to the respective text
    (from a specific group of the dataset). The text is composed by the
    title and the abstract (after each a newline character '\n' is
    added).

    Parameters
    ----------
    group : str
        CHEMPROT dataset group ('training', 'development', or
        'test_gs').

    Returns
    -------
    pmid2text : dict
        A `dict` mapping a PMID to the respective text.

    Example
    -------
    >>> from support import get_pmid2text
    >>> pmid2text = get_pmid2text('training')
    >>> pmid = '10047461'
    >>> title, abstract = pmid2text[pmid].splitlines()
    >>> title[:30]
    'Cyclin E-cdk2 activation is as'
    >>> abstract[:30]
    'Tomudex (ZD1694) is a specific'
    >>>

    """
    return {
        pmid: title + '\n' + abstract + '\n'
        for pmid, title, abstract in get_file(group, 'abstracts')
    }


def get_pmid2entities(group):
    r"""
    Mapping PMID to entities.

    This function returns a `dict` mapping a PMID to the respective
    `list` of entities (from a specific group of the dataset). The value
    of each PMID key is a `list` where each element is a `dict`
    representing an entity. Each `dict` entity has the keys 'tn' (term
    number), 'type', 'i' (character offsets), and 'text'. The value of
    the 'i' key is a 2D-interval `tuple` with two `int` elements: the
    values of the start and end character offsets. For each PMID the
    entities are ascendingly sorted according to the respective term
    number (e.g. 'T1' < 'T2' < 'T3').

    Parameters
    ----------
    group : str
        CHEMPROT dataset group ('training', 'development', or
        'test_gs').

    Returns
    -------
    pmid2entities : dict
        A `dict` mapping a PMID to the respective sorted `list` of
        entities.

    Example
    -------
    >>> from support import get_pmid2entities
    >>> pmid2entities = get_pmid2entities('test_gs')
    >>> entities = pmid2entities['10076535']
    >>> len(entities)
    63
    >>> entities[0]
    {'tn': 'T1', 'type': 'CHEMICAL', 'i': (117, 139), 'text': 'Estramustine phosphate'}
    >>> len(pmid2entities)
    800
    >>> sum(len(e) for p, e in pmid2entities.items())
    20828
    >>>

    """
    pmid2entities = dict()
    for pmid, tn, type_, i1, i2, text in get_file(group, 'entities'):
        if pmid not in pmid2entities:
            pmid2entities[pmid] = list()
        pmid2entities[pmid].append(
            {
                'tn': tn,
                'type': type_,
                'i': (int(i1), int(i2)),
                'text': text,
            }
        )
    for pmid in pmid2entities:
        pmid2entities[pmid].sort(key=lambda x: to_int(x['tn']))
    return pmid2entities


def get_pmid2relations(group):
    r"""
    Mapping PMID to relations.

    This function returns a `dict` mapping a PMID to the respective
    `list` of relations (from a specific group of the dataset). The
    value of each PMID key is a `list` where each element is a `dict`
    representing a relation. Each `dict` relation has the keys 'cpr',
    'eval', 'cpi', 'arg1', and 'arg2'. The value of the 'arg1' and
    'arg2' keys is a `dict` entity with the keys 'tn', 'type', 'i', and
    'text' (see `get_pmid2entities`). The relations are ascendingly
    sorted according to the CPR, and the term numbers of the two
    interactor terms ('arg1' and 'arg2').

    Parameters
    ----------
    group : str
        CHEMPROT dataset group ('training', 'development', or
        'test_gs').

    Returns
    -------
    pmid2relations : dict
        A `dict` mapping a PMID to the respective `list` of relations.

    Example
    -------
    >>> from support import get_pmid2relations
    >>> pmid2relations = get_pmid2relations('development')
    >>> relations = pmid2relations['10064839']
    >>> relation = relations[0]
    >>> relation['cpr']
    'CPR:2'
    >>> relation['arg1']
    {'tn': 'T18', 'type': 'CHEMICAL', 'i': (2036, 2038), 'text': 'DF'}
    >>> relation['arg2']
    {'tn': 'T53', 'type': 'GENE-Y', 'i': (1992, 2009), 'text': 'sigma-1 receptors'}
    >>>

    """
    pmid2entities = get_pmid2entities(group)
    pmid2relations = dict()
    for pmid, cpr, eval_, cpi, arg1, arg2 in get_file(group, 'relations'):
        if pmid not in pmid2relations:
            pmid2relations[pmid] = list()
        tn1, tn2 = its2tns((arg1, arg2))
        e1 = pmid2entities[pmid][to_int(tn1)-1]
        e2 = pmid2entities[pmid][to_int(tn2)-1]
        pmid2relations[pmid].append(
            {
                'cpr': cpr,
                'eval': eval_,
                'cpi': cpi,
                'arg1': e1,
                'arg2': e2,
            }
        )
    for pmid in pmid2relations:
        pmid2relations[pmid].sort(key=lambda x: to_int(x['arg2']['tn']))
        pmid2relations[pmid].sort(key=lambda x: to_int(x['arg1']['tn']))
        pmid2relations[pmid].sort(key=lambda x: to_int(x['cpr']))
    return pmid2relations


def get_pmid2gold_standard(group):
    r"""
    Mapping PMID to gold standard relations.

    This function returns a `dict` mapping a PMID to the respective
    `list` of gold standard relations (from a specific group of the
    dataset). The value of each PMID key is a `list` where each element
    is a `dict` representing a gold standard relation. Each `dict` gold
    standard relation has the keys 'cpr', 'arg1', and 'arg2'. The value
    of the 'arg1' and 'arg2' keys is a `dict` entity with the keys 'tn',
    'type', 'text', and 'i' (see `get_pmid2entities`). The relations are
    ascendingly sorted according to the CPR, and the term numbers of the
    two interactor terms ('arg1' and 'arg2').

    Parameters
    ----------
    group : str
        CHEMPROT dataset group ('training', 'development', or
        'test_gs').

    Returns
    -------
    pmid2gs : dict
        A `dict` mapping a PMID to the respective `list` of gold
        standard relations.

    Example
    -------
    >>> from support import get_pmid2gold_standard
    >>> pmid2gs = get_pmid2gold_standard('test_gs')
    >>> gs = pmid2gs['10076535']
    >>> relation = gs[0]
    >>> relation['cpr']
    'CPR:3'
    >>> relation['arg1']
    {'tn': 'T23', 'type': 'CHEMICAL', 'i': (2285, 2301), 'text': 'hydroxyflutamide'}
    >>> relation['arg2']
    {'tn': 'T56', 'type': 'GENE-Y', 'i': (2378, 2381), 'text': 'PSA'}
    >>>

    """
    pmid2entities = get_pmid2entities(group)
    pmid2gs = dict()
    for pmid, cpr, arg1, arg2 in get_file(group, 'gold_standard'):
        if pmid not in pmid2gs:
            pmid2gs[pmid] = list()
        tn1, tn2 = its2tns((arg1, arg2))
        e1 = pmid2entities[pmid][to_int(tn1)-1]
        e2 = pmid2entities[pmid][to_int(tn2)-1]
        pmid2gs[pmid].append(
            {
                'cpr': cpr,
                'arg1': e1,
                'arg2': e2,
            }
        )
    for p in pmid2gs:
        pmid2gs[p].sort(key=lambda x: to_int(x['arg2']['tn']))
        pmid2gs[p].sort(key=lambda x: to_int(x['arg1']['tn']))
        pmid2gs[p].sort(key=lambda x: to_int(x['cpr']))
    return pmid2gs


# #functions-tees
def get_pmid2tees(group):
    r"""
    Get TEES features (sentence splitting, tokenization, ...).

    This function creates a `dict` mapping a PMID to the respective TEES
    information (sentence splitting, tokenization, POS tagging,
    dependency relations). The TEES gzip file, inside the 'tees/'
    directory is used. Some memory is used to load the TEES necessary
    information.

    Parameters
    ----------
    group : str
        CHEMPROT dataset group ('training', 'development', or
        'test_gs').

    Returns
    -------
    pmid2tees : dict
        A `dict` mapping a PMID to the respective TEES information.

    Example
    -------
    >>> from support import get_pmid2tees
    >>> pmid2tees = get_pmid2tees('development')
    >>> pmid = '10064839'
    >>> sentence_no = 0
    >>> token_id = 'bt_0'
    >>> dependency_id = 'sd_0'
    >>> pmid2tees[pmid][sentence_no]['char_offset']
    (0, 143)
    >>> pmid2tees[pmid][sentence_no]['token'][token_id]
    {'pos': 'NN', 'char_offset': (0, 7), 'head_score': 4, 'text': 'Binding'}
    >>> pmid2tees[pmid][sentence_no]['dependency'][dependency_id]
    {'t1': 'bt_15', 't2': 'bt_0', 'type': 'nsubj'}
    >>>

    """
    fp = os.path.join(
        DATA,
        'chemprot_' + group,
        'tees',
        'chemprot_' + group + '_heads.xml.gz',
    )
    pmid2tees = dict()
    with gzip.open(filename=fp, mode='rt', encoding='utf-8') as f:
        for event, elem in ET.iterparse(source=f, events=['start']):
            if elem.tag == 'document':
                pmid = elem.get('origId')
                assert pmid not in pmid2tees
                pmid2tees[pmid] = list()
            if elem.tag == 'sentence':
                char_offset = tuple(
                    int(i) for i in elem.get('charOffset').split('-')
                )
                sentence_offset = char_offset[0]
                pmid2tees[pmid] += [
                    {
                        'char_offset': char_offset,
                        'token': dict(),
                        'dependency': dict(),
                    }
                ]
            if elem.tag == 'token':
                pos = elem.get('POS')
                char_offset = tuple(
                    int(i) + sentence_offset
                    for i in elem.get('charOffset').split('-')
                )
                head_score = int(elem.get('headScore'))
                id_ = elem.get('id')
                text = elem.get('text')
                assert id_ not in pmid2tees[pmid][-1]['token']
                pmid2tees[pmid][-1]['token'][id_] = dict()
                pmid2tees[pmid][-1]['token'][id_]['pos'] = pos
                pmid2tees[pmid][-1]['token'][id_]['char_offset'] = char_offset
                pmid2tees[pmid][-1]['token'][id_]['head_score'] = head_score
                pmid2tees[pmid][-1]['token'][id_]['text'] = text
            if elem.tag == 'dependency':
                id_ = elem.get('id')
                t1 = elem.get('t1')
                t2 = elem.get('t2')
                type_ = elem.get('type')
                assert id_ not in pmid2tees[pmid][-1]['dependency']
                pmid2tees[pmid][-1]['dependency'][id_] = dict()
                pmid2tees[pmid][-1]['dependency'][id_]['t1'] = t1
                pmid2tees[pmid][-1]['dependency'][id_]['t2'] = t2
                pmid2tees[pmid][-1]['dependency'][id_]['type'] = type_
            elem.clear()
    return pmid2tees


# #functions-support
def get_pmid2sentences(group):
    r"""
    Mapping PMID to sentences.

    Returns a `dict` mapping a PMID to the respective sentence offsets.
    It only works after the `create_sentences` function is called. That
    is, it is necessary having the 'support/sentences.tsv' file.

    Parameters
    ----------
    group : str
        CHEMPROT dataset group ('training', 'development', or
        'test_gs').

    Returns
    -------
    pmid2sentences : dict
        A `dict` mapping a PMID to the respective `list` containing the
        sentence offsets.

    Example
    -------
    >>> from support import get_pmid2sentences
    >>> p2s = get_pmid2sentences('development')
    >>> pmid = '10064839'
    >>> p2s[pmid][:5]
    [(0, 143), (144, 270), (271, 478), (479, 668), (669, 833)]
    >>>

    """
    pmid2sentences = dict()
    for pmid, s, e in get_file(group, 'sentences'):
        if pmid not in pmid2sentences:
            pmid2sentences[pmid] = list()
        pmid2sentences[pmid] += [(int(s), int(e))]
    return pmid2sentences


def get_pmid2sentence_entities(group):
    r"""
    Mapping PMID to sentence entities.

    Returns a `dict` mapping a PMID to the respective `list` of
    "sentence entities" (only the term numbers are shown). Each
    "sentence entities" is a `list` with the respective term numbers.
    The `get_pmid2sentences` function is used because the number of
    sentences of each PMID has to be known (a priori). Also, it only
    works after the `create_sentence_entities` function is called. That
    is, it is necessary having the 'support/sentence_entities.tsv' file.

    Parameters
    ----------
    group : str
        CHEMPROT dataset group ('training', 'development', or
        'test_gs').

    Returns
    -------
    pmid2sentence_entities : dict
        A `dict` mapping a PMID to the respective `list` containing the
        entities of each sentence.

    Example
    -------
    >>> from support import get_pmid2sentence_entities
    >>> p2se = get_pmid2sentence_entities('development')
    >>> pmid = '10064839'
    >>> p2se[pmid][0]
    ['T49', 'T50', 'T51', 'T56']
    >>> p2se[pmid][1]
    ['T1', 'T17', 'T34']
    >>> len(p2se[pmid])
    13
    >>>

    """
    pmid2sentences = get_pmid2sentences(group)
    pmid2sentence_entities = dict()
    for pmid, i, tns in get_file(group, 'sentence_entities'):
        if pmid not in pmid2sentence_entities:
            n_sentences = len(pmid2sentences[pmid])
            pmid2sentence_entities[pmid] = [[] for i in range(n_sentences)]
        pmid2sentence_entities[pmid][int(i)] = tns.split(',')
    return pmid2sentence_entities


def get_pmid2entity_pairs(group):
    r"""
    Mapping PMID to entity pairs.

    Returns a `dict` mapping a PMID to the respective `list` of entity
    pairs (for each sentence) that are considered as possible CPR
    relations. The `get_pmid2sentences` function is used because the
    number of sentence of each PMID has to be known (a priori). Also, it
    only works after the `create_entity_pairs` function is called. That
    is, it is necessary having the 'support/entity_pairs.tsv' file.

    Parameters
    ----------
    group : str
        CHEMPROT dataset group ('training', 'development', or
        'test_gs').

    Returns
    -------
    pmid2entity_pairs : dict
        A `dict` mapping a PMID to the respective `list` of entity pairs
        (for each sentence).

    Example
    -------
    >>> from support import get_pmid2entity_pairs
    >>> p2ep = get_pmid2entity_pairs('development')
    >>> pmid = '10064839'
    >>> entity_pairs = p2ep[pmid]
    >>> sentence_no = 0
    >>> entity_pairs[sentence_no]
    [('T49', 'T56'), ('T50', 'T56'), ('T51', 'T56')]
    >>> sentence_no = 1
    >>> entity_pairs[sentence_no]
    []
    >>> len(entity_pairs)
    13
    >>>

    """
    pmid2sentences = get_pmid2sentences(group)
    pmid2entity_pairs = dict()
    for pmid, i, tn1, tn2 in get_file(group, 'entity_pairs'):
        if pmid not in pmid2entity_pairs:
            n_sentences = len(pmid2sentences[pmid])
            pmid2entity_pairs[pmid] = [[] for i in range(n_sentences)]
        pmid2entity_pairs[pmid][int(i)] += [(tn1, tn2)]
    return pmid2entity_pairs


def entity_pair_in_relations(entity_pair, relations):
    r"""
    Checks if an entity pair is in the relations `list`.

    Returns the `str` of the Chemical-Protein Relation (CPR) that the
    entity pair belongs. Note that the entity pair may have not
    any CPR, and in that case will return the 'CPR:0' `str` meaning no
    relation. Note also that there are repeated CPRs because they can
    have distinct Chemical-Protein Interactions (CPIs), e.g.
    'INDIRECT-DOWNREGULATOR' and 'INHIBITOR' (see development dataset,
    PMID 17234158, Arg1:T5, Arg2:T19). Also, it is very important to
    note that an entity pair can have distinct CPRs but only one is
    chosen. Evaluated CPRs have higher priority, then follow the
    non-evaluated CPRs and finally the 'CPR:0' meaning no relation.
    Also, by default lower values were chosen to have higher priority,
    that is, e.g. inside the evaluated groups: 'CPR:3' has more priority
    than 'CPR:4') (see development dataset, PMID 16917142, Arg1:T11,
    Arg2:T31).

    Parameters
    ----------
    entity_pair : tuple
        The entity pair is a `tuple` containing the two term number
        identifiers. Example: ('T49', 'T56').
    relations : list
        A `list` of relations. Each relation is a `dict` (see
        `get_pmid2relations`).

    Returns
    -------
    cpr : str
        A Chemical-Protein Relation (CPR) `str` is returned. The
        `entity_pair` relation is classified as being in the CPR class.

    Raises
    ------
    AssertionError
        If `entity_pair` input argument is not a valid entity pair.

    Example
    -------
    >>> from support import get_pmid2relations
    >>> from support import entity_pair_in_relations
    >>> group = 'development'
    >>> pmid2relations = get_pmid2relations(group)
    >>> pmid = '16917142'
    >>> relations = pmid2relations[pmid]
    >>> entity_pair = ('T1', 'T2')
    >>> cpr = entity_pair_in_relations(entity_pair, relations)
    >>> cpr
    'CPR:0'
    >>> entity_pair = ('T11', 'T30')
    >>> cpr = entity_pair_in_relations(entity_pair, relations)
    >>> cpr
    'CPR:2'
    >>> entity_pair = ('T4', 'T24')
    >>> cpr = entity_pair_in_relations(entity_pair, relations)
    >>> cpr
    'CPR:4'
    >>> entity_pair = ('T11', 'T31')
    >>> cpr = entity_pair_in_relations(entity_pair, relations)
    >>> cpr
    'CPR:5'
    >>> entity_pair = ('T4', 'T23')
    >>> cpr = entity_pair_in_relations(entity_pair, relations)
    >>> cpr
    'CPR:6'
    >>>
    >>> for pmid, relations in pmid2relations.items():
    ...     pair2cprs = dict()
    ...     for rel in relations:
    ...         r = (rel['arg1']['tn'], rel['arg2']['tn'])
    ...         if r not in pair2cprs:
    ...             pair2cprs[r] = list()
    ...         pair2cprs[r].append(rel['cpr'])
    ...     for pair, cprs in pair2cprs.items():
    ...         if len(cprs) > 1:
    ...             print('{:}\t{:}\t{:}'.format(pmid, pair, cprs))
    ...
    10579749	('T1', 'T44')	['CPR:2', 'CPR:6']
    10579749	('T2', 'T44')	['CPR:2', 'CPR:6']
    10579749	('T3', 'T44')	['CPR:2', 'CPR:6']
    10579749	('T4', 'T44')	['CPR:2', 'CPR:6']
    10579749	('T5', 'T44')	['CPR:2', 'CPR:6']
    10579749	('T30', 'T44')	['CPR:2', 'CPR:6']
    10579749	('T31', 'T44')	['CPR:2', 'CPR:6']
    10579749	('T32', 'T44')	['CPR:2', 'CPR:6']
    16819260	('T1', 'T13')	['CPR:2', 'CPR:3']
    16819260	('T1', 'T19')	['CPR:2', 'CPR:3']
    16917142	('T11', 'T31')	['CPR:5', 'CPR:6']
    17234158	('T5', 'T19')	['CPR:4', 'CPR:4']
    17234158	('T5', 'T20')	['CPR:4', 'CPR:4']
    17234158	('T5', 'T21')	['CPR:4', 'CPR:4']
    19057128	('T3', 'T17')	['CPR:3', 'CPR:3']
    19057128	('T3', 'T18')	['CPR:3', 'CPR:3']
    19057128	('T4', 'T17')	['CPR:4', 'CPR:4']
    19057128	('T4', 'T18')	['CPR:4', 'CPR:4']
    2125244	('T1', 'T7')	['CPR:3', 'CPR:4']
    2125244	('T1', 'T8')	['CPR:3', 'CPR:4']
    23261676	('T15', 'T31')	['CPR:4', 'CPR:4']
    23261676	('T15', 'T32')	['CPR:4', 'CPR:4']
    23261676	('T15', 'T33')	['CPR:4', 'CPR:4']
    23261676	('T15', 'T34')	['CPR:4', 'CPR:4']
    23312278	('T1', 'T3')	['CPR:2', 'CPR:2']
    23319419	('T18', 'T39')	['CPR:2', 'CPR:3']
    23319419	('T18', 'T40')	['CPR:2', 'CPR:3']
    23319419	('T18', 'T41')	['CPR:2', 'CPR:3']
    23611809	('T4', 'T30')	['CPR:2', 'CPR:9']
    23611809	('T4', 'T31')	['CPR:2', 'CPR:9']
    23611809	('T4', 'T33')	['CPR:2', 'CPR:9']
    23611809	('T5', 'T30')	['CPR:2', 'CPR:9']
    23611809	('T5', 'T31')	['CPR:2', 'CPR:9']
    23611809	('T5', 'T32')	['CPR:2', 'CPR:9']
    23611809	('T5', 'T33')	['CPR:2', 'CPR:9']
    23611809	('T6', 'T30')	['CPR:2', 'CPR:9']
    23611809	('T6', 'T31')	['CPR:2', 'CPR:9']
    23611809	('T6', 'T32')	['CPR:2', 'CPR:9']
    23611809	('T6', 'T33')	['CPR:2', 'CPR:9']
    23611809	('T9', 'T35')	['CPR:2', 'CPR:9']
    23611809	('T9', 'T36')	['CPR:2', 'CPR:9']
    23611809	('T9', 'T37')	['CPR:2', 'CPR:9']
    2857786	('T15', 'T19')	['CPR:4', 'CPR:6']
    >>>

    """
    assert entity_pair == its2tns(tns2its(entity_pair))
    def priority(cpr):
        return {
            'CPR:0': 10,
            'CPR:1': 5,
            'CPR:2': 6,
            'CPR:3': 0,
            'CPR:4': 1,
            'CPR:5': 2,
            'CPR:6': 3,
            'CPR:7': 7,
            'CPR:8': 8,
            'CPR:9': 4,
            'CPR:10': 9,
        }[cpr]
    cprs = {'CPR:0'}
    for rel in relations:
        if entity_pair == (rel['arg1']['tn'], rel['arg2']['tn']):
            cprs.add(rel['cpr'])
    return sorted(cprs, key=priority)[0]


def get_processed_sample(pmid2tees, pmid2entities, pmid2entity_pairs, pmid,
    sentence_number, entity_pair):
    r"""
    Get CHEMPROT relation sample to add into the processed corpus.

    This is an auxiliary function of the `create_processed_corpus`. Its
    goal is to return a `str` containing the full sample of a CHEMPROT
    relation. For more details see the documentation of the
    `create_processed_corpus` function.

    Parameters
    ----------
    pmid2tees : dict
        `dict` with TEES information of all PMID abstracts.
    pmid2entities : dict
        Each element is a `list` with entities of all PMID abstracts.
        Each entity is a `dict`.
    pmid2entity_pairs : dict
        Each element is a `list` of entity pairs. Each entry in this
        `list` says respect to one sentence. For example,
        `entity_pairs[0]` is a `list` with the entity pairs present in
        the sentence number 0.
    pmid : str
        PMID identifier.
    sentence_number : int
        The sentence number (starts at 0).
    entity_pair : tuple
        The entity pair is a `tuple` containing the two term number
        identifiers. Example: ('T49', 'T56').

    Returns
    -------
    sample : str
        The `str` processed sample containing five lines (shortest
        dependency path, left/right subsequences of the chemical/gene
        entities). More information see `create_processed_corpus`.

    Raises
    ------
    AssertionError
        If input arguments are inconsistent.

    Example
    -------
    >>> from support import get_pmid2entities
    >>> from support import get_pmid2tees
    >>> from support import get_pmid2entity_pairs
    >>> from support import get_processed_sample
    >>>
    >>> group = 'development'
    >>> pmid2tees = get_pmid2tees(group)
    >>> pmid2entities = get_pmid2entities(group)
    >>> pmid2entity_pairs = get_pmid2entity_pairs(group)
    >>>
    >>> pmid = '10381787'
    >>> sentence_number = 0
    >>> entity_pair = pmid2entity_pairs[pmid][sentence_number][0]
    >>> sample = get_processed_sample(pmid2tees, pmid2entities,
    ...     pmid2entity_pairs, pmid, sentence_number, entity_pair)
    >>> print(sample, end='')
    meloxicam|NN|prep_by|inhibition|NN|prep_of|cyclooxygenase-1|NN|
    and|CC||monocyte|NN|nn|cyclooxygenase-2|NN|prep_of|by|IN|
    in|IN||healthy|JJ|amod|subjects|NNS|prep_in|.|.|punct
    Dose-dependent|JJ|amod|inhibition|NN||of|IN||platelet|NN|nn
    and|CC||monocyte|NN|nn|cyclooxygenase-2|NN|prep_of|by|IN|
    >>>
    >>> # example with multiple SDPs
    >>> pmid = '10082498'
    >>> sentence_number = 4
    >>> entity_pair = ('T16', 'T31')
    >>> sample = get_processed_sample(pmid2tees, pmid2entities,
    ...     pmid2entity_pairs, pmid, sentence_number, entity_pair)
    >>> print(sample, end='')
    Ang|NN|nn|assay|NN|dep|methods|NNS|dep|changes|NNS|prep_in|levels|NNS|nn|Ang|NN|
    Blockade|NN|nsubjpass|of|IN||the|DT|det|renin-angiotensin|JJ|amod|system|NN|prep_of|was|VBD|auxpass|assessed|VBN||before|IN||and|CC||4|CD|dep|,|,|punct|24|CD|num|,|,|punct|and|CC||30|CD|num|hours|NNS|prep_before|after|IN||drug|NN|nn|intake|NN|prep_after|by|IN||3|CD|num|independent|JJ|amod|methods|NNS|agent|:|:|punct|inhibition|NN|dep|of|IN||the|DT|det|blood|NN|nn|pressure|NN|nn|response|NN|prep_of|to|TO||exogenous|JJ|amod|Ang|NN|prep_to|II|CD|num|,|,|punct|in|FW|amod|vitro|FW|dep
    receptor|NN|nn|assay|NN|dep|,|,|punct|and|CC||reactive|JJ|amod|changes|NNS|dep|in|IN||plasma|NN|nn
    receptor|NN|nn|assay|NN|dep|,|,|punct|and|CC||reactive|JJ|amod|changes|NNS|dep|in|IN||plasma|NN|nn
    levels|NNS|prep_in|.|.|punct
    >>>
    >>> # example with token headscore=-1
    >>> pmid = '10432475'
    >>> sentence_number = 4
    >>> entity_pair = ('T35', 'T55')
    >>> sample = get_processed_sample(pmid2tees, pmid2entities,
    ...     pmid2entity_pairs, pmid, sentence_number, entity_pair)
    >>> print(sample, end='')
    WAY100,635|NN|appos|antagonist|NN|nn|5-HT1A|NN|
    receptor|NN|nn|antagonist|NN|nsubj|,|,|punct
    ,|,|punct|slightly|RB|advmod|attenuated|VBD||the|DT|det|(|(|punct|-|CC|cc|)|)|punct|-pindolol-induced|JJ|amod|increase|NN|dobj|in|IN||DA|NN|nn|and|CC||NAD|NN|nn|levels|NNS|prep_in|,|,|punct|while|IN|mark|the|DT|det|selective|JJ|amod|5-HT1B|NN|nn|antagonist|NN|nsubj|,|,|punct|SB224,289|NN|appos|,|,|punct|was|VBD|cop|ineffective|JJ|advcl|.|.|punct
    The|DT|det|selective|JJ|amod
    receptor|NN|nn|antagonist|NN|nsubj|,|,|punct
    >>>

    """
    sample = ''
    # short aliases
    sn = sentence_number
    ep = entity_pair
    # select the necessary information given the PMID input argument
    tees = pmid2tees[pmid]
    entities = pmid2entities[pmid]
    entity_pairs = pmid2entity_pairs[pmid]
    # get tokens and dependencies
    tokens = tees[sn]['token']
    dependencies = tees[sn]['dependency']
    # check if entity pair exists
    e = 'PMID: {}. Sentence number: {}. Entity pair {} not found!'.format(
        pmid, sn, ep)
    assert ep in entity_pairs[sn], e
    # the two interactor arguments obtained from the entity pair
    entity1 = entities[to_int(ep[0]) - 1]
    entity2 = entities[to_int(ep[1]) - 1]
    # assert that the entities are correct
    assert entity1['tn'] == ep[0]
    assert entity2['tn'] == ep[1]
    # CHEMICAL/GENE entities
    if entity1['type'] == 'CHEMICAL':
        chem = entity1
        gene = entity2
        assert entity2['type'][:4] == 'GENE'
    else:
        chem = entity2
        gene = entity1
        assert entity2['type'] == 'CHEMICAL'
    # find the tokens of the CHEMICAL and GENE entities
    chem_tokens = list()
    gene_tokens = list()
    for id_, t in tokens.items():
        d = {
            'id': id_,
            'pos': t['pos'],
            'head_score': t['head_score'],
            'text': t['text'],
        }
        if overlap(chem['i'], t['char_offset']):
            chem_tokens.append(d)
        if overlap(gene['i'], t['char_offset']):
            gene_tokens.append(d)
    # sort the tokens accordingly to their (1) headscore, (2) text and
    # (3) POS, so the most informative is chosen as the head token
    chem_tokens.sort(key=lambda t: t['head_score'], reverse=True)
    gene_tokens.sort(key=lambda t: t['head_score'], reverse=True)
    def hasletters(s):
        return any(c.isalpha() for c in s)
    chem_tokens.sort(key=lambda t: hasletters(t['text']), reverse=True)
    gene_tokens.sort(key=lambda t: hasletters(t['text']), reverse=True)
    def pos2info(pos):
        if not hasletters(pos):
            return 0
        elif pos[:2] == 'NN':
            return 2
        else:
            return 1
    chem_tokens.sort(key=lambda t: pos2info(t['pos']), reverse=True)
    gene_tokens.sort(key=lambda t: pos2info(t['pos']), reverse=True)
    # the head token IDs
    chem_token_id = chem_tokens[0]['id']
    gene_token_id = gene_tokens[0]['id']
    # add tokens (nodes) and dependencies (edges) to a NetworkX Graph
    G = nx.Graph()
    D = nx.DiGraph()
    G.add_nodes_from(tokens)
    D.add_nodes_from(tokens)
    for d in dependencies.values():
        node1 = d['t1']
        node2 = d['t2']
        edge_score = tokens[node1]['head_score'] + tokens[node2]['head_score']
        # sum a offset of 3 because the headscore can be -1
        edge_score = edge_score + 3
        assert edge_score > 0
        weight = 1 / edge_score
        G.add_edge(node1, node2, weight=weight, type=d['type'])
        D.add_edge(node1, node2, weight=weight, type=d['type'])
    try:
        sdps = list(
            nx.all_shortest_paths(
                G=G,
                source=chem_token_id,
                target=gene_token_id,
            )
        )
        if len(sdps) > 1:
            #print('pmid: {}, sn: {}, ep: {}.'.format(pmid, sn, ep))
            # weight of the SDPs
            weights = [calculate_path_weight(G, path) for path in sdps]
            index_min_weight = weights.index(min(weights))
            # choose the SDP with lowest weight
            sdp = sdps[index_min_weight]
        else:
            sdp = sdps[0]
    except nx.exception.NetworkXNoPath:
        sdp = []
    # add shortest dependency path string
    if sdp:
        for i in range(len(sdp)-1):
            node1 = sdp[i]
            node2 = sdp[i + 1]
            sample += tokens[node1]['text'] + '|'
            sample += tokens[node1]['pos'] + '|'
            sample += G[node1][node2]['type'] + '|'
        sample += tokens[sdp[-1]]['text'] + '|'
        sample += tokens[sdp[-1]]['pos'] + '|'
    sample += '\n'
    # find the three subsequences (left/middle/right text)
    e1_offset = min(chem['i'][0], gene['i'][0])
    e2_offset = max(chem['i'][0], gene['i'][0])
    left = middle = right = ''
    for id_, t in tokens.items():
        # if token does not overlap with the CHEMICAL or GENE entities,
        # then it can be used in the subsequences
        if not (
            overlap(chem['i'], t['char_offset'])
            or
            overlap(gene['i'], t['char_offset'])
        ):
            incoming_edges = list(D.in_edges(id_, data=True))
            # select the incoming edge with lower weight (higher score)
            if incoming_edges:
                # it is possible to have more than one incoming edge
                in_edge = min(incoming_edges, key=lambda x: x[2]['weight'])
                in_edge = in_edge[2]['type']
            else:
                in_edge = ''
            s = t['text'] + '|' + t['pos'] + '|' + in_edge + '|'
            t_offset = t['char_offset'][0]
            if t_offset < e1_offset:
                left += s
            if (t_offset > e1_offset) and (t_offset < e2_offset):
                middle += s
            if t_offset > e2_offset:
                right += s
    left = left[:-1] + '\n'
    middle = middle[:-1] + '\n'
    right = right[:-1] + '\n'
    # add CHEMICAL/GENE left/right text
    if chem_token_id == gene_token_id:
        # CHEMICAL and GENE head tokens are the same token
        sample += (left + right) * 2
    elif chem['i'][0] < gene['i'][0]:
        # CHEMICAL before GENE
        sample += left + middle * 2 + right
    else:
        # CHEMICAL after GENE
        sample += middle + right + left + middle
    return sample


def load_features_from_sample(sample, words=None):
    r"""
    Loads the features from a `str` sample to `list` values.

    This function converts the `str` sample data to a `list` of five
    features:
      - SDP: tokens, POS, dependencies (edges).
      - LCH: tokens, POS, dependencies (incoming edges).
      - RCH: tokens, POS, dependencies (incoming edges).
      - LGE: tokens, POS, dependencies (incoming edges).
      - RGE: tokens, POS, dependencies (incoming edges).
    Some processing is done:
      1. The tokens, POS, and dependencies are converted to lower case.
      2. If `words` is a `set`: the tokens (and the respective POS and
         dependencies) not in this `set` are discarded.
      3. If there are at least two tokens in the SDP, then the first and
         the last tokens are respectively the chemical and gene head
         tokens, and in that case they will be replaced by the
         '#chemical' and '#gene' tags.
      4. If there is only one token in the SDP, that token represents
         the chemical and gene head tokens, and in that case the token
         will be replaced by the '#chemical#gene' tag.
      5. Empty incoming edges (dependency edges) will be replaced by the
         '#none' tag.
    Acronyms/abbreviations:
      SDP: Shortest Dependency Path.
      LCH: Left (text relative to) CHemical.
      RCH: Right (text relative to) CHemical.
      LGE: Left (text relative to) GEne.
      RGE: Right (text relative to) GEne.
      POS: Part-Of-Speech.

    Parameters
    ----------
    sample : str
        CHEMPROT processed sample.
    words : set, optional
        The `set` of words to consider. Important note: it cannot be a
        `list` because it will degrade performance. If `None` all words
        will be considered. Default: `None`.

    Returns
    -------
    features: list
        Each entry of the `list` correspondes to a type of feature.
        There are five feature types: SDP, LCH, RCH, LGE, RGE. All the
        features contain tokens, POS and dependency features.

    Example
    -------
    >>> import zipfile
    >>> zf = './data/chemprot_training/support/processed_corpus.zip'
    >>> with zipfile.ZipFile(zf) as z:
    ...   name = z.namelist()[2]
    ...   print(name)
    ...   sample = z.open(name).read().decode('utf-8')
    ...
    processed_corpus/CPR:0/10047461_0_T15_T61.txt
    >>>
    >>> print(sample)
    thymidylate|NN|nn|synthase|NN|nn|Tomudex|NN|agent|induced|VBN|vmod|arrest|NN|prep_with|associated|VBN|nsubjpass|activation|NN|nn|Cyclin|NN|
    activation|NN|nsubjpass|is|VBZ|auxpass|associated|VBN||with|IN||cell|NN|nn|cycle|NN|nn|arrest|NN|prep_with|and|CC||inhibition|NN|prep_with|of|IN||DNA|NN|nn|replication|NN|prep_of|induced|VBN|vmod|by|IN||the|DT|det
    synthase|NN|nn|inhibitor|NN|nn|Tomudex|NN|agent|.|.|punct

    activation|NN|nsubjpass|is|VBZ|auxpass|associated|VBN||with|IN||cell|NN|nn|cycle|NN|nn|arrest|NN|prep_with|and|CC||inhibition|NN|prep_with|of|IN||DNA|NN|nn|replication|NN|prep_of|induced|VBN|vmod|by|IN||the|DT|det

    >>> from support import load_features_from_sample
    >>> features = load_features_from_sample(sample)
    >>> features[0][0][:3]  # SDP: first 3 tokens
    ['#chemical', 'synthase', 'tomudex']
    >>> features[0][1][:3]  # SDP: first 3 POS
    ['nn', 'nn', 'nn']
    >>> features[0][2][:3]  # SDP: first 3 dependencies
    ['nn', 'nn', 'agent']
    >>>

    """
    assert (words is None) or isinstance(words, set)
    lines = [line.split('|') if line else [] for line in sample.splitlines()]
    n_features = 5
    features = [list() for _ in range(n_features)]
    # SDP
    features[0] = [
        [w.lower() for w in lines[0][0::3]],
        [w.lower() for w in lines[0][1::3]],
        [w.lower() or '#none' for w in lines[0][2::3]],
    ]
    if len(features[0][0]) == 1:
        # in this case the unique token represents simultaneously
        # the CHEMICAL and the GENE head tokens
        features[0][0][0] = '#chemical#gene'
    if len(features[0][0]) >= 2:
        # in this case the first and the last tokens represent
        # respectively the CHEMICAL and the GENE head tokens
        features[0][0][0] = '#chemical'
        features[0][0][-1] = '#gene'
    # LCH, RCH, LGE, RGE
    for i in range(1, n_features):
        features[i] = [
            [w.lower() for w in lines[i][0::3]],
            [w.lower() for w in lines[i][1::3]],
            [w.lower() or '#none' for w in lines[i][2::3]],
        ]
    if words is None:
        return features
    else:
        # SDP, LCH, RCH, LGE, RGE: remove the tokens (and the respective
        # POS and dependencies) that are not in the `words` set
        new_features = [[list(), list(), list()] for _ in range(n_features)]
        for i in range(n_features):
            wrd, pos, dep = features[i]
            for w, p, d in zip(wrd, pos, dep):
                if w in words:
                    new_features[i][0].append(w)
                    new_features[i][1].append(p)
                    new_features[i][2].append(d)
        return new_features


def load_data_from_zips(zips, label2index, shuffle=True, random_state=0,
    **kwargs):
    r"""
    Loads data from from a .ZIP file (sklearn-compatible corpus).

    The loading from a .ZIP file is much faster than reading from
    several files. This is due because there are thousands of files,
    which slows the reading process.

    Parameters
    ----------
    zips : list
        A `list` of zip files to read.
    label2index : dict
        A mapping from a CPR label to an `int` index.
    shuffle : bool, optional
        Data is to be shuffled? Default: `True`.
    random_state : int, optional
        Random number to seed the random generator. Default: 0.
    **kwargs : dict, optional
        Keyword arguments that will be used as input in the internal
        call of the function `load_features_from_sample` to customize
        the feature selection.

    Returns
    -------
    data: dict
        This `dict` contains all the information of the dataset.

    Example
    -------
    >>> from support import LABEL2INDEX
    >>> from support import load_data_from_zips
    >>> zips = [
    ...     './data/chemprot_development/support/processed_corpus.zip',
    ... ]
    >>> data = load_data_from_zips(zips=zips, label2index=LABEL2INDEX)
    >>> data['info'][0]['f']
    'processed_corpus/CPR0/11309392_8_T2_T16.txt'
    >>>

    """
    data = dict()
    data['data'] = list()
    data['target'] = list()
    data['info'] = list()
    if len(zips) == 0:
        return data
    # find the total number of samples
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for fpath in zf.namelist():
                if fpath.endswith('.txt'):
                    dpath, fn = os.path.split(fpath)
                    root, cpr = os.path.split(dpath)
                    pmid, sn, t1, t2 = os.path.splitext(fn)[0].split('_')
                    a1, a2 = tns2its((t1, t2))
                    data['data'].append(
                        load_features_from_sample(
                            sample=zf.open(fpath).read().decode('utf-8'),
                            **kwargs,
                        )
                    )
                    data['target'].append(label2index[cpr])
                    data['info'].append(
                        {
                            'z': z,
                            'f': fpath,
                            'cpr': cpr,
                            'pmid': pmid,
                            'sn': int(sn),
                            'a1': a1,
                            'a2': a2,
                        }
                    )
    if shuffle:
        random.seed(random_state)
        z = list(zip(data['data'], data['target'], data['info']))
        random.shuffle(z)
        a, b, c = zip(*z)
        data['data'] = list(a)
        data['target'] = list(b)
        data['info'] = list(c)
    return data


def chemprot_eval(gold_standard_fpaths, predictions_fpaths):
    r"""
    Mimics the behaviour of the CHEMPROT evaluation script.

    The CHEMPROT predictions are evaluated (recall, precision, and
    f-score).

    Parameters
    ----------
    gold_standard_fpaths : list, str
        A `list` containing the gold standard relations file paths.
    predictions_fpaths : list, str
        A `list` containing the prediction relations file paths.

    Returns
    -------
    results: dict
        This is a `dict` containing all the results.

    Example
    -------
    >>> from support import chemprot_eval
    >>> gold_standard_fpaths = [
    ...     'data/chemprot_test_gs/chemprot_test_gs_gold_standard.tsv'
    ... ]
    >>> predictions_fpaths = [
    ...     'data/chemprot_test_gs/chemprot_test_gs_gold_standard.tsv',
    ... ]
    >>> results = chemprot_eval(gold_standard_fpaths, predictions_fpaths)
    >>> results['annotations'], results['predictions']
    (3458, 3458)
    >>> results['TP'], results['FP'], results['FN']
    (3458, 0, 0)
    >>> results['precision'], results['recall'], results['f-score']
    (1.0, 1.0, 1.0)
    >>>

    """
    gold_standard = set()
    for fp in gold_standard_fpaths:
        with open(fp) as f:
            gold_standard.update(f.read().splitlines())
    predictions = set()
    for fp in predictions_fpaths:
        with open(fp) as f:
            predictions.update(f.read().splitlines())
    results = dict()
    results['annotations'] = len(gold_standard)
    results['predictions'] = len(predictions)
    TP = 0
    FP = 0
    FN = 0
    for predicted_relation in predictions:
        if predicted_relation in gold_standard:
            TP += 1
        else:
            FP += 1
    for gold_standard_relation in gold_standard:
        if gold_standard_relation not in predictions:
            FN += 1
    results['TP'] = TP
    results['FP'] = FP
    results['FN'] = FN
    results['precision'] = precision(TP, FP)
    results['recall'] = recall(TP, FN)
    results['f-score'] = f1_score(TP, FP, FN)
    return results


def chemprot_eval_arrays(y_true, y_pred):
    r"""
    Mimics the behaviour of the CHEMPROT evaluation script.

    The CHEMPROT predictions are evaluated (recall, precision, and
    f-score).
    In this function the input arguments use integer labels: `{0, 1, 2,
    3, 4, 5}`. The integer `0` corresponds to the negative class 'CPR:0'
    being ignored. See `INDEX2LABEL`.

    Parameters
    ----------
    y_true : array-like
        Array with the true integer labels.
    y_pred : array-like
        Array with the predicted integer labels.

    Returns
    -------
    results: dict
        This is a `dict` containing all the results.

    Example
    -------
    >>> from support import chemprot_eval_arrays
    >>> y_true = [1, 2, 3, 4, 5, 0, 0, 3, 5, 0]
    >>> y_pred = [1, 2, 3, 4, 5, 1, 5, 1, 0, 0]
    >>> results = chemprot_eval_arrays(y_true, y_pred)
    >>> results['annotations'], results['predictions']
    (7, 8)
    >>> results['TP'], results['FP'], results['FN']
    (5, 3, 2)
    >>> results['precision'], results['recall'], results['f-score']
    (0.625, 0.7142857142857143, 0.6666666666666666)
    >>>

    """
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    annotations = 0
    predictions = 0
    TP = 0
    FP = 0
    FN = 0
    for t, p in zip(y_true, y_pred):
        if t != 0:
            annotations += 1
        if p != 0:
            predictions += 1
        if (t == 0) and (p == 0):
            pass
        elif (t == 0) and (p != 0):
            FP += 1
        elif (t != 0) and (p == 0):
            FN += 1
        else:
            if t == p:
                TP += 1
            else:
                FN += 1
                FP += 1
    results = dict()
    results['annotations'] = annotations
    results['predictions'] = predictions
    results['TP'] = TP
    results['FP'] = FP
    results['FN'] = FN
    results['precision'] = precision(TP, FP)
    results['recall'] = recall(TP, FN)
    results['f-score'] = f1_score(TP, FP, FN)
    return results


# #functions-evaluation
def precision(tp, fp):
    r"""
    Precision is calculated.

    Parameters
    ----------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.

    Returns
    -------
    p : float
        Precision float value.

    """
    try:
        p = tp / (tp + fp)
    except ZeroDivisionError:
        p = 0.0
    return p


def recall(tp, fn):
    r"""
    Recall is calculated.

    Parameters
    ----------
    tp : int
        Number of true positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    r : float
        Recall float value.

    """
    try:
        r = tp / (tp + fn)
    except ZeroDivisionError:
        r = 0.0
    return r


def f1_score(tp, fp, fn):
    r"""
    F1-score is calculated.

    Parameters
    ----------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    f : float
        F1-score float value.

    """
    try:
        p = precision(tp, fp)
        r = recall(tp, fn)
        f = 2 * (p * r) / (p + r)
    except ZeroDivisionError:
        f = 0.0
    return f


# #functions-networkx
def calculate_path_weight(G, path):
    r"""
    Calculate the path weight.

    The path weight is the sum of the weights of all the constituent
    nodes.

    Parameters
    ----------
    G : NetworkX graph
        Graph.
    path : list
        The path containing the constituent nodes.

    Returns
    -------
    weight : float
        The sum of the weights of all the nodes.

    Example
    -------
    >>> import networkx as nx
    >>> from support import calculate_path_weight
    >>> G = nx.Graph()
    >>> G.add_nodes_from([1, 2, 3, 4, 5])
    >>> G.add_edge(1, 2, weight=0.2)
    >>> G.add_edge(2, 3, weight=0.3)
    >>> G.add_edge(3, 4, weight=0.4)
    >>> G.add_edge(4, 5, weight=0.5)
    >>> G.add_edge(1, 5, weight=1.8)
    >>> path = nx.shortest_path(G, 1, 5)
    >>> path
    [1, 5]
    >>> w = calculate_path_weight(G, path)
    >>> w
    1.8
    >>> path = nx.shortest_path(G, 1, 5, weight='weight')
    >>> path
    [1, 2, 3, 4, 5]
    >>> w = calculate_path_weight(G, path)
    >>> w
    1.4
    >>>

    """
    weight = 0.0
    n = len(path)
    for i in range(n-1):
        node1 = path[i]
        node2 = path[i+1]
        weight += G[node1][node2]['weight']
    return weight


# #functions-main
def check_dataset():
    r"""
    Dataset is consistent?

    This function makes a weak consistency check of the CHEMPROT
    dataset. Required tree:

    data/
    |-- chemprot_development/
    |   |-- chemprot_development_abstracts.tsv
    |   |-- chemprot_development_entities.tsv
    |   |-- chemprot_development_gold_standard.tsv
    |   `-- chemprot_development_relations.tsv
    |-- chemprot_test_gs/
    |   |-- chemprot_test_gs_abstracts.tsv
    |   |-- chemprot_test_gs_entities.tsv
    |   |-- chemprot_test_gs_gold_standard.tsv
    |   `-- chemprot_test_gs_relations.tsv
    `-- chemprot_training/
        |-- chemprot_training_abstracts.tsv
        |-- chemprot_training_entities.tsv
        |-- chemprot_training_gold_standard.tsv
        `-- chemprot_training_relations.tsv

    Raises
    ------
    AssertionError
        If dataset is not consistent.

    Example
    -------
    >>> from support import check_dataset
    >>> check_dataset()
    >>>

    """
    e = '{} is expected to be a {}!'
    assert os.path.isdir(DATA), e.format(repr(DATA), 'directory')
    for g in ('training', 'development', 'test_gs'):
        gp = os.path.join(DATA, 'chemprot_' + g)
        assert os.path.isdir(gp), e.format(repr(gp), 'directory')
        required = ['abstracts', 'entities', 'gold_standard', 'relations']
        for f in required:
            fp = os.path.join(gp, 'chemprot_' + g + '_' + f + '.tsv')
            assert os.path.isfile(fp), e.format(repr(fp), 'file')


def create_consistency():
    r"""
    Check for inconsistencies/contradictions in the dataset.

    This function creates a 'consistency.tsv' file inside the CHEMPROT
    dataset directory path. This file contains several consistency
    checks:
      - Search for CHEMICAL entities that have higher term numbers than
        the GENE entities.
      - Interactor terms in (gold standard) relations have to be sorted
        in ascending order according to their corresponding term
        numbers.
      - Search for erroneous (gold standard) relations (relations
        between two entities of the same type: chemical-chemical,
        gene-gene).
      - Search for duplicate (gold standard) relations.

    Example
    -------
    >>> from support import create_consistency
    >>> create_consistency()
    >>>

    """
    x = 0
    n = 9
    progress(x=x, n=n)
    fp = os.path.join(DATA, 'consistency.tsv')
    f = open(fp, mode='w', encoding='utf-8')
    # entities ---------------------------------------------------------
    _ = f.write(
        'CHEMICAL-entities-after-GENES\n'
        'group\tPMID\tterm-number\ttype\ti\ttext\n'
    )
    chemical_entities_after_genes = list()
    for group in GROUPS:
        pmid2entities = get_pmid2entities(group)
        for pmid, entities in pmid2entities.items():
            gene_appeared = False
            for e in entities:
                if e['type'][:4] == 'GENE':
                    gene_appeared = True
                if gene_appeared and (e['type'] == 'CHEMICAL'):
                    chemical_entities_after_genes.append([
                        group,
                        pmid,
                        e['tn'],
                        e['type'],
                        e['i'],
                        e['text'],
                    ])
        x += 1
        progress(x=x, n=n)
    for line in chemical_entities_after_genes:
        _ = f.write(('{}' + '\t{}' * 5 + '\n').format(*line))
    _ = f.write('\n')
    # relations --------------------------------------------------------
    descending = list()
    erroneous = list()
    duplicated = list()
    for group in GROUPS:
        pmid2relations = get_pmid2relations(group)
        for pmid, relations in pmid2relations.items():
            unique = set()
            for r in relations:
                rel = (
                    group,
                    pmid,
                    r['cpr'],
                    r['eval'],
                    r['cpi'],
                    r['arg1']['tn'],
                    r['arg1']['type'],
                    r['arg1']['i'],
                    r['arg1']['text'],
                    r['arg2']['tn'],
                    r['arg2']['type'],
                    r['arg2']['i'],
                    r['arg2']['text'],
                )
                if to_int(r['arg1']['tn']) >= to_int(r['arg2']['tn']):
                    descending.append(rel)
                if r['arg1']['type'] == r['arg2']['type']:
                    erroneous.append(rel)
                if rel in unique:
                    duplicated.append(rel)
                unique.add(rel)
        x += 1
        progress(x=x, n=n)
    _ = f.write(
        'descending-relations\n'
        'group\tPMID\tCPR\teval\tCPI'
        '\targ1-tn\targ1-type\targ1-i\targ1-text'
        '\targ2-tn\targ2-type\targ2-i\targ2-text\n'
    )
    for line in descending:
        _ = f.write(('{}' + '\t{}' * 12 + '\n').format(*line))
    _ = f.write('\n')
    _ = f.write(
        'erroneous-relations\n'
        'group\tPMID\tCPR\teval\tCPI'
        '\targ1-tn\targ1-type\targ1-i\targ1-text'
        '\targ2-tn\targ2-type\targ2-i\targ2-text\n'
    )
    for line in erroneous:
        _ = f.write(('{}' + '\t{}' * 12 + '\n').format(*line))
    _ = f.write('\n')
    _ = f.write(
        'duplicated-relations\n'
        'group\tPMID\tCPR\teval\tCPI'
        '\targ1-tn\targ1-type\targ1-i\targ1-text'
        '\targ2-tn\targ2-type\targ2-i\targ2-text\n'
    )
    for line in duplicated:
        _ = f.write(('{}' + '\t{}' * 12 + '\n').format(*line))
    _ = f.write('\n')
    # gold_standard ----------------------------------------------------
    descending = list()
    erroneous = list()
    duplicated = list()
    for group in GROUPS:
        pmid2gs = get_pmid2gold_standard(group)
        for pmid, gs in pmid2gs.items():
            unique = set()
            for r in gs:
                rel = (
                    group,
                    pmid,
                    r['cpr'],
                    r['arg1']['tn'],
                    r['arg1']['type'],
                    r['arg1']['i'],
                    r['arg1']['text'],
                    r['arg2']['tn'],
                    r['arg2']['type'],
                    r['arg2']['i'],
                    r['arg2']['text'],
                )
                if to_int(r['arg1']['tn']) >= to_int(r['arg2']['tn']):
                    descending.append(rel)
                if r['arg1']['type'] == r['arg2']['type']:
                    erroneous.append(rel)
                if rel in unique:
                    duplicated.append(rel)
                unique.add(rel)
        x += 1
        progress(x=x, n=n)
    _ = f.write(
        'descending-gold_standard\n'
        'group\tPMID\tCPR'
        '\targ1-tn\targ1-type\targ1-i\targ1-text'
        '\targ2-tn\targ2-type\targ2-i\targ2-text\n'
    )
    for line in descending:
        _ = f.write(('{}' + '\t{}' * 10 + '\n').format(*line))
    _ = f.write('\n')
    _ = f.write(
        'erroneous-gold_standard\n'
        'group\tPMID\tCPR'
        '\targ1-tn\targ1-type\targ1-i\targ1-text'
        '\targ2-tn\targ2-type\targ2-i\targ2-text\n'
    )
    for line in erroneous:
        _ = f.write(('{}' + '\t{}' * 10 + '\n').format(*line))
    _ = f.write('\n')
    _ = f.write(
        'duplicated-gold_standard\n'
        'group\tPMID\tCPR'
        '\targ1-tn\targ1-type\targ1-i\targ1-text'
        '\targ2-tn\targ2-type\targ2-i\targ2-text\n'
    )
    for line in duplicated:
        _ = f.write(('{}' + '\t{}' * 10 + '\n').format(*line))
    _ = f.write('\n')
    # ------------------------------------------------------------------
    f.close()
    progress(clear=True)


def create_corpus():
    r"""
    A corpus directory is created.

    This function creates a 'corpus/' directory inside the 'support/'
    directory for each group of the dataset. This 'corpus/' directory
    contains the files (named with their PMIDs) containing the texts
    (title and abstract). This corpus structure is necessary for
    executing the TEES program.

    Example
    -------
    >>> from support import create_corpus
    >>> create_corpus()
    >>>

    """
    x = 0
    n = len(GROUPS)
    progress(x=x, n=n)
    for group in GROUPS:
        dp = os.path.join(DATA, 'chemprot_' + group, 'support', 'corpus')
        create_directory(dp)
        for pmid, text in get_pmid2text(group).items():
            fp = os.path.join(dp, pmid + '.txt')
            with open(fp, mode='w', encoding='utf-8') as f:
                _ = f.write(text)
        x += 1
        progress(x=x, n=n)
    progress(clear=True)


def create_sentences():
    r"""
    A file containing the sentence offsets is created.

    This function creates a 'sentences.tsv' file inside the 'support/'
    directory for each group of the dataset. This 'sentences.tsv' file
    contains the sentence offsets (and the respective PMIDs). This
    information is extracted from the TEES gzip file (expected to be in
    the 'tees/' directory).

    Example
    -------
    >>> from support import create_sentences
    >>> create_sentences()
    >>>

    """
    x = 0
    n = len(GROUPS)
    progress(x=x, n=n)
    for group in GROUPS:
        pmid2tees = get_pmid2tees(group)
        dp = os.path.join(DATA, 'chemprot_' + group, 'support')
        create_directory(dp)
        fp = os.path.join(dp, 'sentences.tsv')
        with open(fp, mode='w', encoding='utf-8') as f:
            for pmid, tees in pmid2tees.items():
                for sentence in tees:
                    s, e = sentence['char_offset']
                    _ = f.write('{}\t{}\t{}\n'.format(pmid, s, e))
        x += 1
        progress(x=x, n=n)
    progress(clear=True)


def create_statistics():
    r"""
    Some dataset statistics are calculated and saved into a file.

    This function creates a 'statistics.tsv' file inside the CHEMPROT
    dataset directory path. This file contains several statistics about
    the:
      - 'abstracts.tsv' file:
        - Number of abstracts.
        - Number of unique PMIDs.
      - 'entities.tsv' file:
        - Number of entities.
        - Number of unique PMIDs.
        - Number of chemicals.
        - Number of genes (gene-y and gene-n).
      - 'relations.tsv' file:
        - Number of relations.
        - Number of unique PMIDs.
        - Number of unique CPR group relations.
        - Number of relations with multiple CPR groups.
        - Number of unique CPR evaluated group relations.
        - Number of evaluated relations with multiple CPR groups.
        - Number of CPR:0, ..., CPR:10 relations.
      - 'gold_standard.tsv' file:
        - Number of gold standard relations.
        - Number of unique PMIDs.
        - Number of unique CPR group relations.
        - Number of relations with multiple CPR groups.
        - Number of CPR:3, CPR:4, CPR:5, CPR:6, CPR:9 relations.
      - Overlapped entities (entities that overlap with each other).
      - Cross-sentence entities (entities that extend to two, or more,
        sentences).
      - Relations with overlapped entities.
      - Relations with cross-sentence entities.
      - Overlapped relations (relations that overlap with other
        relations).
      - Cross-sentence relations (relations between entities from
        different sentences).

    Example
    -------
    >>> from support import create_statistics
    >>> create_statistics()
    >>>

    """
    x = 0
    n = 30
    fp = os.path.join(DATA, 'statistics.tsv')
    f = open(fp, mode='w', encoding='utf-8')
    _ = f.write('group\tabstracts\tunique-PMIDs\n')
    for group in GROUPS:
        abstracts = get_file(group, 'abstracts')
        _ = f.write(('{}' + '\t{}' * 2 + '\n').format(
            group,
            len(abstracts),
            len(get_pmids(group)),
        ))
        x += 1
        progress(x=x, n=n)
    _ = f.write(
        '\ngroup\tentities\tunique-PMIDs\tCHEMICAL\tGENE\tGENE-Y\tGENE-N\n'
    )
    for group in GROUPS:
        entities = get_file(group, 'entities')
        pmids = set()
        counts = {
            'CHEMICAL': 0,
            'GENE-Y': 0,
            'GENE-N': 0,
        }
        for pmid, _, type_, _, _, _ in entities:
            pmids.add(pmid)
            counts[type_] += 1
        _ = f.write(('{}' + '\t{}' * 6 + '\n').format(
            group,
            len(entities),
            len(pmids),
            counts['CHEMICAL'],
            counts['GENE-Y'] + counts['GENE-N'],
            counts['GENE-Y'],
            counts['GENE-N'],
        ))
        x += 1
        progress(x=x, n=n)
    _ = f.write(
        '\ngroup\trelations\tunique-PMIDs'
        '\tunique-CPR-relations'
        '\trelations-with-multiple-CPR'
        '\tunique-CPR-evaluated-relations'
        '\tevaluated-relations-with-multiple-CPR'
        '\t' + '\t'.join(CPR_GROUPS) + '\n'
    )
    for group in GROUPS:
        relations = get_file(group, 'relations')
        r = defaultdict(set)
        r_eval = defaultdict(set)
        pmids = set()
        counts = [0] * len(CPR_GROUPS)
        for pmid, cpr, e, _, a1, a2 in relations:
            pmids.add(pmid)
            cpr = to_int(cpr)
            r[(pmid, a1, a2)].add(cpr)
            if e == 'Y':
                r_eval[(pmid, a1, a2)].add(cpr)
        r_mr = 0
        for _, cprs in r.items():
            if len(cprs) > 1:
                r_mr += 1
            for cpr in cprs:
                counts[cpr] += 1
        r_eval_mr = 0
        for _, cprs in r_eval.items():
            if len(cprs) > 1:
                r_eval_mr += 1
        _ = f.write(('{}' + '\t{}' * 17 + '\n').format(
            group,
            len(relations),
            len(pmids),
            sum(len(cprs) for _, cprs in r.items()),
            r_mr,
            sum(len(cprs) for _, cprs in r_eval.items()),
            r_eval_mr,
            *counts,
        ))
        x += 1
        progress(x=x, n=n)
    _ = f.write(
        '\ngroup\tgold_standard\tunique-PMIDs'
        '\tunique-CPR-relations'
        '\trelations-with-multiple-CPR'
        '\t' + '\t'.join(CPR_EVAL_GROUPS) + '\n'
    )
    for group in GROUPS:
        gold_standard = get_file(group, 'gold_standard')
        r = defaultdict(set)
        pmids = set()
        counts = [0] * len(CPR_EVAL_GROUPS)
        cpr2idx = {3: 0, 4: 1, 5: 2, 6: 3, 9: 4}
        for pmid, cpr, a1, a2 in gold_standard:
            pmids.add(pmid)
            cpr = to_int(cpr)
            r[(pmid, a1, a2)].add(cpr)
        r_mr = 0
        for _, cprs in r.items():
            if len(cprs) > 1:
                r_mr += 1
            for cpr in cprs:
                counts[cpr2idx[cpr]] += 1
        _ = f.write(('{}' + '\t{}' * 9 + '\n').format(
            group,
            len(gold_standard),
            len(pmids),
            sum(len(cprs) for _, cprs in r.items()),
            r_mr,
            *counts,
        ))
        x += 1
        progress(x=x, n=n)
    _ = f.write(
        '\noverlapped-entities\n'
        'group\tPMID'
        '\te1-term-number\te2-term-number'
        '\te1-type\te2-type'
        '\te1-i\te2-i'
        '\te1-text\te2-text\n'
    )
    for group in GROUPS:
        pmid2entities = get_pmid2entities(group)
        overlapped_entities = list()
        for pmid, entities in pmid2entities.items():
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    e1 = entities[i]
                    e2 = entities[j]
                    if overlap(e1['i'], e2['i']):
                        overlapped_entities.append((
                            pmid,
                            e1['tn'],
                            e2['tn'],
                            e1['type'],
                            e2['type'],
                            e1['i'],
                            e2['i'],
                            e1['text'],
                            e2['text'],
                        ))
        for v in overlapped_entities:
            _ = f.write(('{}' + '\t{}' * 9 + '\n').format(group, *v))
        x += 1
        progress(x=x, n=n)
    _ = f.write(
        '\ncross-sentence-entities\n'
        'group\tPMID\tterm-number\ttype\ti\ttext\tsentence-numbers\n'
    )
    for group in GROUPS:
        pmid2entities = get_pmid2entities(group)
        pmid2sentences = get_pmid2sentences(group)
        cross_sentence_entities = list()
        for pmid, entities in pmid2entities.items():
            sentences = pmid2sentences[pmid]
            for e in entities:
                o = where_overlap(sentences, e['i'])
                if len(o) > 1:
                    cross_sentence_entities.append((
                        pmid,
                        e['tn'],
                        e['type'],
                        e['i'],
                        e['text'],
                        o,
                    ))
        for v in cross_sentence_entities:
            _ = f.write(('{}' + '\t{}' * 6 + '\n').format(group, *v))
        x += 1
        progress(x=x, n=n)
    _ = f.write(
        '\nrelations-with-overlapped-entities\n'
        'group\tPMID\tCPR\teval\tCPI'
        '\targ1-term-number\targ2-term-number'
        '\targ1-type\targ2-type'
        '\targ1-i\targ2-i'
        '\targ1-text\targ2-text\n'
    )
    for group in GROUPS:
        pmid2relations = get_pmid2relations(group)
        relations_with_overlapped_entities = list()
        for pmid, relations in pmid2relations.items():
            for rel in relations:
                if overlap(rel['arg1']['i'], rel['arg2']['i']):
                    relations_with_overlapped_entities.append((
                        pmid,
                        rel['cpr'],
                        rel['eval'],
                        rel['cpi'],
                        rel['arg1']['tn'],
                        rel['arg2']['tn'],
                        rel['arg1']['type'],
                        rel['arg2']['type'],
                        rel['arg1']['i'],
                        rel['arg2']['i'],
                        rel['arg1']['text'],
                        rel['arg2']['text'],
                    ))
        for v in relations_with_overlapped_entities:
            _ = f.write(('{}' + '\t{}' * 12 + '\n').format(group, *v))
        x += 1
        progress(x=x, n=n)
    _ = f.write(
        '\nrelations-with-cross-sentence-entities\n'
        'group\tPMID\tCPR\teval\tCPI'
        '\targ1-term-number\targ2-term-number'
        '\targ1-type\targ2-type'
        '\targ1-i\targ2-i'
        '\targ1-text\targ2-text'
        '\targ1-sentence-numbers\targ2-sentence-numbers\n'
    )
    for group in GROUPS:
        pmid2relations = get_pmid2relations(group)
        pmid2sentences = get_pmid2sentences(group)
        relations_with_cross_sentence_entities = list()
        for pmid, relations in pmid2relations.items():
            sentences = pmid2sentences[pmid]
            for rel in relations:
                sn1 = where_overlap(sentences, rel['arg1']['i'])
                sn2 = where_overlap(sentences, rel['arg2']['i'])
                if (len(sn1) > 1) or (len(sn2) > 1):
                    relations_with_cross_sentence_entities.append((
                        pmid,
                        rel['cpr'],
                        rel['eval'],
                        rel['cpi'],
                        rel['arg1']['tn'],
                        rel['arg2']['tn'],
                        rel['arg1']['type'],
                        rel['arg2']['type'],
                        rel['arg1']['i'],
                        rel['arg2']['i'],
                        rel['arg1']['text'],
                        rel['arg2']['text'],
                        sn1,
                        sn2,
                    ))
        for v in relations_with_cross_sentence_entities:
            _ = f.write(('{}' + '\t{}' * 14 + '\n').format(group, *v))
        x += 1
        progress(x=x, n=n)
    _ = f.write(
        '\noverlapped-relations\ngroup\tPMID'
        'rel1-CPR\trel1-eval\trel1-CPI'
        '\trel1-arg1-term-number\trel1-arg2-term-number'
        '\trel1-arg1-type\trel1-arg2-type'
        '\trel1-arg1-i\trel1-arg2-i'
        '\trel1-arg1-text\trel1-arg2-text'
        '\trel2-CPR\trel2-eval\trel2-CPI'
        '\trel2-arg1-term-number\trel2-arg2-term-number'
        '\trel2-arg1-type\trel2-arg2-type'
        '\trel2-arg1-i\trel2-arg2-i'
        '\trel2-arg1-text\trel2-arg2-text\n'
    )
    for group in GROUPS:
        pmid2relations = get_pmid2relations(group)
        overlapped_relations = list()
        for pmid, relations in pmid2relations.items():
            for i in range(len(relations)):
                for j in range(i + 1, len(relations)):
                    r1 = relations[i]
                    r2 = relations[j]
                    if (
                            overlap(r1['arg1']['i'], r2['arg1']['i']) or
                            overlap(r1['arg1']['i'], r2['arg2']['i']) or
                            overlap(r1['arg2']['i'], r2['arg1']['i']) or
                            overlap(r1['arg2']['i'], r2['arg2']['i'])
                    ):
                        overlapped_relations.append((
                            pmid,
                            r1['cpr'],
                            r1['eval'],
                            r1['cpi'],
                            r1['arg1']['tn'],
                            r1['arg2']['tn'],
                            r1['arg1']['type'],
                            r1['arg2']['type'],
                            r1['arg1']['i'],
                            r1['arg2']['i'],
                            r1['arg1']['text'],
                            r1['arg2']['text'],
                            r2['cpr'],
                            r2['eval'],
                            r2['cpi'],
                            r2['arg1']['tn'],
                            r2['arg2']['tn'],
                            r2['arg1']['type'],
                            r2['arg2']['type'],
                            r2['arg1']['i'],
                            r2['arg2']['i'],
                            r2['arg1']['text'],
                            r2['arg2']['text'],
                        ))
        for v in overlapped_relations:
            _ = f.write(('{}' + '\t{}' * 23 + '\n').format(group, *v))
        x += 1
        progress(x=x, n=n)
    _ = f.write(
        '\ncross-sentence-relations\ngroup\tPMID'
        '\tCPR\teval\tCPI'
        '\targ1-term-number\targ2-term-number'
        '\targ1-type\targ2-type'
        '\targ1-i\targ2-i'
        '\targ1-text\targ2-text'
        '\targ1-sentence-numbers\targ2-sentence-numbers\n'
    )
    for group in GROUPS:
        pmid2relations = get_pmid2relations(group)
        pmid2sentences = get_pmid2sentences(group)
        cross_sentence_relations = list()
        for pmid, relations in pmid2relations.items():
            sentences = pmid2sentences[pmid]
            for rel in relations:
                sn1 = where_overlap(sentences, rel['arg1']['i'])
                sn2 = where_overlap(sentences, rel['arg2']['i'])
                if sn1 != sn2:
                    cross_sentence_relations.append((
                        pmid,
                        rel['cpr'],
                        rel['eval'],
                        rel['cpi'],
                        rel['arg1']['tn'],
                        rel['arg2']['tn'],
                        rel['arg1']['type'],
                        rel['arg2']['type'],
                        rel['arg1']['i'],
                        rel['arg2']['i'],
                        rel['arg1']['text'],
                        rel['arg2']['text'],
                        sn1,
                        sn2,
                    ))
        for v in cross_sentence_relations:
            _ = f.write(('{}' + '\t{}' * 14 + '\n').format(group, *v))
        x += 1
        progress(x=x, n=n)
    _ = f.write('\n')
    f.close()
    progress(clear=True)


def create_sentence_entities():
    r"""
    A file containing the sentence entities is created.

    This function creates a 'sentence_entities.tsv' file inside the
    'support/' directory for each group of the dataset. This
    'sentence_entities.tsv' file contains the term numbers of each
    sentence from a PMID abstract. The `get_pmid2sentences` function is
    used to obtain a mapping between each PMID and the sentence offsets.
    Notes (understanding the output of the `create_statistics` function,
    the 'statistics.tsv' file):
      - Cross-sentence entities are ignored (they do not exist).

    Example
    -------
    >>> from support import create_sentence_entities
    >>> create_sentence_entities()
    >>>

    """
    x = 0
    n = len(GROUPS)
    progress(x=x, n=n)
    for group in GROUPS:
        pmid2entities = get_pmid2entities(group)
        pmid2sentences = get_pmid2sentences(group)
        dp = os.path.join(DATA, 'chemprot_' + group, 'support')
        create_directory(dp)
        fp = os.path.join(dp, 'sentence_entities.tsv')
        pmid2sentence_entities = defaultdict(lambda: defaultdict(list))
        for pmid, sentences in pmid2sentences.items():
            if pmid in pmid2entities:
                for entity in pmid2entities[pmid]:
                    o = where_overlap(sentences, entity['i'])
                    if len(o) == 1:
                        pmid2sentence_entities[pmid][o[0]].append(entity['tn'])
        with open(fp, mode='w', encoding='utf-8') as f:
            for pmid in sorted(pmid2sentence_entities):
                sentence_entities = pmid2sentence_entities[pmid]
                for i in range(len(pmid2sentences[pmid])):
                    tns = sentence_entities[i]
                    if tns:
                        tns = ','.join(tns)
                        _ = f.write('{}\t{}\t{}\n'.format(pmid, i, tns))
        x += 1
        progress(x=x, n=n)
    progress(clear=True)


def create_entity_pairs():
    r"""
    A file containing the entity pairs (in each sentence) is created.

    This function creates a 'entity_pairs.tsv' file inside the
    'support/' directory for each group of the dataset. This
    'entity_pairs.tsv' file contains the entity pairs, with the
    respective sentence numbers, that will be considered as possible CPR
    relations. The `get_pmid2sentence_entities` function is used to
    obtain a mapping between each PMID and the entities (term numbers)
    of each sentence. Notes (understanding the output of the
    `create_statistics` function, the 'statistics.tsv' file):
      - Only relations between a CHEMICAL and a GENE are allowed.
      - The term numbers are sorted in ascending order.
      - Relations between overlapped entities are ignored (goal: to
        reduce the False Positives, improving Precision).
      - Only relations between entities from the same sentence are
        considered.

    Example
    -------
    >>> from support import create_entity_pairs
    >>> create_entity_pairs()
    >>>

    """
    x = 0
    n = len(GROUPS)
    progress(x=x, n=n)
    for group in GROUPS:
        p2e = get_pmid2entities(group)
        p2se = get_pmid2sentence_entities(group)
        dp = os.path.join(DATA, 'chemprot_' + group, 'support')
        create_directory(dp)
        fp = os.path.join(dp, 'entity_pairs.tsv')
        with open(fp, mode='w', encoding='utf-8') as f:
            for p, se in p2se.items():
                for i, entities in enumerate(se):
                    for j in range(len(entities)):
                        for k in range(j + 1, len(entities)):
                            e1 = p2e[p][to_int(entities[j]) - 1]
                            e2 = p2e[p][to_int(entities[k]) - 1]
                            if (
                                (e1['type'][0] != e2['type'][0])
                                and
                                (not overlap(e1['i'], e2['i']))
                            ):
                                _ = f.write('{}\t{}\t{}\t{}\n'.format(
                                    p, i, entities[j], entities[k],
                                ))
        x += 1
        progress(x=x, n=n)
    progress(clear=True)


def evaluate_entity_pairs():
    r"""
    Evaluate the quality of the generated entity pairs.

    This function evaluates the entity pairs 'entity_pairs.tsv' file
    inside the 'support/' directory of all the groups of the dataset.
    These entity pairs are compared to the correct (not gold standard)
    relations. Precision, recall, and F1-score are calculated.
    Notes:
      - This function is used just to have an idea how good is the
        quality of the generated entity pairs.
      - Also it is possible to know if the entity pairs are present.

    Example
    -------
    >>> from support import evaluate_entity_pairs
    >>> evaluate_entity_pairs()
    >>>

    """
    s = '{:>12s} {:>12s} {:>12s} {:>12s}\n'.format(
        '', 'precision', 'recall', 'f1-score')
    for group in GROUPS:
        tp, fp, fn = 0, 0, 0
        p2r = get_pmid2relations(group)
        p2ep = get_pmid2entity_pairs(group)
        for pmid, ep in p2ep.items():
            all_ep = [p for s in ep for p in s]
            if pmid in p2r:
                r = [(r['arg1']['tn'], r['arg2']['tn']) for r in p2r[pmid]]
            else:
                r = list()
            for pair in all_ep:
                if pair in r:
                    tp += 1
                else:
                    fp += 1
            for pair in r:
                if pair not in all_ep:
                    fn += 1
        p = precision(tp, fp)
        r = recall(tp, fn)
        f = f1_score(tp, fp, fn)
        s += '{:>12s} {:>12.6f} {:>12.6f} {:>12.6f}\n'.format(group, p, r, f)
    print(s)


def create_processed_corpus():
    r"""
    Create CHEMPROT scikit-learn compatible processed corpus.

    This function creates a scikit-learn (sklearn) compatible corpus.
    A 'processed_corpus/' directory is created inside the 'support/'
    directory of each dataset group ('training', 'development' and
    'test_gs').
    This corpus contains the CHEMPROT relations split into category
    directories ('CPR:0', ..., 'CPR:10'). Inside each category directory
    there are several files where each file is a sample representing a
    CHEMPROT relation.
    Each relation (sample) is saved into a unique file with the name
    'p_sn_arg1_arg2.txt' (where 'p' is replaced by the PMID, 'sn' is
    replaced by the sentence number, 'arg1' and 'arg2' are the term
    numbers of the interactor arguments) and it has the following
    structure:
      - 1st line: shortest dependency path (SDP): traversed from the
        chemical entity to the gene/protein entity (the head tokens
        [highest headScore - see TEES usage] are considered for creating
        the SDP). Also, a weight is calculated for each dependency/edge
        to select the most important SDP when there are multiple SDPs.
        Features: tokenization, POS tagging, relation dependencies. See
        [1]_, [2]_.
      - 2nd, 3rd, 4th, and 5th lines: the sentence sequence is split
        into four subsequences according to the position of the two
        entities:
          - 2nd line: left text relative to the chemical entity.
          - 3rd line: right text relative to the chemical entity.
          - 4th line: left text relative to the gene entity.
          - 5th line: right text relative to the gene entity.
          - Attention: the start/end of the sentence and the
            chemical/gene entities are considered boundaries of the
            subsequences (e.g., that is, the left text of a chemical
            entity is from the start of the sentence or from the token
            immediately after the gene entity).
          - Visual understanding:
            [chemical_left_text] CHEMICAL [chemical_right_text]
            [gene_left_text] GENE [gene_right_text]
          - Visual example:
            sentence: 'A GENE is not a CHEMICAL found on earth.'
            chemical_left_text: ['is', 'not', 'a']
            chemical_right_text: ['found', 'on', 'earth', '.']
            gene_left_text: ['A']
            gene_right_text: ['is', 'not', 'a']
          - Note: with this approach, the classifier knows the position
            of the text relative to the chemical and gene entities,
            however there is always a repeated subsequence (this is
            important to know the relative positions of the chemical
            and gene entities).
          - Features: tokenization, POS tagging, relation dependencies.
            See [1]_, [3]_.
      - Notes:
        - The tokens, POS, and dependencies are separated using the '|'
          character, since this character does not appear in the corpus,
          neither it is used in the TEES program for representing POS
          tags or relation dependencies.
        - Relatively to the shortest dependency path an empty dependency
          edge '' is added after the last token due to consistency.
        - Relatively to the left/right chemical/gene texts, the incoming
          edges of the tokens if not present are set to empty dependency
          edges ''. If multiple incoming edges exist, the one with
          higher score is chosen.

    References
    ----------
    .. [1] [Matos2017] Extracting Chemical-Protein Interactions using
           Long Short-Term Memory Networks
    .. [2] [Mehryary2017] Combining Support Vector Machines and LSTM
           Networks for Chemical-Protein Relation Extraction
    .. [3] [Zhang2017] Drug-drug interaction extraction via hierarchical
           RNNs on sequence and shortest dependency paths
    .. [4] [Corbett2017] Improving the learning of chemical-protein
           interactions from literature using transfer learning and word
           embeddings
    .. [5] [Peng2017] Chemical-protein relation extraction with
           ensembles of SVM, CNN, and RNN models

    Example
    -------
    >>> from support import create_processed_corpus
    >>> create_processed_corpus()
    >>>

    """
    x = 0
    n = sum(len(get_pmids(group)) for group in GROUPS)
    progress(x=x, n=n)
    for group in GROUPS:
        dp = os.path.join(
            DATA,
            'chemprot_' + group,
            'support',
            'processed_corpus',
        )
        create_directory(dp)
        for cpr in CPR_GROUPS:
            create_directory(os.path.join(dp, cpr))
        pmid2entities = get_pmid2entities(group)
        pmid2relations = get_pmid2relations(group)
        pmid2entity_pairs = get_pmid2entity_pairs(group)
        pmid2tees = get_pmid2tees(group)
        for pmid, entity_pairs in pmid2entity_pairs.items():
            relations = pmid2relations.get(pmid, [])
            for sn, entity_pairs_by_sentence in enumerate(entity_pairs):
                for ep in entity_pairs_by_sentence:
                    cpr = entity_pair_in_relations(ep, relations)
                    fn = '{}_{}_{}_{}.txt'.format(pmid, sn, *ep)
                    fp = os.path.join(dp, cpr, fn)
                    s = get_processed_sample(pmid2tees, pmid2entities,
                        pmid2entity_pairs, pmid, sn, ep)
                    with open(fp, mode='w', encoding='utf-8') as f:
                        _ = f.write(s)
            x += 1
            progress(x=x, n=n)
    progress(clear=True)
