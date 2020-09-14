#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Copyright 2018, 2019, 2020 Rui Antunes, Sérgio Matos

https://github.com/ruiantunes/biocreative-vi-track-5-chemprot


Utilities.


Routines listing
----------------
clear
reset
progress
create_directory
str2bool
str2line


Classes listing
---------------
Printer

"""

# third-party modules
import datetime
import os


def clear():
    r"""
    This function clears the console.

    Example
    -------
    >>> from utils import clear
    >>> clear()
    >>>

    """
    os.system('cls' if os.name == 'nt' else 'clear')


def reset():
    r"""
    This function resets the console.

    Example
    -------
    >>> from utils import reset
    >>> reset()
    >>>

    """
    os.system('cls' if os.name == 'nt' else 'reset')


def progress(x=0, n=None, clear=False):
    r"""
    Shows the progress.

    This function shows the progress in percentage of a piece of code.
    The input parameters `x` and `n` specify respectively the current
    iteration and the total number of iterations. If `n` is considered
    `False` then only the number of the current iteration is shown. If
    `clear` is `True` then the progress message is erased.

    Parameters
    ----------
    x : int, optional
        Current iteration. Default: 0.
    n : int, optional
        Total number of iterations. Default: `None` (percentage is not
        shown).
    clear : bool, optional
        If `True`, the progress message is erased. Default: `False`.

    Example
    -------
    >>> from utils import progress
    >>> import time
    >>> n = 10
    >>> for i in range(1, n + 1):
    ...   time.sleep(0.5)
    ...   progress(i, n)
    ...

    """
    print('\r\x1b[K', end='')
    if not clear:
        if n:
            print('{:6.2f}%'.format(100 * x / n), end='', flush=True)
        else:
            print('{:d}'.format(x), end='', flush=True)


def create_directory(path):
    r"""
    Constructs (recursively) a directory.

    This function creates a specific directory. If the directory already
    exists as a directory nothing is made. Otherwise, if the path
    specified exists and is not a directory an exception is raised.

    Parameters
    ----------
    path : str
        Directory to create. Consecutive folders are allowed, e.g.
        'path/to/new/directory'. If `path` already exists but is not a
        directory, an `AssertionError` is raised.

    Raises
    ------
    AssertionError
        If `path` already exists and is not a directory.

    Example
    -------
    >>> from utils import create_directory
    >>> import os
    >>> create_directory(os.path.join('insert', 'here', 'the', 'path'))
    >>>

    """
    if os.path.exists(path):
        e = '{} already exists and is not a directory!'.format(repr(path))
        assert os.path.isdir(path), e
    else:
        os.makedirs(path)


def str2bool(s):
    r"""
    Converts a `str` value to a `bool` value.

    This function is useful to be used in the argument parsing
    `argparse` module to support `bool` types [1]_.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    Parameters
    ----------
    s : str
        Input `str` to convert to `bool`.

    Returns
    -------
    b : bool
        Output `bool`.

    Raises
    ------
    AssertionError
        An error occurs when it is not possible to convert a `str` value
        to a `bool` value.

    Example
    -------
    >>> from utils import str2bool
    >>> str2bool('True')
    True
    >>> str2bool('0')
    False
    >>> str2bool('y')
    True
    >>>

    """
    s = s.strip().lower()
    if s in ('true', 't', 'yes', 'y', '1'):
        return True
    elif s in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise AssertionError('Boolean value expected.')


def str2line(s):
    r"""
    Converts white-space characters to spaces.

    Parameters
    ----------
    s : str
        String to be converted to a string without lines. The string is
        stripped.

    Returns
    -------
    s : str
        String without newlines.

    Example
    -------
    >>> from utils import str2line
    >>> s = 'This is\tsome\nstring.'
    >>> str2line(s)
    'This is some string.'
    >>>

    """
    return ' '.join(str(s).split())


class Printer:
    r"""
    Robust and flush-forced printing (date information can be added).

    This class handles printing. The purpose of this class is to print
    the information to the console (to one can see the progress
    information of a script in real-time) and simulteanously to save it
    to a file. This is useful to store outputs/results of a program.
    Notes:
    1. flush is `True` in the `print` function internal usage, so new
       prints in a same line will be instantaneously printed into the
       console.
    2. Only writing text is accepted (binary mode is not accepted).
    3. UTF-8 encoding is used.

    Attributes
    ----------
    _filepath
    _overwrite
    _sep
    _time_format

    Methods listing
    ---------------
    __init__
    print
    date

    Example
    -------
    >>> from utils import Printer
    >>> p = Printer()
    >>> p.date('Print: console.')
    12:29:04.863692 2018/Mar/17	Print: console.
    >>> p = Printer('test.log', overwrite=True)
    >>> p.date('Print: console and file.')
    12:29:07.434040 2018/Mar/17	Print: console and file.
    >>> p.date('The 1st line.\nThe 2nd line.')
    12:29:09.735111 2018/Mar/17	The 1st line.
    The 2nd line.
    >>> p.date('A new line is (implicitly) added.')
    12:29:11.728569 2018/Mar/17	A new line is (implicitly) added.
    >>> p.date('A new line is (explicitly) added.\n', end='')
    12:29:13.282992 2018/Mar/17	A new line is (explicitly) added.
    >>>

    """
    #
    def __init__(self, filepath=None, overwrite=False, sep='\t'):
        r"""
        Constructor method. It initializes the object.

        Parameters
        ----------
        filepath : str, optional
            The filepath to save the prints. If `filepath` is considered
            `False`, then the prints are not saved to any file. Default:
            `None`.
        overwrite : bool, optional
            If `True` and the `filepath` specifies a file, that file is
            overwritten. If the file exists but `overwrite` is `False`,
            then an `AssertionError` is raised. Default: `False`.
        sep : str, optional
            If it is the case, this separator is used to separate the
            date and the message. Default: '\t' (tab).

        Raises
        ------
        AssertionError
            If `filepath` exists but is not a file. Or if `filepath` is
            already a file but `overwrite` is `False`.

        """
        if filepath:
            if os.path.exists(filepath):
                e = '{} is not a file!'.format(repr(filepath))
                assert os.path.isfile(filepath), e
                e = '{} cannot be overwritten!'.format(repr(filepath))
                assert overwrite, e
            else:
                dpath = os.path.split(filepath)[0]
                if dpath:
                    create_directory(dpath)
            with open(filepath, mode='w', encoding='utf-8') as f:
                _ = f.write('')
        self._filepath = filepath
        self._overwrite = overwrite
        self._sep = sep
        self._time_format = '%H:%M:%S.%f %Y/%b/%d'
    #
    def print(self, s='', end='\n', date=False):
        r"""
        Prints the message `s` (date can be added).

        Parameters
        ----------
        s : str, optional
            String to print. Default: `''`.
        end : str, optional
            String to append to the end of the message. Default: '\n'.
        date : bool, optional
            If `True` the current date is printed followed by the
            separator `sep` and the message `s`. Default: `False`.

        """
        if date:
            now = datetime.datetime.now().strftime(self._time_format)
            s = now + self._sep + s
        s += end
        print(s, end='', flush=True)
        if self._filepath:
            with open(self._filepath, mode='a', encoding='utf-8') as f:
                _ = f.write(s)
    #
    def date(self, s='', end='\n'):
        r"""
        Prints the current date and the message `s`.

        Parameters
        ----------
        s : str, optional
            String to print. Default: `''`.
        end : str, optional
            String to append to the end of the message. Default: '\n'.

        """
        self.print(s, end=end, date=True)
