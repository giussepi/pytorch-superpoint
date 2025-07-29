# -*- coding: utf-8 -*-
""" utils/dict_ops.py """

from copy import deepcopy

import torch


__all__ = [
    'dict_sum',
    'dict_div_by_scalar',
    'dict_only_primitives',
]


def dict_sum(d1: dict, d2: dict) -> dict:
    assert isinstance(d1, dict), type(d1)
    assert isinstance(d2, dict), type(d2)

    summation = deepcopy(d1)

    for k, v in d2.items():
        if k in summation:
            summation[k] += v
        else:
            summation[k] = v

    return summation


# assert dict_sum(a, b) == {'a': 11, 'b': 22, 'c': 3, 'd': 40}


def dict_div_by_scalar(d: dict, denominator: int) -> dict:
    assert isinstance(d, dict), type(d)
    assert isinstance(denominator, int), type(denominator)
    assert denominator != 0, denominator

    division = deepcopy(d)

    for k in division.keys():
        division[k] /= denominator

    return division


# assert dict_div_by_scalar(b, 10) == {'a': 1, 'b': 2, 'd': 4}


def dict_only_primitives(d: dict, cleaned: dict = None) -> dict:
    assert isinstance(d, dict), type(d)

    cleaned = {}

    for k, v in d.items():
        if isinstance(v, dict):
            cleaned[k] = dict_only_primitives(v)
        elif isinstance(v, torch.Tensor):
            cleaned[k] = v.item()
        else:
            cleaned[k] = v

    return cleaned


# assert dict_only_primitives({'a': torch.tensor(1), 'b': torch.tensor(2)}) == {'a': 1, 'b': 2}
