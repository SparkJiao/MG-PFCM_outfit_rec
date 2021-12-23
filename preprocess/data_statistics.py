import json
import copy
import os
from multiprocessing import Pool
from collections import defaultdict
import argparse
from functools import partial
from tqdm import tqdm


def _initializer(_pattern, _edges, _vis_set):
    global __pattern__
    global __edges__
    global __vis_set__
    __pattern__ = _pattern
    __edges__ = _edges
    __vis_set__ = _vis_set


def empty_vis(_vis_set):
    for k in _vis_set.keys():
        _vis_set[k].empty()


def dfs(_pattern, _vis_set, _src_n):
    if len(_pattern) == 1:
        return 1

    _src = _pattern[0]
    rest_pattern = _pattern[1:]
    tgt = rest_pattern[0]

    _e_type = _src + tgt

    _cur_cnt = 0
    if _src_n not in __edges__[_e_type]:
        return 0

    for tgt_n in __edges__[_e_type][_src_n]:
        if tgt_n not in _vis_set[tgt]:
            # _nxt_vis_set = copy.deepcopy(_vis_set)
            # _nxt_vis_set[tgt].add(tgt_n)
            _vis_set[tgt].add(tgt_n)
            _cur_cnt += dfs(rest_pattern, _vis_set, tgt_n)
            _vis_set[tgt].remove(tgt_n)

    return _cur_cnt


def outer_for(_src_n):
    _nxt_vis_set = copy.deepcopy(__vis_set__)
    # _nxt_vis_set = {k: set() for k in ['a', 'i', 'o', 'u']}
    _nxt_vis_set[__pattern__[0]].add(_src_n)
    return dfs(__pattern__, _nxt_vis_set, _src_n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()

    ia = json.load(open(os.path.join(args.data_dir, 'dict/IA.json'), 'r'))
    ai = json.load(open(os.path.join(args.data_dir, 'dict/AI.json'), 'r'))
    io = json.load(open(os.path.join(args.data_dir, 'dict/IO.json'), 'r'))
    oi = json.load(open(os.path.join(args.data_dir, 'dict/OI.json'), 'r'))
    ou = json.load(open(os.path.join(args.data_dir, 'dict/OU.json'), 'r'))
    uo = json.load(open(os.path.join(args.data_dir, 'dict/UO.json'), 'r'))

    edges = defaultdict(dict)
    edges['ia'] = ia
    edges['ai'] = ai
    edges['io'] = io
    edges['oi'] = oi
    edges['ou'] = ou
    edges['uo'] = uo

    node_type = ['a', 'i', 'o', 'u']
    vis_set = {k: set() for k in node_type}

    # path_pattern = ['iai', 'ioi', 'iouoi', 'aia', 'aioia', 'aiouoia']
    # path_pattern = ['aioia', 'aiouoia']
    # path_pattern = ['iouoi', 'aia']
    # path_pattern = ['ioi']
    # path_pattern = ['iai']
    # aioia: 11376496
    # aiouoia: 4493559384
    # iouoi: 139248990
    # aia: 4279406
    # ioi: 351042
    # iai: 24757313232
    path_pattern = ['uoiaiou', 'oiaio']

    for path_p in path_pattern:
        print(path_p)
        src = path_p[0]
        src_n_set = set()
        for e_type in edges.keys():
            if e_type[0] == src:
                src_n_set.update(edges[e_type].keys())
        src_n_ls = list(src_n_set)
        with Pool(1, initializer=_initializer, initargs=(path_p, edges, vis_set)) as p:
            _annotate = partial(outer_for)
            _results = list(tqdm(
                p.imap(_annotate, src_n_ls, chunksize=32),
                total=len(src_n_ls),
                desc="Reading examples"
            ))
        # _initializer(path_p, edges, vis_set)
        # _results = [(outer_for(x)) for x in tqdm(src_n_ls)]

        res = sum(_results)
        print(res)
