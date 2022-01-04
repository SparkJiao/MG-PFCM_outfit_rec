import torch
import argparse
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
import os
from typing import Dict, Set
import json
from tqdm import tqdm

_edges: Dict[str, Dict[str, Set]]


def _initializer(e: Dict[str, Dict[str, Dict[str, Set]]]):
    global _edges
    _edges = e


def bfs(_u: str, _pattern: str, save: bool = False, save_dir: str = None):
    neighbors = defaultdict(set)
    vis_set = defaultdict(set)

    """
    Checklist:
        1.  Each type of nodes can be only connected with the following and preceding types of nodes in the meta-path.
        2.  Each node can be connected with multiple preceding or following nodes of the same type.
    """

    queue = [(0, _u)]
    vis_set[_pattern[0]].add(_u)
    while len(queue) > 0:
        p_rank, cur_n = queue.pop(0)
        if p_rank == len(_pattern) - 1:
            break
        cur_n_tpe = _pattern[p_rank]
        nxt_n_tpe = _pattern[p_rank + 1]

        nxt_set = _edges[cur_n_tpe][nxt_n_tpe][cur_n]

        for nxt in nxt_set:
            if nxt in neighbors and cur_n in neighbors[nxt]:  # No backtracking.
                continue
            neighbors[cur_n].add(nxt)
            if nxt not in vis_set[nxt_n_tpe]:
                queue.append((p_rank + 1, nxt))
                vis_set[nxt_n_tpe].add(nxt)

    if len(neighbors[_u]) == 0 or all(len(s) == 0 for s in neighbors[_u]):
        return None
    if save and save_dir is not None:
        obj = {
            'meta_path': _pattern,
            'src_id': _u,
            'edges': neighbors
        }
        torch.save(obj, os.path.join(save_dir, str(_u)))
    return _u, neighbors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=32)
    # parser.add_argument('--output_file', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    ia = json.load(open(os.path.join(args.data_dir, 'IA.json'), 'r'))
    ai = json.load(open(os.path.join(args.data_dir, 'AI.json'), 'r'))
    ii = json.load(open(os.path.join(args.data_dir, 'II.json'), 'r'))
    iu = json.load(open(os.path.join(args.data_dir, 'IU.json'), 'r'))
    ui = json.load(open(os.path.join(args.data_dir, 'UI.json'), 'r'))

    edges = defaultdict(dict)

    edges['i']['a'] = ia
    edges['a']['i'] = ai
    edges['i']['u'] = iu
    edges['u']['i'] = ui
    edges['i']['i'] = ii
    print(f"Length:\nIA: {len(ia)}\tAI: {len(ai)}\tIU: {len(iu)}\tUI: {len(ui)}\tII:{len(ii)}")

    user_ids = set()
    item_ids = set()
    attr_ids = set()
    for i_id, a_ls in ia.items():
        item_ids.add(i_id)
        attr_ids.update(a_ls)
    for i_id, u_ls in iu.items():
        item_ids.add(i_id)
        user_ids.update(u_ls)
    for i_id, i_ls in ii.items():
        item_ids.add(i_id)
        item_ids.update(i_ls)
    for a_id, i_ls in ai.items():
        attr_ids.add(a_id)
        item_ids.update(i_ls)
    for u_id, i_ls in ui.items():
        user_ids.add(u_id)
        item_ids.update(i_ls)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_nodes = {
        'u': list(user_ids),
        'i': list(item_ids),
        'a': list(attr_ids)
    }
    print(f"Node num:\nAttribute: {len(attr_ids)}\tUser: {len(user_ids)}\tItem: {len(item_ids)}")
    torch.save(all_nodes, os.path.join(args.output_dir, 'vocab.pt'))

    patterns = {
        'u': [
            'uiaiu',  # uiaiu 可能和 uiu有重复
            'uiu',
            'uia'
        ],
        'i': [
            # 'iai',
            'iui',
            'ii',
            'iia'
        ]
    }

    # for n_type, n_id_ls in all_nodes.items():
    #     if n_type not in patterns:
    #         continue
    #     pattern_ls = patterns[n_type]
    #     for ptn in pattern_ls:
    #         print(f"Processing meta-path [{ptn}]...")
    #         with Pool(args.num_workers, initializer=_initializer, initargs=(edges,)) as p:
    #             _annotate = partial(bfs, _pattern=ptn)
    #             _results = list(tqdm(
    #                 p.imap(_annotate, n_id_ls, chunksize=32),
    #                 total=len(n_id_ls),
    #                 desc="BFS searching"
    #             ))
    #
    #         sub_graphs = []
    #         for sub_graph in _results:
    #             if sub_graph is not None:
    #                 sub_graphs.append({
    #                     'meta_path': ptn,
    #                     'src_id': sub_graph[0],
    #                     'edges': sub_graph[1]
    #                 })
    #         print(f"Processed {len(sub_graphs)} sub-graphs.")
    #         torch.save(sub_graphs, args.output_file + f'.{ptn}')
    #         del sub_graphs
    #         del _results

    # uiaiu_output_dir = os.path.join(args.output_dir, 'subgraph-uiaiu')
    # if not os.path.exists(uiaiu_output_dir):
    #     os.makedirs(uiaiu_output_dir)
    # with Pool(args.num_workers, initializer=_initializer, initargs=(edges,)) as p:
    #     _annotate = partial(bfs, _pattern='uiaiu', save=True, save_dir=uiaiu_output_dir)
    #     _results = list(tqdm(
    #         p.imap(_annotate, all_nodes['u'], chunksize=32),
    #         total=len(all_nodes['u']),
    #         desc="BFS searching"
    #     ))
    # del _results

    # 单独处理 ``iai`` 因为数量太大了会爆内存
    chunk_num = 12
    chunk_size = len(all_nodes['i']) // chunk_num
    iai_output_dir = os.path.join(args.output_dir, 'subgraph-iai')
    if not os.path.exists(iai_output_dir):
        os.makedirs(iai_output_dir)
    for idx in range(chunk_num):
        sub_node_ls = all_nodes['i'][(idx * chunk_size): ((idx + 1) * chunk_size)]
        with Pool(args.num_workers, initializer=_initializer, initargs=(edges,)) as p:
            _annotate = partial(bfs, _pattern='iai', save=True, save_dir=iai_output_dir)
            _results = list(tqdm(
                p.imap(_annotate, sub_node_ls, chunksize=32),
                total=len(sub_node_ls),
                desc="BFS searching"
            ))
        del _results
    print("Done.")
