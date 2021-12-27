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


def _initializer(e: Dict[str, Dict[str, Set]]):
    global _edges
    _edges = e


def bfs(_u: str, _pattern: str):
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

        nxt_set = edges[cur_n_tpe][nxt_n_tpe][cur_n]

        for nxt in nxt_set:
            if nxt in neighbors and cur_n in neighbors[nxt]:  # No backtracking.
                continue
            neighbors[cur_n].add(nxt)
            if nxt not in vis_set[nxt_n_tpe]:
                queue.append((p_rank + 1, nxt))
                vis_set[nxt_n_tpe].add(nxt)

        # neighbors[cur_n].update(nxt_sub_set)
        # vis_set[nxt_n_tpe].update(nxt_sub_set)
        # for nxt in nxt_sub_set:
        #     queue.append((p_rank + 1, nxt))

    if len(neighbors[_u]) == 0 or all(len(s) == 0 for s in neighbors[_u]):
        return None
    return _u, neighbors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()

    ia = json.load(open(os.path.join(args.data_dir, 'dict/IA.json'), 'r'))
    ai = json.load(open(os.path.join(args.data_dir, 'dict/AI.json'), 'r'))
    io = json.load(open(os.path.join(args.data_dir, 'dict/IO.json'), 'r'))
    oi = json.load(open(os.path.join(args.data_dir, 'dict/OI.json'), 'r'))
    ou = json.load(open(os.path.join(args.data_dir, 'dict/OU.json'), 'r'))
    uo = json.load(open(os.path.join(args.data_dir, 'dict/UO.json'), 'r'))

    ii = defaultdict(set)
    for item_id_1, o_ls in io.items():
        for o_id in o_ls:
            if o_id in oi:
                for item_id_2 in oi[o_id]:
                    if item_id_2 != item_id_1:
                        ii[item_id_1].add(item_id_2)
                        ii[item_id_2].add(item_id_1)

    ui = defaultdict(set)
    for user_id, o_ls in uo.items():
        for o_id in o_ls:
            if o_id in oi:
                for item_id in oi[o_id]:
                    ui[user_id].add(item_id)

    iu = defaultdict(set)
    for item_id, o_ls in io.items():
        for o_id in o_ls:
            if o_id in ou:
                for user_id in ou[o_id]:
                    iu[item_id].add(user_id)

    edges = defaultdict(dict)
    # ia = {item_id: set(a_ls) for item_id, a_ls in ia.items()}
    ia = {item_id: set(filter(lambda x: x[:2] != '2_', a_ls)) for item_id, a_ls in ia.items()}  # Remove ``brand`` attribute.
    # ai = {a_id: set(item_ls) for a_id, item_ls in ai.items()}
    ai = {a_id: set(item_ls) for a_id, item_ls in ai.items() if a_id[:2] != '2_'}  # Remove ``brand`` attribute.

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

    all_nodes = {
        'u': user_ids,
        'i': item_ids,
        'a': attr_ids
    }
    print(f"Node num:\nAttribute: {len(attr_ids)}\tUser: {len(user_ids)}\tItem: {len(item_ids)}")

    patterns = {
        'u': [
            # 'uiaiu',  # uiaiu 可能和 uiu有重复
            # 'uiu',
            # 'uia'
        ],
        'i': [
            'iai',
            # 'iui',
            # 'ii',
            # 'iia'
        ]
    }

    for n_type, n_id_ls in all_nodes.items():
        if n_type not in patterns:
            continue
        pattern_ls = patterns[n_type]
        for ptn in pattern_ls:
            sub_len = len(n_id_ls) // 12
            for idx in range(12):
                node_id_ls = list(n_id_ls)[(idx * sub_len): ((idx + 1) * sub_len)]
                print(f"Processing meta-path [{ptn}]...")
                with Pool(args.num_workers, initializer=_initializer, initargs=(edges,)) as p:
                    _annotate = partial(bfs, _pattern=ptn)
                    _results = list(tqdm(
                        p.imap(_annotate, node_id_ls, chunksize=32),
                        total=len(node_id_ls),
                        desc="BFS searching"
                    ))

                sub_graphs = []
                for sub_graph in _results:
                    if sub_graph is not None:
                        sub_graphs.append({
                            'n_type': n_type,
                            'src_id': sub_graph[0],
                            'edges': sub_graph[1]
                        })
                print(f"Processed {len(sub_graphs)} sub-graphs.")
                torch.save(sub_graphs, args.output_file.replace('.feat', f'.{ptn}.{idx}.feat'))
                del sub_graphs
                del _results
    print("Done.")
