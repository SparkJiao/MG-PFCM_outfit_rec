import glob
import json
import os
from collections import defaultdict
from typing import Dict, Any, Union, Callable

import dgl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from general_util.logger import get_child_logger

logger = get_child_logger('Dataset')


class SubgraphDataset(Dataset):
    def __init__(self, quadruple_file: str, meta_path_dict: DictConfig, graph_sampler: Callable = None, max_tuple_num: int = 100):
        logger.info(f'Loading data file from {quadruple_file}.')
        self.quadruples = json.load(open(quadruple_file, 'r'))
        self.meta_path = self._parse_meta_path(meta_path_dict)
        self.graph_sampler = graph_sampler
        self.max_tuple_num = max_tuple_num

    def __getitem__(self, index) -> T_co:
        # user, anchor_item, pos_item, neg_item = self.quadruples[index]
        # quadruple = [user, anchor_item, pos_item, neg_item]
        quadruple = self.quadruples[index][:self.max_tuple_num]

        all_nodes = set()
        all_dgl_graph, all_src, all_dst, all_node2re_id, all_re_id2node = [], [], [], [], []

        for i, x in enumerate(quadruple):  # [user, anchor_item, pos_item, neg_item]
            # _dgl_graph_ls, _mapped_src_ls, _mapped_dst_ls, _node2re_id_ls, _re_id2node_ls, _nodes_ls = zip(*[
            #     self._load_subgraph(src=x, graph=y) for y in self.meta_path[x]])
            _dgl_graph_ls, _mapped_src_ls, _mapped_dst_ls, _node2re_id_ls, _re_id2node_ls, _nodes_ls = [], [], [], [], [], []
            res_ls = tuple([self._load_subgraph(src=x, graph=y) for y in self.meta_path[x]])
            # _dgl_graph_ls, _mapped_src_ls, _mapped_dst_ls, _node2re_id_ls, _re_id2node_ls, _nodes_ls = zip(*res_ls)
            for _res in res_ls:
                _dgl_graph_ls.append(_res[0])
                _mapped_src_ls.append(_res[1])
                _mapped_dst_ls.append(_res[2])
                _node2re_id_ls.append(_res[3])
                _re_id2node_ls.append(_res[4])
                _nodes_ls.append(_res[5])
            for subgraph_node_ls in _nodes_ls:
                all_nodes.update(subgraph_node_ls)
            all_dgl_graph.append(_dgl_graph_ls)
            all_src.append(_mapped_src_ls)
            all_dst.append(_mapped_dst_ls)
            all_node2re_id.append(_node2re_id_ls)
            all_re_id2node.append(_re_id2node_ls)

        return all_dgl_graph, all_node2re_id, all_re_id2node, list(all_nodes), quadruple

    def __len__(self):
        return len(self.quadruples)

    @staticmethod
    def _parse_meta_path(meta_path_dict: DictConfig):
        logger.info(f'Parsing meta-path...')
        meta_path = defaultdict(list)
        for path_type, path_no_path in meta_path_dict.items():
            if os.path.isfile(path_no_path):  # All subgraphs are saved into single file.
                path_subgraph = torch.load(path_no_path)
                assert path_type == path_subgraph[0]['meta_path'], (path_type, path_subgraph[0]['meta_path'], path_subgraph[1]['meta_path'])
                for _subgraph in path_subgraph:
                    meta_path[_subgraph["src_id"]].append(_subgraph)
            else:
                files = list(glob.glob(path_no_path))
                for file in files:
                    src = file.split('/')[-1]
                    meta_path[src].append(file)
        return meta_path

    def _load_subgraph(self, src: str, graph: Union[Dict[str, Any], str]):
        if isinstance(graph, str):  # Load the single subgraph file.
            graph: Dict[str, Any] = torch.load(graph)

        if self.graph_sampler is not None:
            graph = self.graph_sampler(graph)

        meta_path_type = graph['meta_path']
        assert graph['src_id'] == src
        neighbours = graph['edges']

        orig_edges = []
        # =======================================================
        # 在``__init__``方法里已经保证了不同种类的节点的id不会出现重复，因此直接合并即可，无需在区分种类。节点之间的位置顺序也不重要
        # 但我们需要按照``nodes``里的节点顺序初始化节点的embedding序列。
        # =======================================================
        # Since we have made sure that the id of each node is unique among all nodes across different node types in the ``__init__`` method,
        # and the relative order among all the nodes does not matter, either,
        # we can directly merge all nodes from ``neighbours`` without considering their node types.
        # However, we should keep the same order with that in ``nodes`` with the input node embedding list.
        nodes = set()
        for u, v_set in neighbours.items():
            for v in v_set:
                orig_edges.append((u, v))
                nodes.add(u)
                nodes.add(v)
        nodes = list(nodes)

        node2re_id = {}
        re_id2node = {}
        for i, node in enumerate(nodes):
            node2re_id[node] = i
            re_id2node[i] = node

        mapped_src = []
        mapped_dst = []
        for e in orig_edges:
            mapped_src.append(node2re_id[e[0]])
            mapped_dst.append(node2re_id[e[1]])

        # You can initialize the DGL graph from here
        dgl_graph = dgl.graph((torch.tensor(mapped_src + mapped_dst), torch.tensor(mapped_dst + mapped_src)))
        # TODO:
        #   Any other graphs? e.g., relation based edges?
        #   As a result, we may use different types of initialization for DGL graph.
        # print(dgl_graph, mapped_src, mapped_dst, node2re_id, re_id2node, nodes, '\n')
        return dgl_graph, mapped_src, mapped_dst, node2re_id, re_id2node, nodes
