import json
from typing import Dict, List, Union, Tuple
import os

import dgl
import torch
from torch import Tensor
from omegaconf import DictConfig


class DataCollatorBase:
    def __init__(self, node_vocab: Dict[str, List],
                 ui_edge_file: str,
                 emb_path_dic: DictConfig):
        self.node_vocab = node_vocab
        self.node2type = {}
        for k, v_ls in node_vocab.items():
            for v in v_ls:
                if v not in self.node2type:
                    self.node2type[v] = k
                else:
                    assert self.node2type[v] == k, (self.node2type[v], k)  # Check repetition.
        assert 'u' in self.node_vocab
        assert 'i' in self.node_vocab
        assert 'a' in self.node_vocab
        self.ui_edges: Dict[str, List[str]] = json.load(open(ui_edge_file, 'r'))
        self.emb_path_dic = emb_path_dic

    def load_embedding(self, node) -> Union[Tensor, Tuple[Tensor, ...]]:
        node_type = self.node2type[node]
        if node_type == 'a':
            attr = torch.load(os.path.join(self.emb_path_dic['a'], f'{node}.pt'))
            return attr
        elif node_type == 'i':
            text = torch.load(os.path.join(self.emb_path_dic['text'], f'{node}_t.pt'))
            mask = torch.load(os.path.join(self.emb_path_dic['mask'], f'{node}_mask.pt'))
            image = torch.load(os.path.join(self.emb_path_dic['image'], f'{node}_v.pt'))
            return image, text, mask
        else:
            raise RuntimeError(f'Unrecognized node and node type: {node}, {node_type}.')

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
