import json

import dgl
import torch

from general_util.logger import get_child_logger
from data_loader.data_utils import EmbeddingMatrix

logger = get_child_logger('SubgraphCollatorVocab')


class SubgraphCollatorVocab:
    def __init__(self,
                 user_vocab: str,
                 attr_vocab: str,
                 item_vocab: str,
                 node_vocab: str,
                 embedding: EmbeddingMatrix):
        self.user_vocab = json.load(open(user_vocab, 'r'))
        self.attr_vocab = json.load(open(attr_vocab, 'r'))
        self.item_vocab = json.load(open(item_vocab, 'r'))
        self.node_vocab = torch.load(node_vocab)
        self.node2type = {}
        for k, v_ls in self.node_vocab.items():
            for v in v_ls:
                if v not in self.node2type:
                    self.node2type[v] = k
                else:
                    assert self.node2type[v] == k, (self.node2type[v], k)  # Check repetition.
        assert 'u' in self.node_vocab
        assert 'i' in self.node_vocab
        assert 'a' in self.node_vocab
        self.embedding = embedding

    def __call__(self, batch):
        """
        :param batch:
            all_dgl_graph: [batch, quadruple_num == 4, subgraph_num], List[List[List[dgl.graph]]]
            all_node2re_id: [batch, quadruple_num == 4, subgraph_num], List[List[List[Dict[str, int]]]]
            all_re_id2node: [batch, quadruple_num == 4, subgraph_num], List[List[List[Dict[int, str]]]]
            all_nodes: [batch], List[str]
            all_quadruples: [batch, quadruple_num == 4], List[List[str]]

        :return:
            graph (dgl.graph):
                Union graph through ``dgl.batch(List[dgl.graph])``.
            input_emb_index: (torch.Tensor):
                Size: [input_node_num]
                Concatenated embedding indices, by which the input embedding can be gathered from different modalities.
            src_index (torch.Tensor):
                Size: [batch, quadruple_num, max_subgraph_num]
                The source nodes index among the input node sequence.
            sub_graph_mask (torch.Tensor):
                Size: [batch, quadruple_num, max_subgraph_num]
                The mask of the input subgraphs, for transformer encoding.
            item_emb_index (torch.Tensor):
                ...
            attr_emb_index (torch.Tensor):
                ...
            user_emb_index (torch.Tensor):
                ...
        """
        all_dgl_graph, all_node2re_id, all_re_id2node, all_nodes, all_quadruples = zip(*batch)

        batch_size = len(all_dgl_graph)
        _nodes = set()
        _node2emb = {}
        max_subgraph_num = 0
        for b in range(batch_size):
            _nodes.update(all_nodes[b])
            max_subgraph_num = max(max_subgraph_num, max(map(lambda x: len(x), all_dgl_graph[b])))

        users = []
        user_emb_index = []
        items = []
        item_emb_index = []
        attributes = []
        attr_emb_index = []
        for _node in _nodes:
            _node_type = self.node2type[_node]
            if _node_type == 'i':
                items.append(_node)
                item_emb_index.append(self.item_vocab[_node])
            elif _node_type == 'a':
                attributes.append(_node)
                attr_emb_index.append(self.attr_vocab[_node])
            elif _node_type == 'u':
                users.append(_node)
                user_emb_index.append(self.user_vocab[_node])
            else:
                raise RuntimeError(f"Unrecognized node type: {_node_type}.")

        item_emb_index = torch.tensor(item_emb_index, dtype=torch.long)
        attr_emb_index = torch.tensor(attr_emb_index, dtype=torch.long)
        user_emb_index = torch.tensor(user_emb_index, dtype=torch.long)

        node2emb_index = {}
        for i, _node in enumerate(items + attributes + users):
            node2emb_index[_node] = i

        quadruple_num = len(all_dgl_graph[0])
        assert quadruple_num == len(all_node2re_id[0]) == len(all_re_id2node[0]), (quadruple_num, len(all_node2re_id[0]),
                                                                                   len(all_re_id2node[0]))

        # index of source node of each subgraph, used for gathering the corresponding hidden states.
        src_index = torch.zeros(batch_size, quadruple_num, max_subgraph_num, dtype=torch.long)
        subgraph_mask = torch.zeros(batch_size, quadruple_num, max_subgraph_num, dtype=torch.long)
        input_emb_index = []
        graphs = []

        for b in range(batch_size):
            for t in range(quadruple_num):
                for g, (subgraph_re_id2node, subgraph_node2re_id) in enumerate(zip(all_re_id2node[b][t], all_node2re_id[b][t])):
                    src_index[b, t, g] = subgraph_node2re_id[all_quadruples[b][t]] + len(input_emb_index)  # offset
                    subgraph_mask[b, t, g] = 1
                    sg_node_num = len(subgraph_re_id2node)
                    sg_mapped_emb_index = list(map(lambda x: node2emb_index[subgraph_re_id2node[x]], range(sg_node_num)))
                    input_emb_index.extend(sg_mapped_emb_index)
                graphs.extend(all_dgl_graph[b][t])
        graph = dgl.batch(graphs)
        input_emb_index = torch.tensor(input_emb_index, dtype=torch.long)

        return {
            "graph": graph,
            "input_emb_index": input_emb_index,
            "src_index": src_index,
            "subgraph_mask": subgraph_mask,
            # "item_emb_index": item_emb_index,
            # "attr_emb_index": attr_emb_index,
            # "user_emb_index": user_emb_index
            "item_image": torch.index_select(self.embedding.item_image, dim=0, index=item_emb_index),
            "item_text": torch.index_select(self.embedding.item_text, dim=0, index=item_emb_index),
            "attr_text": torch.index_select(self.embedding.attr_text, dim=0, index=attr_emb_index),
            "user_emb_index": user_emb_index
        }
