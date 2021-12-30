from typing import Dict, List

import dgl
import torch
from omegaconf import DictConfig

from data_collator_base import DataCollatorBase


class SubgraphCollator(DataCollatorBase):
    def __init__(self, node_vocab: str,
                 ui_edge_file: str,
                 emb_path_dic: DictConfig):
        super().__init__(node_vocab=node_vocab,
                         ui_edge_file=ui_edge_file,
                         emb_path_dic=emb_path_dic)

    def __call__(self, batch):
        """
        :param batch:
            all_dgl_graph: [batch, quadruple_num == 4, subgraph_num], List[List[List[dgl.graph]]]
            all_node2re_id: [batch, quadruple_num == 4, subgraph_num], List[List[List[Dict[str, int]]]]
            all_re_id2node: [batch, quadruple_num == 4, subgraph_num], List[List[List[Dict[int, str]]]]
            all_nodes: [batch, quadruple_node_num], List[List[str]]
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
            user_neighbour_emb_index (torch.Tensor):
                Size: [all_user_num, max_neighbour_num]
            user_neighbour_mask (torch.Tensor):
                Size: [all_user_num, max_neighbour_num]
            item_image (torch.Tensor):
                Size: [all_item_num, *]
            item_text (torch.Tensor):
                [all_item_num, hidden_size]
            attribute_text (torch.Tensor):
                Size: [all_attribute_num]
                The index of the src nodes (with corresponding to the ids in ``all_quadruples``).
        """
        all_dgl_graph, all_node2re_id, all_re_id2node, all_nodes, all_quadruples = zip(*batch)

        batch_size = len(all_dgl_graph)
        _nodes = set()
        _node2emb = {}
        max_subgraph_num = 0
        for b in range(batch_size):
            _nodes.update(all_nodes[b])
            max_subgraph_num = max(max_subgraph_num, max(map(lambda x: len(x), all_dgl_graph[b])))

        users = [_node for _node in _nodes if self.node2type[_node] == 'u']
        user_neighbours: Dict[str, List[str]] = {}
        max_user_neighbour_num = 0
        for user in users:
            assert user not in users
            user_neighbours[user] = self.ui_edges[user]
            max_user_neighbour_num = max(max_user_neighbour_num, len(user_neighbours[user]))
            for new_node in user_neighbours[user]:
                assert self.node2type[new_node] == 'i'
                if new_node not in _nodes:
                    _nodes.add(new_node)

        items = []
        item_image = []
        item_text = []
        item_text_mask = []
        attributes = []
        attr_text = []
        for _node in _nodes:
            _node_type = self.node2type[_node]
            if _node_type == 'i':
                items.append(_node)
                item_image_t, item_text_h, item_mask = self.load_embedding(_node)
                item_image.append(item_image_t)
                item_text.append(item_text_h)
                item_text_mask.append(item_mask)
            elif _node_type == 'a':
                attributes.append(_node)
                attr_text_h = self.load_embedding(_node)
                attr_text.append(attr_text_h)
            elif _node_type == 'u':
                continue
            else:
                raise RuntimeError(f"Unrecognized node type: {_node_type}.")
        # Item embedding: item_image + item_text
        # Attribute embedding: attr_text
        item_image = torch.cat(item_image, dim=0)
        item_text = torch.cat(item_text, dim=0)
        item_text_mask = torch.cat(item_text_mask, dim=0)
        attr_text = torch.cat(attr_text, dim=0)

        node2emb_index = {}
        for i, _node in enumerate(items + attributes + users):
            node2emb_index[_node] = i

        # User embedding: will be gathered through ``user_neighbour_emb_index``.
        user_neighbour_emb_index = torch.zeros(len(users), max_user_neighbour_num, dtype=torch.long)
        user_neighbour_mask = torch.zeros(len(users), max_user_neighbour_num, dtype=torch.long)
        for u_id, user in enumerate(users):
            mapped_item_emb_index = list(map(lambda x: node2emb_index[x], user_neighbours[user]))
            assert all(x < len(items) for x in mapped_item_emb_index)  # make sure all neighbours use the embedding from the items.
            user_neighbour_emb_index[u_id, :len(user_neighbours[user])] = torch.tensor(mapped_item_emb_index, dtype=torch.long)
            user_neighbour_mask[u_id, :len(user_neighbours[user])] = 1

        quadruple_num = len(all_dgl_graph[0])
        assert quadruple_num == len(all_node2re_id[0]) == len(all_re_id2node[0]) == len(all_nodes[0])

        # index of source node of each subgraph, used for gathering the corresponding hidden states.
        src_index = torch.zeros(batch_size, quadruple_num, max_subgraph_num, dtype=torch.long).fill_(-1)
        subgraph_mask = torch.zeros(batch_size, quadruple_num, max_subgraph_num, dtype=torch.long)
        input_emb_index = []
        graphs = []

        for b in range(batch_size):
            for t in range(quadruple_num):
                for g, (subgraph_re_id2node, subgraph_node2re_id) in enumerate(all_re_id2node[b][t], all_node2re_id[b][t]):
                    src_index[b, t, g] = subgraph_node2re_id[all_quadruples[b][t]] + len(input_emb_index)  # offset
                    subgraph_mask[b, t, g] = 1
                    sg_node_num = len(subgraph_re_id2node)
                    sg_mapped_emb_index = list(map(lambda x: node2emb_index[subgraph_re_id2node[x]], range(sg_node_num)))
                    input_emb_index.extend(sg_mapped_emb_index)
                graphs.extend(all_dgl_graph[b][t])
        graph = dgl.batch(graphs)
        input_emb_index = torch.tensor(input_emb_index, dtype=torch.long)

        return graph, input_emb_index, src_index, subgraph_mask, user_neighbour_emb_index, user_neighbour_mask, \
               item_image, item_text, item_text_mask, attr_text
