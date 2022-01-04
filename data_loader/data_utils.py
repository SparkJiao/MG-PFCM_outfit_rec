import random
from typing import Dict, Any
from collections import defaultdict

import torch
from torch import Tensor


class EmbeddingMatrix:
    """
    Cache the embedding on cpu instead of cuda to avoid
    """

    def __init__(self,
                 attr_text: str,
                 item_image: str,
                 item_text: str):
        self.attr_text: Tensor = torch.load(attr_text, map_location='cpu')
        self.item_image: Tensor = torch.load(item_image, map_location='cpu')
        self.item_text: Tensor = torch.load(item_text, map_location='cpu')


class MaximusNeighbourSampler:
    def __init__(self, max_neighbour_num: int = 10):
        self.max_neighbour_num = max_neighbour_num

    def __call__(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        src = graph['src_id']
        neighbours = graph['edges']

        sampled_neighbours = {}

        queue = [src]
        vis = {src}
        node2hop = {src: 0}
        hop_node_list = defaultdict(set)
        hop_node_list[0].add(src)
        while len(queue) > 0:
            node_i = queue.pop(0)
            node_u_set = neighbours[node_i]
            if len(node_u_set) > self.max_neighbour_num:
                sampled_node_u = random.sample(node_u_set, self.max_neighbour_num)
            else:
                sampled_node_u = node_u_set
            sampled_neighbours[node_i] = set(sampled_node_u)
            cur_hop = node2hop[node_i]
            for node_u in sampled_node_u:
                node2hop[node_u] = cur_hop + 1
                hop_node_list[cur_hop + 1].add(node_u)
                if node_u not in vis:
                    queue.append(node_u)
                    vis.add(node_u)
                # Add the ignored edges.
                for node_p in hop_node_list[cur_hop]:
                    if node_p in neighbours and node_u in neighbours[node_p] and \
                            node_p in sampled_neighbours and node_u not in sampled_neighbours[node_p]:
                        sampled_neighbours[node_p].add(node_u)

        return {
            'meta_path': graph['meta_path'],
            'src_id': src,
            'edges': sampled_neighbours
        }
