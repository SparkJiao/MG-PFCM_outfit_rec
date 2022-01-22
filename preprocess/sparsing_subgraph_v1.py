"""
This script aims at sparsing the subgraphs off-line.
"""

import sys
import argparse
import glob
import os.path
import random
from multiprocessing import Pool
from typing import Union, Dict

import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from data_loader.data_utils import MaximusNeighbourSampler

sampler: MaximusNeighbourSampler


def init(_sampler: MaximusNeighbourSampler):
    global sampler
    sampler = _sampler


def sparse_graph(graph: Union[str, Dict]):
    if isinstance(graph, str):
        graph = torch.load(graph)

    graph = sampler(graph)
    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--max_neighbour_num', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    if os.path.isfile(args.path):
        subgraph = torch.load(args.path)
    else:
        subgraph = list(glob.glob(args.path))

    max_neighbour_sampler = MaximusNeighbourSampler(args.max_neighbour_num)

    with Pool(args.num_workers, initializer=init, initargs=(max_neighbour_sampler,)) as p:
        results = list(tqdm(
            p.imap(sparse_graph, subgraph, chunksize=32),
            total=len(subgraph),
            desc="Sparsing graph."
        ))

    torch.save(results, args.output_file + f"_{args.seed}_{args.max_neighbour_num}")
    print("Done.")
