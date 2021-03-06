import argparse
from multiprocessing import Pool

"""
This script have several aims:
1.  Process or load a processed subgraph (generated by process_subgraph_v2.py)
2.  For each node, load the corresponding subgraph, load the corresponding embedding, 
    relabeling node id to prepare dgl.graph, and save the final node set.

As a result, the data loader can directly load the embedding of each node.
"""

