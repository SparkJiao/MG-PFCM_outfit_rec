from typing import Dict, List, Set, Tuple, Any
from torch import Tensor

node_vocab: Dict[str, List]
node2type: Dict[str, str]


def get_node_type(node: str) -> str:
    return node2type[node]


def load_embedding(node: str) -> Tuple[str, Tensor]:
    pass

