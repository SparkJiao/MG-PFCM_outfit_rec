import torch
from torch import nn, Tensor

from general_util.logger import get_child_logger

logger = get_child_logger("layer")


class LearnableLayerCombine(nn.Module):
    def __init__(self, num_layers: int = 2):
        super().__init__()

        self.weight = nn.Parameter(torch.FloatTensor(num_layers))

    def forward(self, x: Tensor):
        """
        :param x: [num_layers, seq_len, h]
        :return: y: [seq_len, h]
        """
        return torch.einsum("l,lsh->sh", self.weight, x)


def bpr_loss(positive_logits: Tensor, negative_logits: Tensor):
    ...

