import dgl
import torch.nn
from dgl.nn.pytorch import GATConv
from torch import nn, Tensor


class GAT(nn.Module):
    def __init__(self,
                 num_layers: int = 3,
                 input_size: int = 768,
                 num_heads: int = 12,
                 head_size: int = 64,
                 feat_dropout: float = 0.1,
                 attn_dropout: float = 0.1,
                 residual: bool = True):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size

        self.gat = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat.append(GATConv(in_feats=input_size,
                                        out_feats=head_size,
                                        num_heads=num_heads,
                                        feat_drop=feat_dropout,
                                        attn_drop=attn_dropout,
                                        residual=residual,
                                        activation=torch.nn.ELU()))
            elif i < num_layers - 1:
                self.gat.append(GATConv(in_feats=head_size * num_heads,
                                        out_feats=head_size,
                                        num_heads=num_heads,
                                        feat_drop=feat_dropout,
                                        attn_drop=attn_dropout,
                                        residual=residual,
                                        activation=torch.nn.ELU()))
            else:
                self.gat.append(GATConv(in_feats=head_size * num_heads,
                                        out_feats=head_size * num_heads,
                                        num_heads=1,
                                        feat_drop=feat_dropout,
                                        attn_drop=attn_dropout,
                                        residual=residual,
                                        activation=None))

    def forward(self, graph: dgl.graph, node_features: Tensor):
        num_nodes = node_features.size(0)
        for layer_idx in range(self.num_layers):
            node_features = self.gat[layer_idx](graph, node_features)
            node_features = node_features.reshape(num_nodes, -1)

        return node_features
