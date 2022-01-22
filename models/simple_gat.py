"""Most code are copied from https://github.com/THUDM/HGB/blob/master/LP/benchmark/methods/baseline/GNN.py."""

import torch
from torch import nn

from models.simple_gat_conv import SimpleGATConv
from models.modeling_utils import get_activation_func
from general_util.logger import get_child_logger

logger = get_child_logger("SimpleGAT")


class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        thW = self.W[r_id]
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()


class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()

    def forward(self, left_emb, right_emb, r_id):
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(left_emb, right_emb).squeeze()


class SimpleGAT(nn.Module):
    def __init__(self,
                 edge_dim,
                 num_etypes,
                 num_hidden,
                 head_size,
                 num_layers,
                 heads,
                 activation,
                 feat_dropout,
                 attn_dropout,
                 negative_slope=0.01,
                 residual=False,
                 alpha=0.,
                 decode='distmult'):
        super(SimpleGAT, self).__init__()

        # self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = get_activation_func(activation) if isinstance(activation, str) else activation

        logger.info(f'Simple-GAT parameters:\theads: {heads}')

        # self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        # for fc in self.fc_list:
        #     nn.init.xavier_normal_(fc.weight, gain=1.414)

        # input projection (no residual)
        out_dim = num_hidden
        self.gat_layers.append(SimpleGATConv(edge_dim, num_etypes,
                                             num_hidden, head_size, heads[0],
                                             feat_dropout, attn_dropout, negative_slope, False, self.activation, alpha=alpha))
        out_dim += head_size
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(SimpleGATConv(edge_dim, num_etypes,
                                                 head_size * heads[l - 1], head_size, heads[l],
                                                 feat_dropout, attn_dropout, negative_slope, residual, self.activation, alpha=alpha))
            out_dim += head_size
        # output projection
        self.gat_layers.append(SimpleGATConv(edge_dim, num_etypes,
                                             head_size * heads[-2], num_hidden, heads[-1],
                                             feat_dropout, attn_dropout, negative_slope, residual, None, alpha=alpha))
        out_dim += num_hidden

        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        if decode == 'distmult':
            self.decoder = DistMult(num_etypes, num_hidden * (num_layers + 2))
        elif decode == 'dot':
            self.decoder = Dot()
        elif decode == 'proj':
            self.decoder = nn.Linear(out_dim, num_hidden)
        else:
            raise RuntimeError()

    def l2_norm(self, x):
        # This is an equivalent replacement for tf.l2_normalize,
        # see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        return x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))

    def forward(self, graph, h, e_feat):
        # h = []
        # for fc, feature in zip(self.fc_list, features_list):
        #     h.append(fc(feature))
        # h = torch.cat(h, 0)

        emb = [self.l2_norm(h)]
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](graph, h, e_feat, res_attn=res_attn)
            emb.append(self.l2_norm(h.mean(1)))
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](graph, h, e_feat, res_attn=res_attn)  # None)
        logits = logits.mean(1)
        logits = self.l2_norm(logits)
        emb.append(logits)
        logits = torch.cat(emb, 1)

        # left_emb = logits[left]
        # right_emb = logits[right]
        # return self.decoder(left_emb, right_emb, mid)
        return self.decoder(logits)
