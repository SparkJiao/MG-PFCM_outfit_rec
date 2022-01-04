import json

import dgl
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import Module

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from models.modeling_utils import init_weights, initialize_vision_backbone, get_accuracy

logger = get_child_logger('GATTransformerVocab')


class GATTransformer(Module, LogMixin):
    def __init__(self,
                 user_embedding: str,
                 user_vocab: str,
                 freeze_user_emb: bool = False,
                 vision_model: str = 'resnet18',
                 text_hidden_size: int = 768,
                 img_hidden_size: int = 2048,
                 hidden_size: int = 768,
                 gnn: Module = None,
                 transformer: Module = None):
        super().__init__()

        # User embedding
        user_vocab = json.load(open(user_vocab, 'r'))
        user_embedding = torch.load(user_embedding, map_location='cpu')
        self.user_embedding_layer = nn.Embedding(len(user_vocab),
                                                 text_hidden_size + img_hidden_size).from_pretrained(user_embedding, freeze=freeze_user_emb)
        self.user_proj = nn.Linear(text_hidden_size + img_hidden_size, hidden_size)

        # Item image encoding
        self.resnet = initialize_vision_backbone(vision_model)
        # Item text embedding
        self.item_proj = nn.Linear(text_hidden_size + img_hidden_size, hidden_size)
        # Attribute text embedding
        self.attr_proj = nn.Linear(text_hidden_size, hidden_size)

        self.gat = gnn
        self.transformer = transformer

        self.f_h = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.f_h.data.normal_(mean=0.0, std=self.transformer.config.init_std)
        self.register_buffer("pad", torch.ones(1, 1))
        self.register_buffer("pos_label", torch.ones(1, dtype=torch.long))
        self.register_buffer("neg_label", torch.zeros(1, dtype=torch.long))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.gat.apply(init_weights)
        self.init_metric("loss", "acc")

    def forward(self,
                graph: dgl.graph,
                input_emb_index: Tensor,
                src_index: Tensor,
                subgraph_mask: Tensor,
                item_image: Tensor,
                item_text: Tensor,
                attr_text: Tensor,
                user_emb_index: Tensor):

        num_images = item_image.size(0)
        image_emb = self.resnet(item_image)
        image_emb = image_emb.reshape(num_images, -1)

        # [num_layer, seq_len, h] -> [seq_len, num_layer * h]
        item_emb = self.item_proj(torch.cat([image_emb, item_text], dim=-1))

        attr_emb = self.attr_proj(attr_text[:, 0])

        node_emb = torch.cat([item_emb, attr_emb], dim=0)

        user_emb = self.user_proj(self.user_embedding_layer(user_emb_index))

        node_emb = torch.cat([node_emb, user_emb], dim=0)

        node_feat = torch.index_select(node_emb, dim=0, index=input_emb_index)
        with autocast(enabled=False):
            node_feat = self.gat(graph, node_feat.float())

        # select the hidden states of the source nodes as those of their corresponding subgraph.
        batch, tuple_len, max_subgraph_num = src_index.size()
        assert tuple_len == 4
        sg_h = torch.gather(node_feat, dim=0,
                            index=src_index.reshape(batch * tuple_len * max_subgraph_num, 1).expand(-1, node_feat.size(-1)))
        sg_h = sg_h.reshape(batch * tuple_len, max_subgraph_num, -1)
        sg_h = torch.cat([self.f_h[None, :, :].expand(batch * tuple_len, -1, -1), sg_h], dim=1)

        # transformer fusion
        g_h = self.transformer(
            hidden_states=sg_h,
            attention_mask=torch.cat([self.pad.expand(sg_h.size(0), -1), subgraph_mask.reshape(-1, max_subgraph_num)], dim=1),
        )[0][:, 0].reshape(batch, tuple_len, sg_h.size(-1))
        u_h, i_h, p_h, n_h = g_h.split(1, dim=1)

        pos_logits = self.mlp(torch.cat([u_h, i_h, p_h], dim=-1)).squeeze(1)
        neg_logits = self.mlp(torch.cat([u_h, i_h, n_h], dim=-1)).squeeze(1)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([self.pos_label.expand(batch), self.neg_label.expand(batch)], dim=0)
        loss = self.loss_fn(logits, labels)

        if not self.training:
            acc, true_label_num = get_accuracy(logits, labels)
            self.eval_metrics.update("acc", val=acc, n=true_label_num)
            self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

        return {
            "loss": loss,
            "logits": logits
        }
