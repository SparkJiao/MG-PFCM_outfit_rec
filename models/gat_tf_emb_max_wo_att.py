import json
from copy import deepcopy

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
                 loss_type: int = 0,
                 add_ctr_loss: bool = False,
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
        # self.attr_proj = nn.Linear(text_hidden_size, hidden_size)

        self.gat = gnn
        self.transformer = transformer

        self.f_h = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.f_h.data.normal_(mean=0.0, std=self.transformer.config.init_std)
        self.register_buffer("pad", torch.ones(1, 1))
        self.register_buffer("label", torch.ones(1, dtype=torch.long))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        self.loss_type = loss_type
        if loss_type == 0:
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == 1:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise RuntimeError()

        self.add_ctr_loss = add_ctr_loss
        if self.add_ctr_loss:
            mlp = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1)
            )
            self.query = deepcopy(mlp)
            self.key = deepcopy(mlp)
            self.ctr_loss_fn = nn.CrossEntropyLoss()
            self.register_buffer("ctr_label", torch.zeros(1, dtype=torch.long))

        self.gat.apply(init_weights)
        if self.add_ctr_loss:
            self.init_metric("loss", "acc", "ctr_acc")
        else:
            self.init_metric("loss", "acc")

    def forward(self,
                graph: dgl.graph,
                input_emb_index: Tensor,
                src_index: Tensor,
                subgraph_mask: Tensor,
                item_image: Tensor,
                item_text: Tensor,
                attr_text: Tensor,
                user_emb_index: Tensor,
                **gnn_kwargs):

        num_images = item_image.size(0)
        image_emb = self.resnet(item_image)
        image_emb = image_emb.reshape(num_images, -1)

        # [num_layer, seq_len, h] -> [seq_len, num_layer * h]
        item_emb = self.item_proj(torch.cat([image_emb, item_text], dim=-1))

        # attr_emb = self.attr_proj(attr_text[:, 0])
        attr_emb = torch.zeros(attr_text.size(0), item_emb.size(-1), dtype=item_emb.dtype, device=item_emb.device)

        node_emb = torch.cat([item_emb, attr_emb], dim=0)

        user_emb = self.user_proj(self.user_embedding_layer(user_emb_index))

        node_emb = torch.cat([node_emb, user_emb], dim=0)

        node_feat = torch.index_select(node_emb, dim=0, index=input_emb_index)
        node_feat = node_feat.to(dtype=torch.float)
        with autocast(enabled=False):
            node_feat = self.gat(graph, node_feat, **gnn_kwargs)
        node_feat = node_feat.to(dtype=node_emb.dtype)

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

        pos_triplet = torch.cat([u_h, i_h, p_h], dim=-1).reshape(batch, -1)
        neg_triplet = torch.cat([u_h, i_h, n_h], dim=-1).reshape(batch, -1)

        pos_logits = self.mlp(pos_triplet).reshape(batch)
        neg_logits = self.mlp(neg_triplet).reshape(batch)

        if self.loss_type == 0:
            logits = pos_logits - neg_logits
            labels = self.label.expand(batch).float()
            loss = self.loss_fn(logits, labels)
            logits = torch.stack([neg_logits, pos_logits], dim=-1)
        elif self.loss_type == 1:
            logits = torch.stack([neg_logits, pos_logits], dim=-1)
            labels = self.label.expand(batch)
            loss = self.loss_fn(logits, labels)
        else:
            raise RuntimeError()

        if self.add_ctr_loss:
            pos_query = self.query(pos_triplet)
            # neg_query = self.query(neg_triplet)
            pos_key = self.key(pos_triplet)
            neg_key = self.key(neg_triplet)

            pos_sim = torch.einsum("ah,bh->ab", pos_query, pos_key)  # [batch, batch]
            neg_sim = torch.einsum("ah,bh->ab", pos_query, neg_key)  # [batch, batch]

            pos_sim = pos_sim.unsqueeze(2)  # [batch, batch, 1]
            neg_sim = neg_sim.unsqueeze(1).expand(-1, batch, -1)  # [batch, 1, batch] -> [batch, batch, batch]
            sim_matrix = torch.cat([pos_sim, neg_sim], dim=-1).reshape(batch * batch, (batch + 1))
            ctr_labels = self.ctr_label.expand(batch * batch)
            ctr_loss = self.ctr_loss_fn(sim_matrix, ctr_labels)
            loss += ctr_loss
        else:
            sim_matrix = ctr_labels = None

        if not self.training:
            acc, true_label_num = get_accuracy(logits, labels)
            self.eval_metrics.update("acc", val=acc, n=true_label_num)
            self.eval_metrics.update("loss", val=loss.item(), n=true_label_num)

            if self.add_ctr_loss:
                ctr_acc, ctr_label_num = get_accuracy(sim_matrix, ctr_labels)
                self.eval_metrics.update("ctr_acc", val=ctr_acc, n=ctr_label_num)

        return {
            "loss": loss,
            "logits": logits
        }
