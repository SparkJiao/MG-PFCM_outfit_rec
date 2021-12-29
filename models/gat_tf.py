import dgl
import torch
from torch import Tensor, nn
from torch.nn import Module
from torchvision.models import resnet18

from gat import GAT
from modeling_utils import init_weights
from transformer import initialize_transformer


class GATTransformer(Module):
    def __init__(self):
        super().__init__()

        # Item image embedding
        self.resnet = nn.Sequential(*list(resnet18(pretrained=True).children())[:-2])
        # Item text embedding
        self.item_fc = nn.Linear(2048 + 768, 768)
        # Attribute text embedding
        self.attr_text_fc = nn.Linear(768, 768)
        # User embedding
        self.user_fc = nn.Linear(768, 768)

        self.gat = GAT()
        self.tf_config, self.transformer = initialize_transformer()

        self.f_h = nn.Parameter(torch.FloatTensor(1, self.tf_config.d_model))
        self.f_h.data.normal_(mean=0.0, std=self.tf_config.init_std)
        self.register_buffer("pad", torch.ones(1, 1))
        self.register_buffer("pos_label", torch.ones(1))
        self.register_buffer("neg_label", torch.zeros(1))

        self.mlp = nn.Sequential(
            nn.Linear(768 * 3, 768),
            nn.GELU(),
            nn.Linear(768, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.gat.apply(init_weights)

    def forward(self,
                graph: dgl.graph,
                input_emb_index: Tensor,
                src_index: Tensor,
                subgraph_mask: Tensor,
                user_neighbour_emb_index: Tensor,
                user_neighbour_mask: Tensor,
                item_image: Tensor,
                item_text: Tensor,
                attr_text: Tensor):
        num_images = item_image.size(0)
        image_emb = self.resnet(item_image).view(num_images, -1)
        item_emb = self.item_fc(torch.cat([image_emb, item_text], dim=0))

        attr_emb = self.attr_text_fc(attr_text)

        node_emb = torch.cat([item_emb, attr_emb], dim=0)

        user_num, max_neighbour_num = user_neighbour_emb_index.size()
        user_neigh_emb = torch.gather(node_emb, dim=0, index=node_emb.unsqueeze(-1).expand(-1, -1, node_emb.size(-1)))

        # mean pooling
        user_neighbour_num = user_neighbour_mask.sum(dim=1)
        user_emb = user_neigh_emb * user_neighbour_mask.unsqueeze(-1).to(user_neigh_emb.dtype).sum(dim=1)
        user_emb = user_emb / user_neighbour_num.unsqueeze(-1)
        user_emb = self.user_fc(user_emb)

        node_emb = torch.cat([node_emb, user_emb], dim=0)
        node_num = node_emb.size(0)

        node_feat = torch.index_select(node_emb, dim=0, index=input_emb_index)
        node_feat = self.gat(graph, node_feat)

        # select the hidden states of the source nodes as those of their corresponding subgraph.
        batch, tuple_len, max_subgraph_num = src_index.size()
        assert tuple_len == 4
        sg_h = torch.gather(node_feat, dim=0,
                            index=src_index.reshape(batch * tuple_len * max_subgraph_num, 1).expand(-1, node_feat.size(-1)))
        sg_h = sg_h.reshape(batch * tuple_len, max_subgraph_num, -1)
        sg_h = torch.cat([self.f_h[None, None, :].expand(batch * tuple_len, -1, -1), sg_h], dim=1)

        # transformer fusion
        g_h = self.transformer(
            hidden_states=sg_h,
            attention_mask=torch.cat([self.pad.expand(sg_h.size(0), -1), subgraph_mask.reshape(-1, max_subgraph_num)], dim=1),
        )[0][:, 0].reshape(batch, tuple_len, sg_h.size(-1))
        u_h, i_h, p_h, n_h = g_h.split(1, dim=1)

        pos_logits = self.mlp(torch.cat([u_h, i_h, p_h], dim=-1))
        neg_logits = self.mlp(torch.cat([u_h, i_h, n_h], dim=-1))
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([self.pos_label.expand(batch), self.neg_label.expand(batch)], dim=0)
        loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits
        }
