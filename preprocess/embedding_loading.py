import json
import os
from typing import Union, Tuple

import torch
from torch import Tensor
from tqdm import tqdm


def load_embedding(node, node_type, emb_path_dic) -> Union[Tensor, Tuple[Tensor, ...]]:
    if node_type == 'a':
        attr = torch.load(os.path.join(emb_path_dic['a'], f'{node}.pt')).detach()
        return attr
    elif node_type == 'i':
        if not os.path.exists(os.path.join(emb_path_dic['text'], f'{node}_t.pt')):
            text = torch.zeros(2, 768)
        else:
            text = torch.load(os.path.join(emb_path_dic['text'], f'{node}_t.pt'))[:, 0].detach()
        # mask = torch.load(os.path.join(self.emb_path_dic['mask'], f'{node}_mask.pt'))
        if not os.path.exists(os.path.join(emb_path_dic['image'], f'{node}_v.pt')):
            image = torch.zeros(3, 224, 224)
        else:
            image = torch.load(os.path.join(emb_path_dic['image'], f'{node}_v.pt'))
        return image, text.mean(dim=0)
    else:
        raise RuntimeError(f'Unrecognized node and node type: {node}, {node_type}.')


if __name__ == "__main__":
    # node_vocab = torch.load("/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/vocab.pt")
    # text_emb_dir = "/home/wangchun/work3/Initialization/text"
    # mask_emb_dir = "/home/wangchun/work3/Initialization/mask"
    # img_dir = "/home/wangchun/work3/Initialization/img"
    # att_dir = "/home/wangchun/work3/Initialization/attribute"
    node_vocab = torch.load("/home/jiaofangkai/gp-bpr/vocab.pt")
    text_emb_dir = "/home/wangchun/work3/Initialization_all/text"
    mask_emb_dir = "/home/wangchun/work3/Initialization_all/mask"
    img_dir = "/home/wangchun/work3/Initialization_all/img"
    att_dir = "/home/wangchun/work3/Initialization/attribute"

    emb_path = {
        'a': att_dir,
        'image': img_dir,
        'mask': mask_emb_dir,
        'text': text_emb_dir
    }

    a_emb = []
    a_vocab = {}
    for i, a in enumerate(tqdm(node_vocab['a'])):
        tmp = load_embedding(a, 'a', emb_path)
        a_emb.append(tmp)
        a_vocab[a] = i

    # torch.save(torch.stack(a_emb, dim=0), "/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/attribute_emb_weight.pt")
    # json.dump(a_vocab, open("/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/attribute.json", "w"))
    torch.save(torch.stack(a_emb, dim=0), "/home/jiaofangkai/gp-bpr/attribute_emb_weight.pt")
    json.dump(a_vocab, open("/home/jiaofangkai/gp-bpr/attribute.json", "w"))

    i_t_emb = []
    i_img_tensor = []
    i_vocab = {}
    for j, i in enumerate(tqdm(node_vocab['i'])):
        i_img, i_text = load_embedding(i, 'i', emb_path)
        i_t_emb.append(i_text)
        i_img_tensor.append(i_img)
        i_vocab[i] = j

    # torch.save(torch.stack(i_t_emb, dim=0), "/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/item_text_emb_weight.cls.pt")
    # torch.save(torch.stack(i_img_tensor, dim=0), "/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/item_img.pt")
    # json.dump(i_vocab, open("/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/item_vocab.json", "w"))
    torch.save(torch.stack(i_t_emb, dim=0), "/home/jiaofangkai/gp-bpr/item_text_emb_weight.cls.pt")
    torch.save(torch.stack(i_img_tensor, dim=0), "/home/jiaofangkai/gp-bpr/item_img.pt")
    json.dump(i_vocab, open("/home/jiaofangkai/gp-bpr/item_vocab.json", "w"))
