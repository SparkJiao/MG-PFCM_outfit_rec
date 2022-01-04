import torch
from torchvision.models import resnet18
from torch import nn

import json
from multiprocessing import Pool
from tqdm import tqdm

import os

device = torch.device("cuda:0")

resnet = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1]).to(device=device)
resnet.eval()


def initializer(_ui_edges):
    global ui_edges
    ui_edges = _ui_edges


def load_item_embedding(item):
    text = torch.load(os.path.join("/home/wangchun/work3/Initialization/text/", f"{item}_t.pt"), map_location='cpu')[:, 0].detach()
    image = torch.load(os.path.join("/home/wangchun/work3/Initialization/img/", f"{item}_v.pt"))

    text_h = text.mean(dim=0)
    image = resnet(image.unsqueeze(0).to(device)).reshape(-1).cpu()
    # print(text_h.size(), image.size())
    return text_h, image


def get_user_embedding(user):
    emb_ls = []
    with torch.no_grad():
        for item in ui_edges[user]:
            text_h, image_h = load_item_embedding(item)
            emb_ls.append(torch.cat([text_h, image_h], dim=-1))
    return torch.stack(emb_ls, dim=0).mean(dim=0)


if __name__ == '__main__':
    node_vocab = torch.load("/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/vocab.pt")
    ui = json.load(open("/home/jiaofangkai/IQON_pair_remove_edge/UI.json", 'r'))

    # Process user embedding
    # initializer(ui)
    #
    # user_embedding = {}
    # for u in tqdm(node_vocab['u'], total=len(node_vocab['u'])):
    #     user_embedding[u] = get_user_embedding(u)
    #
    # torch.save(user_embedding, "/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/user_embedding.pt")

    # Process user vocabulary
    user_embedding = torch.load("/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/user_embedding.pt")
    user_emb_weight = []
    user_vocab = {}
    for i, (user, user_emb) in enumerate(user_embedding.items()):
        user_emb_weight.append(user_emb)
        user_vocab[user] = i
    user_emb_weight = torch.stack(user_emb_weight, dim=0)

    torch.save(user_emb_weight, "/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/user_emb_weight.pt")
    json.dump(user_vocab, open("/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/user_vocab.json", "w"))

    print("Done.")
