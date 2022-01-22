import json
import os

import torch
from torch import nn
from torchvision.models import resnet18
from tqdm import tqdm
import argparse

device = torch.device("cuda:0")
batch_size = 256

resnet = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1]).to(device=device)
resnet.eval()

# img_dir = "/home/wangchun/work3/Initialization/img/"
# text_dir = "/home/wangchun/work3/Initialization/text/"
img_dir = "/home/wangchun/work3/Initialization_all/img/"
text_dir = "/home/wangchun/work3/Initialization_all/text/"


def initializer(_ui_edges):
    global ui_edges
    ui_edges = _ui_edges


def load_item_embedding(item):
    if not os.path.exists(os.path.join(text_dir, f"{item}_t.pt")) and not os.path.join(img_dir, f"{item}_v.pt"):
        return None, None
    if not os.path.exists(os.path.join(text_dir, f"{item}_t.pt")):
        text = torch.zeros(2, 768)
    else:
        text = torch.load(os.path.join(text_dir, f"{item}_t.pt"), map_location='cpu')[:, 0].detach()

    if not os.path.exists(os.path.join(img_dir, f"{item}_v.pt")):
        image = torch.zeros(3, 224, 224)
    else:
        image = torch.load(os.path.join(img_dir, f"{item}_v.pt"))

    text_h = text.mean(dim=0)
    # image = resnet(image.unsqueeze(0).to(device)).reshape(-1).cpu()
    # print(text_h.size(), image.size())
    return text_h, image


def get_user_embedding(_user):

    with torch.no_grad():
        text_emb_ls = []
        img_ls = []
        for item in ui_edges[_user]:
            # text_h, image_h = load_item_embedding(item)
            # if text_h is None and image_h is None:
            #     continue
            # emb_ls.append(torch.cat([text_h, image_h], dim=-1))
            text, img = load_item_embedding(item)
            if text is None and img is None:
                continue
            text_emb_ls.append(text)
            img_ls.append(img)
        text_emb = torch.stack(text_emb_ls, dim=0)
        all_img = torch.stack(img_ls, dim=0)
        # print(all_img.size())
        idx = 0
        img_emb_ls = []
        while True:
            s_idx = idx * batch_size
            if s_idx >= all_img.size(0):
                break
            e_idx = (idx + 1) * batch_size
            batch_img = all_img[s_idx: e_idx]
            # print(batch_img.size())
            batch_img_emb = resnet(batch_img.to(device)).reshape(batch_img.size(0), -1).cpu()
            img_emb_ls.append(batch_img_emb)
            idx += 1
        img_emb = torch.cat(img_emb_ls, dim=0)
        assert img_emb.size(0) == text_emb.size(0)
    # return torch.stack(emb_ls, dim=0).mean(dim=0)
    return torch.cat([text_emb, img_emb], dim=-1).mean(dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_vocab', type=str, required=True)
    parser.add_argument('--ui_edge_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    # node_vocab = torch.load("/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/vocab.pt")
    node_vocab = torch.load(args.node_vocab)
    # ui = json.load(open("/home/jiaofangkai/IQON_pair_remove_edge/UI.json", 'r'))
    ui = json.load(open(args.ui_edge_file, 'r'))

    # Process user embedding
    initializer(ui)

    user_embedding = {}
    for u in tqdm(node_vocab['u'], total=len(node_vocab['u'])):
        user_embedding[u] = get_user_embedding(u)

    # torch.save(user_embedding, "/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/user_embedding.pt")
    torch.save(user_embedding, os.path.join(args.output_dir, 'user_embedding.pt'))

    # user_embedding = torch.load(os.path.join(args.output_dir, 'user_embedding.pt'))
    # for u, emb in user_embedding.items():
    #     user_embedding[u] = emb.mean(dim=0)
    # torch.save(user_embedding, os.path.join(args.output_dir, 'user_embedding.pt'))

    # Process user vocabulary
    # user_embedding = torch.load("/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/user_embedding.pt")
    user_emb_weight = []
    user_vocab = {}
    for i, (user, user_emb) in enumerate(user_embedding.items()):
        user_emb_weight.append(user_emb)
        user_vocab[user] = i
    user_emb_weight = torch.stack(user_emb_weight, dim=0)

    # torch.save(user_emb_weight, "/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/user_emb_weight.pt")
    # json.dump(user_vocab, open("/home/jiaofangkai/IQON_pair_remove_edge/subgraphs/user_vocab.json", "w"))
    torch.save(user_emb_weight, os.path.join(args.output_dir, 'user_emb_weight.pt'))
    json.dump(user_vocab, open(os.path.join(args.output_dir, 'user_vocab.json'), 'w'))

    print("Done.")
