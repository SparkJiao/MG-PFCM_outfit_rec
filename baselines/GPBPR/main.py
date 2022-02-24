# coding=utf-8
from GPBPR import GPBPR
import torch

from torch.utils.data import TensorDataset,DataLoader
from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from torch.nn.functional import logsigmoid
from torch.nn.init import uniform_

from torch.optim import Adam
from sys import argv
import pickle
from tensorboardX import SummaryWriter
writer = SummaryWriter()
"""
    my_config is a dict which contains necessary filepath in trainning and evaluating GP-BPR model

    visual_features is the output of last avgpool in resnet50 of torchvision, obtained by 

    textural_features is the input of word embeding layer

    embedding_matrix is the word embedding vector from nwjc2vec. Missing word initialed as zero vector 
"""
opt_num = [1]
my_config = {
    #"visual_features_dict": "visualfeatures",
    #"textural_idx_dict": "textfeatures",
    #"textural_embedding_matrix": "smallnwjc2vec",
    "visual_features_dict": "reprocess/visualfeatures",
    "textural_idx_dict": "reprocess/textfeatures",
    "textural_embedding_matrix": "reprocess/word2vec",
    "train_data": r"data/train.csv",
    "valid_data": r"data/dev.csv",
    "test_data": r"data/test.csv",
    "test_mrr_data": r"data/test_mrr.csv",
    "model_file": r"/model/GPBPR.model",
}

def get_rank(pos_score, neg_scores):
    rank = 1
    for score in neg_scores:
        if score > pos_score:
            rank += 1
    return rank


def get_mrr(scores_list):
    mrr = 0
    for scores in scores_list:
        pos_score = scores[0]
        neg_scores = scores[1:]
        rank = get_rank(pos_score, neg_scores)
        mrr += 1.0 / rank
    return mrr / len(scores_list)


def load_csv_data(train_data_path):
    result = []
    with open(train_data_path,'r') as fp:
        for line in fp:
            t = line.strip().split(',')
            t = [int(i) for i in t]
            result.append(t)
    return result

def write_loss(loss):  
    error_path = "/home/share/wangchun/baseline/GPBPRcode/result/loss.txt"
    with open(error_path,'a') as file_handle:   # .txt可以不自己新建,代码会自动新建
        file_handle.write(str(loss.item()))     # 写入
        file_handle.write('\n')

def write_auc(res):  
    error_path = "/home/share/wangchun/baseline/GPBPRcode/result/auc.txt"
    with open(error_path,'a') as file_handle:   # .txt可以不自己新建,代码会自动新建
        file_handle.write(res)     # 写入
        file_handle.write('\n')


""" 这里的字典读取有些奇怪
def load_embedding_weight(device):
    jap2vec = torch.load(my_config['textural_embedding_matrix'])
    embeding_weight = []
    for jap, vec in jap2vec.items():
        embeding_weight.append(vec.tolist())
    embeding_weight.append(torch.zeros(300))
    embedding_weight = torch.tensor(embeding_weight, device=device)
    return embedding_weight
"""

def load_embedding_weight(device):
    #jap2vec = torch.load(my_config['textural_embedding_matrix'])

    df=open(my_config['textural_embedding_matrix'],'rb')#注意此处是rb
    #此处使用的是load(目标文件)
    jap2vec=pickle.load(df)
    df.close()

    word_num = len(jap2vec)
    embeding_weight = []
    embeding_weight.append(torch.zeros(768).tolist())
    for i in range(1,word_num):
        embeding_weight.append(jap2vec[str(i)].tolist())
    embedding_weight = torch.tensor(embeding_weight, device=device)
    return embedding_weight


def trainning(model, mode, train_data_loader, device, visual_features, text_features, opt):
    r"""
        using data from Args to train model

        Args:

            mode: -

            train_data_loader: mini-batch iteration

            device: device on which model train

            visual_features: look up table for item visual features

            text_features: look up table for item textural features

            opt: optimizer of model
    """
    model.train()
    model = model.to(device)

    for iteration,aBatch in enumerate(train_data_loader):

        output , outputweight = model.fit(aBatch[0], visual_features, text_features, weight=False)  
        # print(output.size())
        loss = (-logsigmoid(output)).sum() + 0.001*outputweight
        #print(iteration,'====',loss.item())
        write_loss(loss)
        writer.add_scalar('result/loss', loss.item(), opt_num[0])
        iteration += 1
        opt_num[0] = opt_num[0]+1
        opt.zero_grad()
        loss.backward()
        opt.step()

def evaluating(model, mode, test_csv, visual_features, text_features):
    r"""
        using data from Args to train model

        Args:

            mode: -

            train_data_loader: mini-batch iteration

            test_csv: valid file or test file

            visual_features: look up table for item visual features

            text_features: look up table for item textural features
    """
    model.eval()
    testData = load_csv_data(test_csv)
    pos = 0
    batch_s = 100
    for i in range(0, len(testData), batch_s):
        data = testData[i:i+batch_s] if i+batch_s <=len(testData) else testData[i:]
        output = model.forward(data, visual_features, text_features)  
        
        pos += float(torch.sum(output.ge(0)))
    auc = pos/len(testData)
    print( "evaling process: " , test_csv , model.epoch, auc)
    res = "evaling process: "+test_csv+'========'+str(model.epoch)+'===='+ str(auc)
    write_auc(res)
    return auc



def evaluating_mrr2(model, mode, test_mrr_csv, visual_features, text_features):
    r"""
        
    """
    model.eval()
    testData = load_csv_data(test_mrr_csv)
    MRR_all = 0
    batch_s = 100
    for i in range(0, len(testData), batch_s):
        data = testData[i:i+batch_s] if i+batch_s <=len(testData) else testData[i:]
        res_mrr = model.forward_mrr2(data, visual_features, text_features)  

        MRR_n = float(torch.sum(res_mrr))

        MRR_all += MRR_n
    MRR = MRR_all/(len(testData))
    #print('MRR==================',MRR,MRR_all)
    print( "evaling process_mrr: " , test_mrr_csv , model.epoch, MRR)
    return MRR



def F(mode ,hidden_dim, batch_size, device):
    print('loading top&bottom features')
    # torch.cuda.set_device("")
    train_data = load_csv_data(my_config['train_data'])
    valid_data = load_csv_data(my_config['valid_data'])
    test_data = load_csv_data(my_config['test_data'])



    #visual_features = torch.load(my_config['visual_features_dict'], map_location= lambda a,b:a.cpu())
    
    #text_features = torch.load(my_config['textural_idx_dict'], map_location= lambda a,b:a.cpu())

    df=open(my_config['visual_features_dict'],'rb')
    visual_features=pickle.load(df)
    df.close()

    df=open(my_config['textural_idx_dict'],'rb')
    text_features=pickle.load(df)
    df.close()
    #print(text_features)


    try:
        print("loading model")
        gpbpr = load(my_config['model_file'], map_location=lambda x,y: x.cuda(device))
    except Exception as e:
        print(e)
        print('no module exists, created new one {}'.format(my_config['model_file']))
        embedding_weight = load_embedding_weight(device)

        item_set1= set()
        user_set1 = set([str(i[0]) for i in train_data])
        for i in train_data:
            item_set1.add(str(int(i[1])))
            item_set1.add(str(int(i[2])))
            item_set1.add(str(int(i[3])))

        item_set2= set()
        user_set2 = set([str(i[0]) for i in valid_data])
        for i in valid_data:
            item_set2.add(str(int(i[1])))
            item_set2.add(str(int(i[2])))
            item_set2.add(str(int(i[3])))

        item_set3= set()
        user_set3 = set([str(i[0]) for i in test_data])
        for i in test_data:
            item_set3.add(str(int(i[1])))
            item_set3.add(str(int(i[2])))
            item_set3.add(str(int(i[3])))

        all_user_set = user_set1.union(user_set2,user_set3)

        all_item_set = item_set1.union(item_set2,item_set3)


        gpbpr = GPBPR(user_set = all_user_set, item_set = all_item_set, 
            embedding_weight=embedding_weight, uniform_value = 0.3).to(device)
    
    opt = Adam([
    {
        'params': gpbpr.parameters(),
        'lr': 0.001,
        'weight_decay':1e-3,
    }
    ])

    train_data = TensorDataset(torch.tensor(train_data, dtype=torch.int))
    train_loader = DataLoader(train_data, batch_size= batch_size,shuffle=True, drop_last=True)


    for i in range(800):

        # 这里是单个进程的训练
        trainning(gpbpr, mode, train_loader,device, visual_features, text_features, opt)
        
        
        gpbpr.epoch+=1
        torch.save(gpbpr, my_config['model_file'])

        auc_train = evaluating(gpbpr,mode, my_config['train_data'],  visual_features, text_features,)
        auc_valid = evaluating(gpbpr,mode, my_config['valid_data'],  visual_features, text_features,)
        auc_test = evaluating(gpbpr,mode, my_config['test_data'],  visual_features, text_features,)

        writer.add_scalar('result/train_auc', auc_train, i)
        writer.add_scalar('result/valid_auc', auc_valid, i)
        writer.add_scalar('result/test_auc', auc_test, i)

        mrr2_test = evaluating_mrr2(gpbpr,mode, my_config['test_data'],  visual_features, text_features)
        writer.add_scalar('result/test_mrr2', mrr2_test, i)

if __name__ == "__main__":

    # "cpu" or "cuda:x" x is GPU index like (0,1,2,3,)
    import os
    try:
        os.mkdir('./model')
    except Exception: pass
    F(mode = 'final', hidden_dim = 512, batch_size = 128, device = 0)
