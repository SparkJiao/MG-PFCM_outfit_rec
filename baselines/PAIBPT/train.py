# coding=utf-8
from GPBPR2 import GPBPR
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

    "visual_features_dict": "visualfeatures",
    "textural_idx_dict": "textfeatures",
    "textural_embedding_matrix": "word2vec",
    "train_data": r"data/train.csv",
    "valid_data": r"data/dev.csv",
    "test_data": r"data/test.csv",
    "model_file": r"model/PAI-BPR.model",

}


def load_csv_data(train_data_path):
    result = []
    with open(train_data_path,'r') as fp:
        for line in fp:
            t = line.strip().split(',')
            t = [int(i) for i in t]
            result.append(t)
    return result


def write_loss(loss):  
    error_path = "result/loss.txt"
    with open(error_path,'a') as file_handle:  
        file_handle.write(str(loss.item()))    
        file_handle.write('\n')

def write_auc(res):  
    error_path = "result/auc.txt"
    with open(error_path,'a') as file_handle:   
        file_handle.write(res)    
        file_handle.write('\n')


def load_embedding_weight(device):
    #jap2vec = torch.load(my_config['textural_embedding_matrix'])

    df=open(my_config['textural_embedding_matrix'],'rb')

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
        #print(output.size())
        loss = (-logsigmoid(output)).sum() + 0.001*outputweight
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
        output = model.forward(data, visual_features,text_features)  
        #print(output.shape)
        pos += float(torch.sum(output.ge(0)))
    #print( "evaling process: " , test_csv , model.epoch, pos/len(testData))
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

def F(mode, hidden_dim, vis_feat, uniform_val, batch_size, epochs, device):
    print('loading top&bottom features')

    train_data = load_csv_data(my_config['train_data'])
    valid_data = load_csv_data(my_config['valid_data'])
    test_data = load_csv_data(my_config['test_data'])


    print('loading visual and text features')
    df=open(my_config['visual_features_dict'],'rb')
    #此处使用的是load(目标文件)
    visual_features=pickle.load(df)
    #r1=pickle.load(df)
    #print(r1.popitem())
    #print(r1['41955463'])
    #print(r1['41955463'].size())
    df.close()

    df=open(my_config['textural_idx_dict'],'rb')
    text_features=pickle.load(df)
    df.close()
    print('successful')


    try:
        print("loading model")
        gpbpr = load(my_config['model_file'], map_location=lambda x,y: x.cuda(device))
        print('successful')
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

        gpbpr = GPBPR(user_set = all_user_set, item_set = all_item_set, visual_feature_dim = int(vis_feat), 
                      hidden_dim= int(hidden_dim), embedding_weight=embedding_weight, 
                      uniform_value = float(uniform_val)).to(device)
    
    opt = Adam([
    {
        'params': gpbpr.parameters(),
        'lr': 0.001,
        'weight_decay':1e-3,
    }
    ])
    print('loading training data')
    train_data = TensorDataset(torch.tensor(train_data, dtype=torch.int))
    train_loader = DataLoader(train_data, batch_size= batch_size,shuffle=True, drop_last=True)
    print('successful')

    for i in range(int(epochs)):
        print('iteration ', str(i))

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

    F(mode = 'final', hidden_dim = 512, 
      vis_feat = 2048, 
      uniform_val = 0.05, batch_size = 128, 
      epochs = 200, device = 0)
