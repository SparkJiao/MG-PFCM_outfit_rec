'''
Created on June, 2020
Tensorflow Implementation of HFGN model in:
Xingchen Li et al. In SIGIR 2020.
Hierarchical Fashion Graph Network for Personalized Outfit Recommendation.

@author: Xingchen Li (xingchenl@zju.edu.cn)
'''

import utility.metrics as metrics
import multiprocessing
import heapq
import numpy as np
import sys
_cores = multiprocessing.cpu_count() // 2


_data_generator = None
_USR_NUM = None
_OUTFIT_NUM = None
_N_TRAIN = None
_N_TEST = None


def ranklist_by_heapq(user_pos_test, test_outfits, rating, Ks):
    outfit_score = {}
    for i in test_outfits:
        outfit_score[i] = rating[i]

    K_max = max(Ks)
    K_max_outfit_score = heapq.nlargest(K_max, outfit_score, key=outfit_score.get) #找到最大的K个元素

    r = []
    for i in K_max_outfit_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    return r

def get_auc(outfit_score, user_pos_test):
    outfit_score = sorted(outfit_score.outfits(), key=lambda kv: kv[1])
    outfit_score.reverse()
    outfit_sort = [x[0] for x in outfit_score]
    posterior = [x[1] for x in outfit_score]

    r = []
    for i in outfit_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_outfits, rating, Ks):
    outfit_score = {}
    for i in test_outfits:
        outfit_score[i] = rating[i]

    K_max = max(Ks)
    K_max_outfit_score = heapq.nlargest(K_max, outfit_score, key=outfit_score.get)

    r = []
    for i in K_max_outfit_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(outfit_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}

def test_one_user(x):
    #print('aaaaaaaaaaaaaaa')
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's outfits in the training set
    try:
        training_outfits = _data_generator.train_u_outfits_dict[u]
    except Exception:
        training_outfits = []
    #user u's outfits in the test set
    user_pos_test = _data_generator.test_u_outfits_dict[u]

    all_outfits = set(range(_OUTFIT_NUM))

    test_outfits = list(all_outfits - set(training_outfits)) # 计算了全部的items

    r = ranklist_by_heapq(user_pos_test, test_outfits, rating, Ks)

    return get_performance(user_pos_test, r, Ks)

def test_one_user_auc(x):
    #print('aaaaaaaaaaaaaaa')
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's outfits in the training set

    user_pos_test = _data_generator.test_u_outfits_dict[u]
    user_pos_test_neg = _data_generator.test_u_outfits_dict_neg[u]
    res = 0
    #if len(user_pos_test) != len(user_pos_test_neg):
        #print('error======',u,len(user_pos_test),len(user_pos_test_neg))
        #sys.exit()
    all_num = len(user_pos_test)
    for i in range(all_num):
        pos_o_id = user_pos_test[i]
        neg_o_id = user_pos_test_neg[i]
        pos_o_score = rating[pos_o_id]
        neg_o_score = rating[neg_o_id]
        if pos_o_score>neg_o_score:
            res = res+1
    #print('===============================')
    #print(res,all_num)
    return [res,all_num]

def test_one_user_mrr2(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's outfits in the training set

    user_pos_test = _data_generator.test_u_outfits_dict[u]
    user_pos_test_neg = _data_generator.test_u_outfits_dict_neg[u]
    res = 0

    all_num = len(user_pos_test)
    for i in range(all_num):
        pos_o_id = user_pos_test[i]
        neg_o_id = user_pos_test_neg[i]
        pos_o_score = rating[pos_o_id]
        neg_o_score = rating[neg_o_id]
        if pos_o_score>=neg_o_score:
            res = res+1
        else:
            res = res+0.5
    #print('===============================')
    #print(res,all_num)
    return [res,all_num]


def test(sess, model, users_to_test, data_generator, args, drop_flag=True, batch_test_flag=False):
    global _data_generator
    global _USR_NUM
    global _OUTFIT_NUM
    global _N_TRAIN
    global _N_TEST
    global Ks
    global  _BATCH_SIZE

    Ks = eval(args.Ks)
    _BATCH_SIZE = args.batch_size

    _data_generator = data_generator

    _USR_NUM, _OUTFIT_NUM = _data_generator.n_users, _data_generator.n_train_outfits
    _N_TEST = _data_generator.n_recom_tests

    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks))}

    #pool = multiprocessing.Pool(_cores)
    pool2 = multiprocessing.Pool(_cores)

    u_batch_size = _BATCH_SIZE * 2
    i_batch_size = _BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        outfit_batch = range(_OUTFIT_NUM)
        #outfit_batch = [1,13,15]
        #print('=====',outfit_batch)

        if drop_flag == False:
            rate_batch = sess.run(model.batch_ratings, {model.user_input:user_batch,  model.po_input: outfit_batch})
        else:
            rate_batch = sess.run(model.batch_ratings, {model.user_input:user_batch,  model.po_input: outfit_batch,
                                                          model.node_dropout: [0.],
                                                          model.mess_dropout: [0.]})

        user_batch_rating_uid = zip(rate_batch, user_batch)


        mrr2_result = pool2.map(test_one_user_mrr2, user_batch_rating_uid)
        mrr = 0
        all_num = 0
        for mrr2_res in mrr2_result:
            mrr = mrr+mrr2_res[0]
            all_num = all_num+mrr2_res[1]

        mrr2 = mrr/all_num
        result['mrr2'] = mrr2

    #pool.close()
    pool2.close()
    return result

def _add_mask(num_items, items_u, num_max):
    '''
    uniformalize the length of each batch
    '''
    for i in range(len(items_u)):
        items_u[i] = items_u[i] + [num_items] * (num_max + 1 - len(items_u[i]))

    return items_u
