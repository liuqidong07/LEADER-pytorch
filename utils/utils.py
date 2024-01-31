# here put the import lib
import json
import random
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import numpy as np
import os
import logging
import time
import torch
import torch.functional as F
import jsonlines
import pickle


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return score

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(
                y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(
                y_gt[b], y_prob[b], average='macro'))
        return all_micro

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    auc = roc_auc(y_gt, y_prob)
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_pred)
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)
    mean, std = multi_test(prauc, ja, avg_f1)

    return np.mean(ja), np.mean(prauc), np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1), mean, std



def ddi_rate_score(record, ddi_A):
    # ddi rate
    # ddi_A = pickle.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt



def metric_report(logger, y_pred, y_true, therhold=0.5, ddi_graph=None):
    y_prob = y_pred.copy()
    y_pred[y_pred > therhold] = 1
    y_pred[y_pred <= therhold] = 0

    acc_container = {}
    ja, prauc, avg_p, avg_r, avg_f1, mean, std = multi_label_metric(
        y_true, y_pred, y_prob)
    acc_container['jaccard'] = ja
    acc_container['f1'] = avg_f1
    acc_container['prauc'] = prauc

    pred_label = [[sorted(np.where(meta_pred == 1)[0])] for meta_pred in y_pred]
    ddi = ddi_rate_score(pred_label, ddi_graph)
    acc_container["ddi"] = ddi

    for k, v in acc_container.items():
        logger.info('%-10s : %-10.4f' % (k, v))
    logger.info("10-rounds PRAUC: %.5f + %.5f" % (mean[0], std[0]))
    logger.info("10-rounds Jaccard: %.5f + %.5f" % (mean[1], std[1]))
    logger.info("10-rounds F1-score: %.5f + %.5f" % (mean[2], std[2]))

    return acc_container


def metric_report_group(logger, y_pred, y_true, seq_len, therhold=0.5, ddi_graph=None):
    y_prob = y_pred.copy()
    y_pred[y_pred > therhold] = 1
    y_pred[y_pred <= therhold] = 0

    # get the single visit and multi visit index to take out them
    single_index = (seq_len == 1)
    multi_index = (seq_len != 1)
    acc_container = {}
    if single_index.sum() == 0:  # no single-visit condition
        s_ja, s_prauc, s_avg_p, s_avg_r, s_avg_f1 = 0, 0, 0, 0, 0
        s_mean = [0, 0, 0]
        s_std = [0, 0, 0]
    else:
        s_ja, s_prauc, s_avg_p, s_avg_r, s_avg_f1, s_mean, s_std = multi_label_metric(y_true[single_index], 
                                                                    y_pred[single_index], 
                                                                    y_prob[single_index])
    m_ja, m_prauc, m_avg_p, m_avg_r, m_avg_f1, m_mean, m_std = multi_label_metric(y_true[multi_index], 
                                                                   y_pred[multi_index], 
                                                                   y_prob[multi_index])
    acc_container['single-jaccard'] = s_ja
    acc_container['single-f1'] = s_avg_f1
    acc_container['single-prauc'] = s_prauc
    acc_container['multiple-jaccard'] = m_ja
    acc_container['multiple-f1'] = m_avg_f1
    acc_container['multiple-prauc'] = m_prauc

    s_pred_label = [[sorted(np.where(meta_pred == 1)[0])] for meta_pred in y_pred[single_index]]
    s_ddi = ddi_rate_score(s_pred_label, ddi_graph)
    acc_container["single-ddi"] = s_ddi
    m_pred_label = [[sorted(np.where(meta_pred == 1)[0])] for meta_pred in y_pred[multi_index]]
    m_ddi = ddi_rate_score(m_pred_label, ddi_graph)
    acc_container["multi-ddi"] = m_ddi

    for k, v in acc_container.items():
        logger.info('%-10s : %-10.4f' % (k, v))

    logger.info("Single-visit 10-rounds PRAUC: %.5f + %.5f" % (s_mean[0], s_std[0]))
    logger.info("Single-vist 10-rounds Jaccard: %.5f + %.5f" % (s_mean[1], s_std[1]))
    logger.info("Single-visit 10-rounds F1-score: %.5f + %.5f" % (s_mean[2], s_std[2]))
    logger.info("Multi-visit 10-rounds PRAUC: %.5f + %.5f" % (m_mean[0], m_std[0]))
    logger.info("Multi-vist 10-rounds Jaccard: %.5f + %.5f" % (m_mean[1], m_std[1]))
    logger.info("Multi-visit 10-rounds F1-score: %.5f + %.5f" % (m_mean[2], m_std[2]))

    return acc_container


def t2n(x):
    return x.detach().cpu().numpy()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def set_seed(seed):
    '''Fix all of random seed for reproducible training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_efficiency(best_epoch, train_time, fp_num, ap_num, args, now_str):

    out_path = args.out_exp.split('/')[-1]
    out_path = os.path.join('./log/efficiency', out_path)

    if not os.path.exists(out_path):
        with open(out_path, 'w+') as f:
            json.dump({}, f)
    
    with open(out_path, 'r') as f:
        res_dict = json.load(f)

    res = {'best_epoch': best_epoch,
           'train_time': train_time,
           'fp_num': fp_num,
           'ap_num':ap_num}
    new_dict = {'model': args.model_name, 'hos_id': args.hos_id}
    new_dict.update(res)
    res_dict.update({now_str: new_dict})

    with open(out_path, 'w') as f:
        json.dump(res_dict, f)
    

def read_jsonlines(data_path):
    '''read data from jsonlines file'''
    data = []

    with jsonlines.open(data_path, "r") as f:
        for meta_data in f:
            data.append(meta_data)

    return data


def save_jsonlines(data_path, data):
    '''write all_data list to a new jsonl'''
    with jsonlines.open(data_path, "w") as w:
        for meta_data in data:
            w.write(meta_data)


def log_res(args, res):

    # create the folder to save result json
    res_dir = "./log/results/"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    if args.out_file == "none":
        out_file = args.dataset + ".json"
    else:
        out_file = args.out_file
    out_path = os.path.join(res_dir, out_file)

    # for the first record
    if not os.path.exists(out_path):
        with open(out_path, 'w+') as f:
            json.dump({}, f)
    
    with open(out_path, 'r') as f:
        res_dict = json.load(f)

    res_dict.update({args.model_name + "-" + args.mark_name: res})

    with open(out_path, 'w') as f:
        json.dump(res_dict, f)


def multi_test(prauc, ja, f1):

    result = []
    for _ in range(10):
        data_num = len(ja)
        final_length = int(0.8 * data_num)
        idx_list = list(range(data_num))
        random.shuffle(idx_list)
        idx_list = idx_list[:final_length]
        avg_ja = np.mean([ja[i] for i in idx_list])
        avg_prauc = np.mean([prauc[i] for i in idx_list])
        avg_f1 = np.mean([f1[i] for i in idx_list])
        result.append([avg_prauc, avg_ja, avg_f1])
    result = np.array(result)
    mean = result.mean(axis=0)
    std = result.std(axis=0)

    outstring = ""
    for m, s in zip(mean, std):
        outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

    # print(outstring)

    return mean, std

