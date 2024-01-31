# here put the import lib
import copy
import os
import pickle
import numpy as np
from utils.utils import read_jsonlines, multi_label_metric, ddi_rate_score, multi_test
from generators.data import Voc, EHRTokenizer


def evaluate_jsonlines(data_path, ehr_tokenizer, threshold=0.5, ddi_path='./data/mimic4/handled/'):

    pred_data_prob, pred_data = [], []
    true_data = np.zeros((len(read_jsonlines(data_path)), len(ehr_tokenizer.med_voc.word2idx)))
    seq_len = []
    pred_label = []

    for row, meta_data in enumerate(read_jsonlines(data_path)):
        
        # noramlize the predicted scores by sigmoid, and get the prob
        meta_pred_data_prob = np.array(meta_data["target"])
        pred_data_prob.append(np_sigmoid(meta_pred_data_prob))
        
        # transform y to 0-1 by threshold
        meta_pred_data = copy.deepcopy(np_sigmoid(meta_pred_data_prob))
        meta_pred_data[meta_pred_data>=threshold] = 1
        meta_pred_data[meta_pred_data<threshold] = 0
        pred_data.append(meta_pred_data)
        
        # get the true data
        true_index = ehr_tokenizer.convert_med_tokens_to_ids(meta_data["drug_code"])
        true_data[row][true_index] = 1

        seq_len.append(int(meta_data["input"].split("The patient has ")[1].split(" times ICU visits.")[0]))

        # prepare the labels for DDI calculation
        meta_label = np.where(meta_pred_data == 1)[0]
        pred_label.append([sorted(meta_label)])
    
    ja, prauc, avg_p, avg_r, avg_f1, mean, std = multi_label_metric(true_data, 
                                                         np.array(pred_data), 
                                                         np.array(pred_data_prob))
    ddi_adj = pickle.load(open(os.path.join(ddi_path, 'ddi_A_final.pkl'), 'rb'))
    ddi = ddi_rate_score(pred_label, ddi_adj)
    
    print('\nJaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, DDI_rate: {:.4}\n'.format(
          ja, prauc, avg_p, avg_r, avg_f1, ddi
    ))
    print("10-rounds PRAUC: %.5f + %.5f" % (mean[0], std[0]))
    print("10-rounds Jaccard: %.5f + %.5f" % (mean[1], std[1]))
    print("10-rounds F1-score: %.5f + %.5f" % (mean[2], std[2]))

    seq_len = np.array(seq_len)
    pred_data = np.array(pred_data)
    pred_data_prob = np.array(pred_data_prob)
    single_index = (seq_len == 0)
    multi_index = (seq_len >= 1)
    acc_container = {}
    s_ja, s_prauc, s_avg_p, s_avg_r, s_avg_f1, s_mean, s_std = multi_label_metric(true_data[single_index], 
                                                                   pred_data[single_index], 
                                                                   pred_data_prob[single_index])
    m_ja, m_prauc, m_avg_p, m_avg_r, m_avg_f1, m_mean, m_std = multi_label_metric(true_data[multi_index], 
                                                                   pred_data[multi_index], 
                                                                   pred_data_prob[multi_index])
    acc_container['single-jaccard'] = s_ja
    acc_container['single-f1'] = s_avg_f1
    acc_container['single-prauc'] = s_prauc
    acc_container['multiple-jaccard'] = m_ja
    acc_container['multiple-f1'] = m_avg_f1
    acc_container['multiple-prauc'] = m_prauc

    for k, v in acc_container.items():
        print('%-10s : %-10.4f' % (k, v))
    
    print("Single-visit 10-rounds PRAUC: %.5f + %.5f" % (s_mean[0], s_std[0]))
    print("Single-vist 10-rounds Jaccard: %.5f + %.5f" % (s_mean[1], s_std[1]))
    print("Single-visit 10-rounds F1-score: %.5f + %.5f" % (s_mean[2], s_std[2]))
    print("Multi-visit 10-rounds PRAUC: %.5f + %.5f" % (m_mean[0], m_std[0]))
    print("Multi-vist 10-rounds Jaccard: %.5f + %.5f" % (m_mean[1], m_std[1]))
    print("Multi-visit 10-rounds F1-score: %.5f + %.5f" % (m_mean[2], m_std[2]))

    return ja, prauc, avg_p, avg_r, avg_f1

    
def np_sigmoid(x):
    # sigmoid function using numpy
    return 1 / (1+np.exp(-x))



if __name__ == "__main__":

    # load diag, proc, med word2id tokenizer
    voc_dir = "data/mimic3/handled/voc_final.pkl"
    ehr_tokenizer = EHRTokenizer(voc_dir)

    pred_path = "./results/0105/test_predictions.json"

    evaluate_jsonlines(pred_path, ehr_tokenizer, threshold=0.16)


