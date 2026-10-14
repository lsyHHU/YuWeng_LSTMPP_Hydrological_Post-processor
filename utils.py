import torch
import numpy as np
import random
import tqdm


def get_rolling_window_multistep(forecasting_length, interval_length, window_length, features, labels,labels_sim):
    output_features = np.zeros((1, features.shape[0], window_length))
    output_labels = np.zeros((1, 1, forecasting_length))
    output_labels_sim = np.zeros((1, 1, forecasting_length))
    if features.shape[1] != labels.shape[1]:
        assert 'cant process such data'
    else:
        output_features = np.zeros((1, features.shape[0], window_length))
        output_labels = np.zeros((1, 1, forecasting_length))
        output_labels_sim = np.zeros((1, 1, forecasting_length))
        for index in tqdm.tqdm(range(0, features.shape[1]-interval_length-window_length-forecasting_length+1), desc='data preparing'):
            output_features = np.concatenate((output_features, np.expand_dims(features[:, index:index+window_length], axis=0)))
            output_labels = np.concatenate((output_labels, np.expand_dims(labels[:, index+interval_length+window_length: index+interval_length+window_length+forecasting_length], axis=0)))
            output_labels_sim = np.concatenate((output_labels_sim, np.expand_dims(labels_sim[:, index+interval_length+window_length: index+interval_length+window_length+forecasting_length], axis=0)))
    output_features = output_features[1:, :, :]
    output_labels = output_labels[1:, :, :]
    output_labels_sim = output_labels_sim[1:, :, :]
    return torch.from_numpy(output_features), torch.from_numpy(output_labels), torch.from_numpy(output_labels_sim)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def feature_label2src_trg_trgy(features, labels):
    trg_len = labels.shape[1]
    labels = labels.unsqueeze(1)
    gather_ = torch.cat((features[:, 0, :].unsqueeze(1), labels), dim=2)  # 将features中labels的历史数据和labels拼接
    trg = gather_[:, :, -1 - trg_len:-1]
    #trg = gather_[:, :, -1-trg_len:-1]  # 构建训练时decoder输入trg
    #  features.shape[n, features_num, input_len]
    #  trg.shape[n, features_num(always as 1), trg_len(equals output_len]
    #  labels.shape[n, features_num(always as 1), trg_len(equals output_len]
    #  example features[1,2,3], trg[2,3], labels[3,4]
    return trg


def generate_square_subsequent_mask(dim1, dim2):
    return torch.triu(torch.triu(torch.ones(dim1, dim2)*float('-inf'), diagonal=1))









