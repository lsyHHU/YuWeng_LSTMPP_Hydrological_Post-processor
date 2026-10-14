#%%
import random
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from typing import Dict
import torch
import torch.nn as nn

ONE_OVER_2PI_SQUARED = 1.0 / np.sqrt(2.0 * np.pi)

#%%
class mlp_processor(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hidden1: int = 10,n_hidden2:int=10):
        super(mlp_processor, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1,n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_out)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a simple multi layer network.
        Parameters
        ----------
        x : torch.Tensor
         """
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return h

# %%
# net1 = mlp_processor(14,7,16,16 )
# total = np.sum( p.numel() for p in net1.parameters() )
# print(total)

#%%

class Config:
    def __init__(self,n_distributions):
        self.n_distributions=n_distributions


class GaussianLoss(nn.Module):
    """单高斯分布的负对数似然损失"""
    def __init__(self):
        super(GaussianLoss, self).__init__()
        self.eps = 1e-6
    
    def forward(self, prediction, target):
        mu = prediction['mu']      # [batch, seq_len, features]
        sigma = prediction['sigma']
        y = target['y']            # [batch, seq_len, features]

        # 计算标准正态分布的对数概率
        # NLL = 0.5 * ((y - mu)/sigma)^2 + log(sigma) + 0.5 * log(2*pi)
        log_sqrt_2pi = 0.5 * np.log(2 * np.pi)
        
        # 核心计算：点对点的 NLL
        variance = sigma ** 2
        nll = 0.5 * ((y - mu) ** 2) / (variance + self.eps) + torch.log(sigma + self.eps) + log_sqrt_2pi
        
        return nll.mean()

class Encoder(nn.Module):
# 新增 dropout_rate 参数
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu", dropout_rate=0.2):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        
        # LSTM 内部的 dropout (仅当 num_layers > 1 时有效)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, 
                            batch_first=True, bidirectional=False, 
                            dropout=dropout_rate if num_layers > 1 else 0)
        
        # 显式添加 Dropout 层，确保 num_layers=1 时也有随机性
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        
        input_seq = input_seq.permute((0,2,1))
        
        # output: [batch, seq_len, hidden]
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        
        # 对 h 进行 dropout (增加隐状态的不确定性)
        h = self.dropout(h)
        
        return h, c

class Decoder(nn.Module):
# 新增 dropout_rate 参数
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, dropout_rate=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                             batch_first=True, bidirectional=False,
                             dropout=dropout_rate if num_layers > 1 else 0)
        
        # 显式 Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq, h, c):
        input_seq = input_seq.permute((0,2,1))
        
        output, (h, c) = self.lstm(input_seq, (h, c))
        
        # 在进入 Linear 层之前进行 Dropout
        output = self.dropout(output)
        
        pred = self.linear(output)
        return pred
        # return pred.permute((0,2,1)), h
    
#这个 Decoder 类的 forward 方法返回两个值：pred.permute((0,2,1)) 和 h。
# pred.permute((0,2,1)) 是经过线性层 (self.linear) 处理后的输出。
# 这个输出是经过一些线性变换的隐藏状态 output，通过 self.linear 将隐藏状态映射到了 output_size 维度上，然后对维度进行了排列，得到形状为 (batch_size, output_size, 1) 的张量。
# h 是 LSTM 层输出的隐藏状态，用于传递到下一个时间步。
# 在这个方法中，虽然返回了 h，但实际上并没有用到这个 h，因为在下一步计算中会重新更新 h 和 c。
# 这里的 h 是为了满足 LSTM 层的输出要求而返回的，实际上在这个方法中并没有用到这个 h，可以理解为暂时保留的中间状态。
 
    
# LSTM input PARAMETERS: 
# batch_first – If True, then the input / output tensors are provided as (batch, seq, feature) 
# HIDDEN OUT: (num_layers, batch, hidden feature size)
# input_size	输入数据的特征维数
# hidden_size	LSTM中隐层的维度
# output_size 输出数据的特征维数
# num_layers	循环神经网络的层数
# LSTM output: (batch, length, hidden feature size)

#%%
class GaussianHead(nn.Module):
    # 新增 dropout_rate 参数
    def __init__(self, n_in: int, n_out_features: int, n_hidden: int = 100, dropout_rate=0.2):
        super(GaussianHead, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        
        # 新增 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(n_hidden, n_out_features * 2)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = torch.relu(self.fc1(x))
        
        # 在两层全连接之间随机丢弃神经元
        h = self.dropout(h)
        
        h = self.fc2(h)

        mu, sigma = h.chunk(2, dim=-1)
        return {
            'mu': mu, 
            'sigma': torch.exp(sigma) + self._eps
        }
#%%
class GaussianCRPSLoss(nn.Module):
    """
    针对高斯分布输出的 CRPS Loss (解析解)
    公式来源: Gneiting et al. (2005)
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_dist, target):
        """
        pred_dist: 字典, 包含 'mu' (B, T, ...) 和 'sigma' (B, T, ...)
        target: 字典, 包含 'y' (B, T, ...)
        """
        mu = pred_dist['mu']
        sigma = pred_dist['sigma']
        y = target['y']

        # 1. 计算标准化残差 z
        # 加上极小值防止 sigma 为 0 导致除零错误
        eps = 1e-6 
        z = (y - mu) / (sigma + eps)

        # 2. 标准正态分布的 PDF 和 CDF
        # pdf(z) = 1/sqrt(2pi) * exp(-0.5 * z^2)
        pdf = (1.0 / np.sqrt(2.0 * np.pi)) * torch.exp(-0.5 * z**2)
        
        # cdf(z) = 0.5 * (1 + erf(z / sqrt(2)))
        cdf = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))

        # 3. CRPS 解析公式
        # CRPS = sigma * [ z * (2 * CDF(z) - 1) + 2 * PDF(z) - 1/sqrt(pi) ]
        crps_score = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi))

        if self.reduction == 'mean':
            return torch.mean(crps_score)
        elif self.reduction == 'sum':
            return torch.sum(crps_score)
        else:
            return crps_score
    
class Seq2Seq(nn.Module):
# 增加 dropout 参数，默认为 0.2
    def __init__(self, hidden_size, num_layers, batch_size, device="cuda", dropout=0.2):
        super().__init__()
        
        self.encoder = Encoder(input_size=2, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_size=batch_size, 
                               device=device, dropout_rate=dropout) # 传入
        
        self.decoder = Decoder(input_size=1, hidden_size=hidden_size, 
                               num_layers=num_layers, output_size=hidden_size, 
                               batch_size=batch_size, dropout_rate=dropout) # 传入
        
        self.head = GaussianHead(n_in=hidden_size, n_out_features=1, 
                                 n_hidden=hidden_size, dropout_rate=dropout) # 传入
                                 
        self.device = device

    def forward(self, past_obs_sim, future_sim):
        h, c = self.encoder(past_obs_sim)
        outputs = self.decoder(future_sim, h, c)
        residual_params = self.head(outputs)
        return residual_params 
#%%
# model = Seq2Seq ( 2, 1, 
#                  32 ,
#                  num_layers=1, 
#                  output_size_decoder=32,  # output feature size
#                  batch_size=128, device='cpu' )
# input_seq = torch.randn(128, 2,7)
# trg_seq = torch.randn(128, 1,5) 
# output = model(  input_seq, trg_seq)
# print(output['mu'].shape)


#%%
def _get_predict_last_n(cfg: Config) -> dict:
    predict_last_n = cfg.predict_last_n
    if isinstance(predict_last_n, int):
        predict_last_n = {'': predict_last_n}
    if len(predict_last_n) == 1:
        predict_last_n = {'': list(predict_last_n.values())[0]}  # if there's only one frequency, we omit its identifier
    return predict_last_n

# %%
