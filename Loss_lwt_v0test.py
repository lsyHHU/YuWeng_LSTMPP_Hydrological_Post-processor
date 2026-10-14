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


class MaskedCMALLoss(nn.Module):
    """Average negative log-likelihood for a model that uses the CMAL head.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    eps : float, optional
        Small constant for numeric stability.
    """

    def __init__(self, cfg: Config, eps: float = 1e-8):
        super(MaskedCMALLoss, self).__init__()
        self.eps = eps  # stability epsilon

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]):
        mask = ~torch.isnan(ground_truth['y']).any(1).any(1)
        y = ground_truth['y'][mask]
        m = prediction['mu'][mask]
        b = prediction['b'][mask]
        t = prediction['tau'][mask]
        p = prediction['pi'][mask]
        error = y - m
        log_like = torch.log(t) + \
                   torch.log(1.0 - t) - \
                   torch.log(b) - \
                   torch.max(t * error, (t - 1.0) * error) / b
        log_weights = torch.log(p + self.eps)

        result = torch.logsumexp(log_weights + log_like, dim=2)
        result = -torch.mean(torch.sum(result, dim=1))
        return result


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, 
                            batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        input_seq = input_seq.permute((0,2,1))
        # print(input_seq.shape)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        # print(output.shape)
        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                             batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq, h, c):
        batch_size = input_seq.shape[0]
        input_seq = input_seq.permute((0,2,1))
        # print(input_seq.shape)
        
        output, (h, c) = self.lstm(input_seq, (h, c))
        # print(output.shape,h.shape,c.shape)
        pred = self.linear(output)  # pred(batch_size, 1, output_size)
        # print(output.shape)
        
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
class CMAL(nn.Module):
    """Countable Mixture of Asymmetric Laplacians.

    An mixture density network with Laplace distributions as components.

    The CMAL-head uses an additional hidden layer to give it more expressiveness (same as the GMM-head).
    CMAL is better suited for many hydrological settings as it handles asymmetries with more ease. However, it is also
    more brittle than GMM and can more often throw exceptions. Details for CMAL can be found in [#]_.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 4 times the number of components.
    n_hidden : int
        Size of the hidden layer.

    References
    ----------
    .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing:
        Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295, 2020.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(CMAL, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

        self._softplus = torch.nn.Softplus(2)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a CMAL head forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Output of the previous model part. It provides the basic latent variables to compute the CMAL components.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary, containing the mixture component parameters and weights; where the key 'mu'stores the means,
            the key 'b' the scale parameters, the key 'tau' the skewness parameters, and the key 'pi' the weights).
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent, t_latent, p_latent = h.chunk(4, dim=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = self._softplus(b_latent) + self._eps  # scale > 0 (softplus was working good in tests)
        t = (1 - self._eps) * torch.sigmoid(t_latent) + self._eps  # 0 > tau > 1
        p = (1 - self._eps) * torch.softmax(p_latent, dim=-1) + self._eps  # sum(pi) = 1 & pi > 0

        return {'mu': m, 'b': b, 'tau': t, 'pi': p}

#%%

class Seq2Seq(nn.Module):
    def __init__(self, enc_channel_num, dec_channel_num, hidden_size, num_layers,
                 output_size_decoder, batch_size, 
                 cmal_out_size, device="cuda"):
        super().__init__()
        self.output_size = output_size_decoder
        self.Encoder = Encoder(enc_channel_num, hidden_size, num_layers, batch_size, device=device)
        self.Decoder = Decoder(dec_channel_num, hidden_size, num_layers, output_size_decoder, batch_size)
        self.device = device
        self.head=CMAL(n_in=output_size_decoder,n_out=4*cmal_out_size,n_hidden=hidden_size)

    def forward(self, input_seq, trg_seq):
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[2], input_seq.shape[1]
        trg_seq_len = trg_seq.shape[2]
        # h, c = self.Encoder(input_seq.permute((0,2,1)))
        h, c = self.Encoder(input_seq )
        
        decoder_input = trg_seq[:, :, :] #修改：input all sequences
        outputs  = self.Decoder(decoder_input, h, c)
            
        # outputs = torch.zeros(batch_size, self.output_size,3).to(self.device)
        # for t in range(3):
            # output  = self.Decoder(decoder_input, h, c)
            # outputs[:, :, t] = output[:,0,:] 
        # print(outputs.shape)

        CMAL_outputs=self.head(outputs)
        return CMAL_outputs
    
    
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
# LSTM input PARAMETERS: 
# batch_first – If True, then the input / output tensors are provided as (batch, seq, feature) 
# HIDDEN OUT: (num_layers, batch, hidden feature size)
# input_size	输入数据的特征维数
# hidden_size	LSTM中隐层的维度
# output_size 输出数据的特征维数
# num_layers	循环神经网络的层数
# LSTM output: (batch, length, hidden feature size)

class Seq2Seq_nohead(nn.Module):
    """
    in/out: (batch, length, channel)
    """
    
    def __init__(self, enc_channel_num, dec_channel_num, hidden_size, num_layers, output_size, batch_size, device="cuda"):
        
        super().__init__()
        self.output_size = output_size
        self.Encoder = Encoder(enc_channel_num, hidden_size, num_layers, batch_size, device=device)
        self.Decoder = Decoder(dec_channel_num, hidden_size, num_layers, output_size, batch_size)
        self.device = device
        # self.head=CMAL(n_in=3,n_out=12,n_hidden=100)

    def forward(self, input_seq, trg_seq ):
        # batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[2], input_seq.shape[1]
        batch_size  = input_seq.shape[0] 
        trg_seq_len = trg_seq.shape[2]
        h, c = self.Encoder(input_seq )
        outputs = torch.zeros(batch_size, self.output_size, trg_seq_len).to(self.device)
        decoder_input = trg_seq #修改 ；不应该只用最后一个时刻的[:, :, 0].unsqueeze(-1)
        # print(trg_seq.shape)
        outputs  = self.Decoder(decoder_input, h, c)

        return outputs


#Decoder的outputs本来就是能输出 (batch, seq, feature=1)，不需要做这个循环？

# model = Seq2Seq_nohead( 2, 1, 32 , 10, 1, 128, device='cpu' )
# input_seq = torch.randn(128, 7, 2)
# trg_seq = torch.randn(128, 3, 1)
# # input_seq = torch.randn((128, 2, 7))
# # trg_seq = torch.randn((128, 1, 3))
# # time_step = torch.randn((128, 1 ))
# # xt = torch.randn((128, 3 ))
# output = model(  input_seq, trg_seq)
# print(output.shape)



 
#%%
class Seq2Seq_diff(nn.Module):
    """
    in/out: (batch, length, channel)
    """
   
    def __init__(self, enc_channel_num, dec_channel_num, hidden_size, num_layers, output_size, batch_size,time_dim, device="cuda"):

        super().__init__()
        self.dec_channel_num = dec_channel_num
        self.hidden_size = hidden_size
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim*2),
            nn.SiLU(),
            nn.Linear(time_dim*2, hidden_size+dec_channel_num+1),
        )
        self.output_size = output_size
        self.Encoder = Encoder(enc_channel_num, hidden_size, num_layers, batch_size, device=device)
        self.Decoder = Decoder(dec_channel_num, hidden_size, num_layers, output_size, batch_size)
        self.device = device
        # self.head=CMAL(n_in=3,n_out=12,n_hidden=100)


    def forward(self, x, input_seq, trg_seq, timestep):
        """x: x_t ( Batch,1,trg_seq_length )
        
        """
        time_emb = self.time_mlp(timestep)    
        # print(time_emb.shape)    

        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[2], input_seq.shape[1]
        trg_seq_len = trg_seq.shape[2]
        
        #encoder
        h, c = self.Encoder(input_seq  )
        # print('H',h.shape,'c',c.shape) #(n layer, batch, hiddensize)


        #time embedding 加到hidden states
        # print(time_emb.shape)
        h  = h  + time_emb[:,:,:self.hidden_size].permute((1,0,2))  # Time embed is (B, 1, hidden_size)
        # time embedding 加到 cell states
        c  = c  + time_emb[:,:,:self.hidden_size].permute((1,0,2))  # Time embed is (B, 1, hidden_size)
        
        #concate xt, target condition
        x_cond = torch.concat([x,trg_seq], dim=1)#concat by channel
        # print(time_emb.shape, trg_seq.shape, x.shape)#batch, channel,length
        
        # time embedding  加到 decoder inputs 
        x_cond = x_cond + time_emb[:,:,self.hidden_size:] 

        #decoder
        outputs = self.Decoder(x_cond, h, c)
            
        
        return outputs
        
        # x = x * (scale + 1) + shift


        
#%%
class Seq2Seq_diff_doy(nn.Module):
    """
    doy: add time embedding 
    in:  (batch, length, channel)
    enc: encoder inputs
    decoder: xt and decoder inputs
    out: (batch, length, channel)
    
    """
   
    def __init__(self, enc_channel_num, dec_channel_num, 
                 hidden_size, num_layers, output_size, batch_size,
                 time_dim, doy_dim , device="cuda"):

        super().__init__()
        self.dec_channel_num = dec_channel_num
        self.hidden_size = hidden_size
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim*2),
            nn.SiLU(),
            nn.Linear(time_dim*2, hidden_size+dec_channel_num+1),
        )
        self.doy_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(doy_dim),
            nn.Linear(doy_dim, doy_dim*2),
            nn.SiLU(),
            nn.Linear(doy_dim*2, hidden_size+dec_channel_num+1),
        )
        
        self.output_size = output_size
        self.Encoder = Encoder(enc_channel_num, hidden_size, num_layers, batch_size, device=device)
        self.Decoder = Decoder(dec_channel_num, hidden_size, num_layers, output_size, batch_size)
        self.device = device
        # self.head=CMAL(n_in=3,n_out=12,n_hidden=100)


    def forward(self, x, input_seq, trg_seq, timestep): 
        """x: x_t ( Batch,1,trg_seq_length )
        
        """
        time_emb = self.time_mlp(timestep) 
           
        doy_emb = self.doy_emb(input_seq[:,1,-1:])# batch*1,   day of year    
        # print(time_emb.shape)    

        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[2], input_seq.shape[1]
        trg_seq_len = trg_seq.shape[2]
        
        #encoder
        input_comb = torch.concat([input_seq[:,:1,:], input_seq[:,-1:,:]],dim=1)# batch*2*nday 
        h, c = self.Encoder( input_comb  )
        # print('H',h.shape,'c',c.shape) #(n layer, batch, hiddensize)


        #time embedding 加到hidden states
        # print(time_emb.shape)
        h  = h  + time_emb[:,:,:self.hidden_size].permute((1,0,2)) + doy_emb[:,:,:self.hidden_size].permute((1,0,2))
        # Time embed is (B, 1, hidden_size)
        # time embedding 加到 cell states
        c  = c  + time_emb[:,:,:self.hidden_size].permute((1,0,2)) + doy_emb[:,:,:self.hidden_size].permute((1,0,2)) # Time embed is (B, 1, hidden_size)
        
        #concate xt, target condition
        x_cond = torch.concat([x,trg_seq], dim=1)#concat by channel
        # print(time_emb.shape, trg_seq.shape, x.shape)#batch, channel,length
        
        # time embedding  加到 decoder inputs 
        x_cond = x_cond + time_emb[:,:,self.hidden_size:] + doy_emb[:,:,self.hidden_size:] 

        #decoder
        outputs = self.Decoder(x_cond, h, c)
            
        
        return outputs
        
        # x = x * (scale + 1) + shift
#%%
class Seq2Seq_diff_doy_v2(nn.Module):
    """
    doy:   time embedding multiply with inputs??
    in:  (batch, length, channel)
    enc: encoder inputs
    decoder: xt and decoder inputs
    out: (batch, length, channel)
    
    """
   
    def __init__(self, enc_channel_num, dec_channel_num, 
                 hidden_size, num_layers, output_size, batch_size,
                 time_dim, doy_dim , device="cuda"):

        super().__init__()
        self.dec_channel_num = dec_channel_num
        self.hidden_size = hidden_size
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim*2),
            nn.SiLU(),
            nn.Linear(time_dim*2, hidden_size+dec_channel_num+1),
        )
        self.doy_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(doy_dim),
            nn.Linear(doy_dim, doy_dim*2),
            nn.SiLU(),
            nn.Linear(doy_dim*2, hidden_size+dec_channel_num+1),
        )
        
        self.output_size = output_size
        self.Encoder = Encoder(enc_channel_num, hidden_size, num_layers, batch_size, device=device)
        self.Decoder = Decoder(dec_channel_num, hidden_size, num_layers, output_size, batch_size)
        self.device = device
        # self.head=CMAL(n_in=3,n_out=12,n_hidden=100)


    def forward(self, x, input_seq, trg_seq, timestep): 
        """x: x_t ( Batch,1,trg_seq_length )
        
        """
        time_emb = self.time_mlp(timestep) 
           
        doy_emb = self.doy_emb(input_seq[:,1,-1:])# batch*1,   day of year    
        # print(time_emb.shape)    

        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[2], input_seq.shape[1]
        trg_seq_len = trg_seq.shape[2]
        
        #encoder
        input_comb = torch.concat([input_seq[:,:1,:], input_seq[:,-1:,:]],dim=1)# batch*2*nday 
        h, c = self.Encoder( input_comb  )
        # print('H',h.shape,'c',c.shape) #(n layer, batch, hiddensize)


        #time embedding 加到hidden states
        # print(time_emb.shape)
        h  = (h  + time_emb[:,:,:self.hidden_size].permute((1,0,2))) * doy_emb[:,:,:self.hidden_size].permute((1,0,2))
        # Time embed is (B, 1, hidden_size)
        # time embedding 加到 cell states
        c  = (c  + time_emb[:,:,:self.hidden_size].permute((1,0,2))) *  doy_emb[:,:,:self.hidden_size].permute((1,0,2)) # Time embed is (B, 1, hidden_size)
        
        #concate xt, target condition
        x_cond = torch.concat([x,trg_seq], dim=1)#concat by channel
        # print(time_emb.shape, trg_seq.shape, x.shape)#batch, channel,length
        
        # time embedding  加到 decoder inputs 
        x_cond = (x_cond + time_emb[:,:,self.hidden_size:]) * doy_emb[:,:,self.hidden_size:] 

        #decoder
        outputs = self.Decoder(x_cond, h, c)
            
        
        return outputs
        
        # x = x * (scale + 1) + shift

#%%
# model = Seq2Seq_diff_doy(2, 2, 
#                      4, 3, 
#                      1,  128, 
#                      32,8, device='cpu' )
# input_seq = torch.randn((128, 3, 7))
# trg_seq = torch.randn((128, 1, 3))
# time_step = torch.randn((128, 1 ))
# xt = torch.randn((128, 1 , 3))
# output = model(xt, input_seq, trg_seq, time_step)
# print(output.shape)






#%%
def _get_predict_last_n(cfg: Config) -> dict:
    predict_last_n = cfg.predict_last_n
    if isinstance(predict_last_n, int):
        predict_last_n = {'': predict_last_n}
    if len(predict_last_n) == 1:
        predict_last_n = {'': list(predict_last_n.values())[0]}  # if there's only one frequency, we omit its identifier
    return predict_last_n

# %%
