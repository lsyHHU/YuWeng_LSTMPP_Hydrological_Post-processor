#%%
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import xarray
# from numba import njit
from sklearn import preprocessing
from torch.distributions import Categorical

# from neuralhydrology.utils.config import Config

def sample_gmm(pred: Dict[str, torch.Tensor], 
               n_samples: int = 100,
               number_of_targets: int = 3) -> Dict[str, torch.Tensor]:
    
    """
    鲁棒性更强的单高斯采样函数，支持自动识别 Batch 维度
    """
    mu = pred['mu']        # 可能是 [3, 250, 1] 或 [250, 3, 1]
    sigma = pred['sigma']
    device = mu.device

    # 1. 移除最后可能存在的冗余维度 (size=1)
    if mu.dim() == 3 and mu.shape[-1] == 1:
        mu = mu.squeeze(-1)
        sigma = sigma.squeeze(-1)

    # 2. 自动调整维度：确保输出形状为 [Batch_Size, Sequence_Length]
    # 如果第一维等于 output_length (3)，说明目前是 [Seq, Batch] 结构，需要转置
    if mu.shape[0] == number_of_targets and mu.shape[1] != number_of_targets:
        mu = mu.transpose(0, 1)      # 转为 [250, 3]
        sigma = sigma.transpose(0, 1)
    
    # 此时 mu 和 sigma 的形状统一为 [batch, targets]
    current_batch_size = mu.shape[0]

    # 3. 创建随机噪声 epsilon: [batch, targets, n_samples]
    eps = torch.randn(current_batch_size, number_of_targets, n_samples).to(device)
    
    # 4. 使用广播机制计算采样值
    # mu.unsqueeze(-1) 变为 [batch, targets, 1]
    # 结果 y_hat 为 [batch, targets, n_samples]
    y_hat = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * eps
    
    return {'y_hat': y_hat}


def sample_gaussians(ids: List[int], m_sub: torch.Tensor, s_sub: torch.Tensor,
                              p_sub: torch.Tensor) -> torch.Tensor:
    # unbound sampling:
    categorical = Categorical(p_sub)
    pis = categorical.sample().data
    mask_gmm = torch.zeros(p_sub.shape, dtype=torch.bool) \
        .to(device=p_sub.device) \
        .scatter_(2, pis.unsqueeze(2), True)

    # The ids are used for location-specific resampling for 'truncation' in '_handle_negative_values'
    values = s_sub \
        .data.new(s_sub[ids][mask_gmm[ids]].shape[0]) \
        .normal_() \
        .flatten() \
        .mul(s_sub[ids][mask_gmm[ids]]) \
        .add(m_sub[ids][mask_gmm[ids]])
    return values

def _handle_negative_values(values: torch.Tensor, sample_values: Callable,
                            scaler: Dict[str, Union[pd.Series, xarray.Dataset]], handle) -> torch.Tensor:
    """Handle negative samples that arise while sampling from the uncertainty estimates.

    Currently supports (a) 'clip' for directly clipping values at zero and (b) 'truncate' for resampling values
    that are below zero.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    values : torch.Tensor
        Tensor with the sampled values.
    sample_values : Callable
        Sampling function to allow for repeated sampling in the case of truncation-handling.
    scaler : Dict[str, Union[pd.Series, xarray.Dataset]]
        Scaler of the run.
    nth_target : int
        Index of the sampled target variable in cfg.target_variables.

    Returns
    -------
    torch.Tensor
        Bound values according to user specifications.
    """
    center = scaler.data_min_
    scale = scaler.data_range_
    if handle == 'clip':
        values = torch.clamp(values, min=normalized_zero)
    elif handle == 'truncate':
        values_smaller_zero = values < normalized_zero
        try_count = 0
        while torch.any(values_smaller_zero.flatten()):
            values[values_smaller_zero] = sample_values(values_smaller_zero)
            values_smaller_zero = values < normalized_zero
            try_count += 1
            if try_count >=3:
                break
    elif handle is None or handle == 'none':
        pass
    else:
        raise NotImplementedError(
            f"The option {handle} is not supported for handling negative samples!")

    return values
# %%
