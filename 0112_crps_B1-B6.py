import numpy as np
import xarray as xr
import pandas as pd
import datetime
import torch
import os
import copy
from torch.utils.data import DataLoader, TensorDataset
from numba import jit
from NQT import NQTransformer
# 导入你的自定义库
from head_gmm_residual import Seq2Seq, GaussianCRPSLoss
from utils import get_rolling_window_multistep, feature_label2src_trg_trgy
from sampling_GMM import sample_gmm
from auxiliary_function_transform_map import (
    Fit_power_transform_map, 
    power_transform_function, 
    power_inverse_transform_function
)
def merge_and_save_results(bid, mode, y_ens_wet, y_ens_dry, test_meta, cfg, nqt_engine):
    """
    将 Wet 和 Dry 模型的结果根据索引合并，并保存
    """
    idx_wet, idx_dry = test_meta
    all_idx = idx_wet + idx_dry
    if not all_idx:
        return
    
    # 1. 确定测试期总长度并初始化容器 [Time, Lead_Time, Ensemble]
    start_idx = min(all_idx)
    total_len = max(all_idx) - start_idx + 1
    y_merged = np.zeros((total_len, cfg['out_len'], 100))
    
    # 2. 回填 Wet 数据
    if y_ens_wet is not None:
        rel_idx_wet = np.array(idx_wet) - start_idx
        y_merged[rel_idx_wet] = y_ens_wet
        
    # 3. 回填 Dry 数据
    if y_ens_dry is not None:
        rel_idx_dry = np.array(idx_dry) - start_idx
        y_merged[rel_idx_dry] = y_ens_dry

    # 4. 按 Lead-day 分别保存结果
    for d in range(cfg['out_len']):
        save_path = f'0414data/lstmday{d+1}_{bid}_{mode}.txt'
        # 这里已经是经过 inverse_transform 的物理流量值
        np.savetxt(save_path, y_merged[:, d, :], delimiter=',', fmt='%.4f')
# --- 1. 核心辅助函数 ---
@jit(nopython=True)
def lyne_hollick_fast(q, alpha=0.925):
    b = np.zeros_like(q)
    b[0] = q[0]
    for i in range(1, len(q)):
        b[i] = min(alpha * b[i-1] + (1 - alpha) * q[i], q[i])
    return b

def get_residual_windows(in_len, out_len, z_obs, z_sim):
    """
    生成 NQT 空间下的特征和残差标签
    Feature: [z_sim, z_obs] (过去 in_len)
    Target Residual: z_obs - z_sim (未来 out_len)
    Decoder Input: z_sim (未来 out_len)
    """
    n_samples = len(z_obs) - in_len - out_len + 1
    res_z = z_obs - z_sim
    
    features = np.zeros((n_samples, 2, in_len))
    labels_res = np.zeros((n_samples, out_len))
    trg_sim = np.zeros((n_samples, 1, out_len))
    
    for i in range(n_samples):
        features[i, 0, :] = z_sim[i : i+in_len]
        features[i, 1, :] = z_obs[i : i+in_len]
        labels_res[i, :] = res_z[i+in_len : i+in_len+out_len]
        trg_sim[i, 0, :] = z_sim[i+in_len : i+in_len+out_len]
        
    return features, labels_res, trg_sim

def prepare_dataloaders(z_obs, z_sim, is_wet, date_range, date_train, date_val, date_test, cfg):
    f_all, l_res, t_sim = get_residual_windows(cfg['in_len'], cfg['out_len'], z_obs, z_sim)
    
    # 建立日期到索引的映射
    data_start = date_range[0]
    def get_indices(d_range, condition):
        offsets = (pd.to_datetime(d_range) - data_start).days
        indices = []
        for d in offsets:
            idx = d - cfg['in_len']
            if 0 <= idx < len(f_all) and is_wet[d] == condition:
                indices.append(idx)
        return indices

    def make_loader(idx, shuffle=False):
        if not idx: return None
        
        # [修改] 1. Encoder Input: 保持 (Batch, 2, 10)
        # 假设 head_gmm_residual 内部会处理维度，或者它期望 Channel First
        f_tensor = torch.tensor(f_all[idx], dtype=torch.float32) # 去掉 .permute(0, 2, 1)
        
        # 2. 标签残差: (Batch, 3, 1) 保持不变
        l_tensor = torch.tensor(l_res[idx], dtype=torch.float32).unsqueeze(-1)
        
        # [修改] 3. Decoder Input: 保持 (Batch, 1, 3)
        t_tensor = torch.tensor(t_sim[idx], dtype=torch.float32) # 去掉 .permute(0, 2, 1)
        
        ds = TensorDataset(f_tensor, l_tensor, t_tensor)
        return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle)

    return {
        'wet': (make_loader(get_indices(date_train, True), True), 
                make_loader(get_indices(date_val, True)), 
                make_loader(get_indices(date_test[:-cfg['out_len']], True))),
        'dry': (make_loader(get_indices(date_train, False), True), 
                make_loader(get_indices(date_val, False)), 
                make_loader(get_indices(date_test[:-cfg['out_len']], False))),
        'test_meta': (get_indices(date_test[:-cfg['out_len']], True), 
                      get_indices(date_test[:-cfg['out_len']], False))
    }

# --- 3. 训练与推理逻辑 ---
def run_residual_session(model, loaders, nqt_engine, config, info):
    train_ld, val_ld, test_ld = loaders
    bid, mode, regime = info
    device = config['device']
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config.get('wd', 0))
    
    # 使用 CRPS Loss
    loss_func = GaussianCRPSLoss() 
    
    # === 1. 训练与验证 (带早停) ===
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None # 用于存储最佳模型参数
    
    for epoch in range(config['epochs']):
        # --- Train ---
        model.train()
        train_loss = 0
        for f_, res_, t_ in train_ld:
            optimizer.zero_grad()
            pred_res_dist = model(f_.to(device), t_.to(device))
            loss = loss_func(pred_res_dist, {'y': res_.to(device)})
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- Validation (用于早停) ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            if val_ld is not None:
                for f_, res_, t_ in val_ld:
                    pred = model(f_.to(device), t_.to(device))
                    loss = loss_func(pred, {'y': res_.to(device)})
                    val_loss += loss.item()
                val_loss /= len(val_ld) # 计算平均 Loss
            else:
                # 如果没有验证集，就用训练 Loss 代替 (虽不推荐但不报错)
                val_loss = train_loss / len(train_ld)

        # --- Early Stopping Check ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 深度拷贝保存当前最佳模型参数 (在内存中保存，比写磁盘快)
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # === 2. 恢复最佳模型 ===
    # 这一步非常关键！确保预测时用的是验证集上表现最好的模型，而不是过拟合的最后一代
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # === 3. 推理阶段 ===
    if test_ld is None:
        return None

    model.eval()
    results = []
    with torch.no_grad():
        for f_, res_, t_ in test_ld:
            pred = model(f_.to(device), t_.to(device))
            
            mu_res = pred['mu'].cpu().numpy()     # [B, 3, 1]
            sig_res = pred['sigma'].cpu().numpy() # [B, 3, 1]
            
            # 维度修正: Decoder Input 转置
            z_sim_future = t_.permute(0, 2, 1).cpu().numpy() # [B, 3, 1]
            
            batch_ens = []
            # 循环生成 100 个集合成员
            for _ in range(100):
                # 采样残差
                eps = np.random.normal(mu_res, sig_res) 
                # 叠加到模拟值上
                z_post = z_sim_future + eps
                # NQT 反变换
                q_post = nqt_engine.inverse_transform(z_post)
                batch_ens.append(q_post)
            
            # Stack 后形状: [B, 3, 1, 100] -> 取第0维 -> [B, 3, 100]
            results.append(np.stack(batch_ens, axis=-1)[:, :, 0, :])

    # 拼接所有 Batch
    if len(results) > 0:
        return np.concatenate(results, axis=0)
    return None
# --- 2. 主程序入口 ---
if __name__ == "__main__":
    cfg = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 512, 'hidden_size': 150, 'in_len': 10, 'out_len': 3,
        'lr': 1e-3, 'wd': 1e-4, 'epochs': 5000, 'patience': 50, 'model_dir': 'model'
    }
    
    if not os.path.exists('model'): os.makedirs('model')
    if not os.path.exists('0414data'): os.makedirs('0414data')

    # 定义日期（放在循环外，确保作用域全局可见）
    date_train = pd.date_range(start='1962-01-01', end='1985-12-31')
    date_val   = pd.date_range(start='1986-01-01', end='1987-12-31')
    date_test  = pd.date_range(start='1988-01-01', end='1997-12-31')
    date_range = pd.date_range(start='1960-01-01', end='1998-12-31')

    basin_info = {
    '07378500': 3315,  # B1
    '03451500': 2448,  # B2
    '03054500': 2372,  # B3
    '07186000': 3015,  # B4
    '01608500': 3810,  # B5
    '01643000': 2116   # B6
}
    mode_list = ['swb','gr4j','isba','noah','prms','sac','swap','vic']
    def get_new_model():
        return Seq2Seq(
            hidden_size=cfg['hidden_size'], 
            num_layers=2,
            batch_size=cfg['batch_size'], 
            device=cfg['device'],
            dropout=0.2
        ).to(cfg['device'])

    for bid, area in basin_info.items():
        input_csv = f'cal/{bid}.csv'
        hb = pd.read_csv(input_csv)
        obs_raw = hb['OBS_Q'].values
        if not os.path.exists(input_csv): continue
        nqt = NQTransformer().fit(obs_raw[(date_range >= '1962-01-01') & (date_range <= '1985-12-31')])
        is_wet = obs_raw > lyne_hollick_fast(obs_raw)
        
        for mode in mode_list:
            print(f"正在处理: 流域 {bid}, 模式 {mode}")
            sim_raw = hb[mode].values
            # 2. 转换到 Z 空间
            z_obs = nqt.transform(obs_raw)
            z_sim = nqt.transform(sim_raw)
            # 3. 准备数据负载
            # 找到这一行并修改：将第二个 date_train 改为 date_val
            loaders = prepare_dataloaders(z_obs, z_sim, is_wet, date_range, date_train, date_val, date_test, cfg)
            
            # 5. 分别运行 Wet 和 Dry 会话
            model_wet = get_new_model()
            y_ens_wet = run_residual_session(model_wet, loaders['wet'], nqt, cfg, (bid, mode, 'Wet'))

            model_dry = get_new_model()
            y_ens_dry = run_residual_session(model_dry, loaders['dry'], nqt, cfg, (bid, mode, 'Dry'))
            
            # 6. 结果合并与保存 (自动处理索引对齐和 Lead-day 拆分)
            merge_and_save_results(bid, mode, y_ens_wet, y_ens_dry, loaders['test_meta'], cfg, nqt)
            
            print(f">>> {bid}_{mode} 已完成并保存。")