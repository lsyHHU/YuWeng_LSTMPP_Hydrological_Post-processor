import numpy as np
from scipy.stats import norm, rankdata
from scipy.interpolate import interp1d

class NQTransformer:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.train_q = None
        self.train_z = None
        self.cdf_table = None
        self.q_mesh = None

    def fit(self, q_data):
        """
        学习训练集的分布参数
        q_data: 形状为 (N,) 的一维数组
        """
        self.train_q = np.sort(q_data.flatten())
        n = len(self.train_q)
        
        # 1. 计算经验累计频率 (Weibull公式或简单的秩比)
        # 使用 (rank - 0.5) / n 避免 0 和 1 导致正态映射出现无穷大
        ranks = np.arange(1, n + 1)
        self.cdf_table = (ranks - 0.5) / n
        self.q_mesh = self.train_q
        
        return self

    def transform(self, q):
        """
        将流量 Q 转换为正态变量 Z (Forward NQT)
        """
        # 1. 通过插值找到 Q 在训练集分布中的概率值
        # 设定 fill_value 处理超出训练集范围的极值
        interp_func = interp1d(self.q_mesh, self.cdf_table, kind='linear', 
                               bounds_error=False, fill_value=(self.cdf_table[0], self.cdf_table[-1]))
        cdf = interp_func(q)
        
        # 2. 概率转为正态 Z-score
        z = norm.ppf(cdf)
        return z

    def inverse_transform(self, z):
        """
        将正态变量 Z 转换为流量 Q (Inverse NQT)
        """
        # 1. 正态 Z-score 转为概率
        cdf = norm.cdf(z)
        
        # 2. 概率转回流量空间
        interp_func = interp1d(self.cdf_table, self.q_mesh, kind='linear', 
                               bounds_error=False, fill_value=(self.q_mesh[0], self.q_mesh[-1]))
        q = interp_func(cdf)
        return q