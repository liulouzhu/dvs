import torch
import torch.nn as nn

class AnomalyScorer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 全连接层将特征维度映射到1维
        self.fc = nn.Linear(input_dim, 1)
        # 参数初始化
        self._init_weights()
        
    def _init_weights(self):
        """Xavier初始化全连接层参数"""
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.1)
        
    def forward(self, x):
        """
        输入:
            x: (B, T, D) 来自MSF模块的输出特征
        输出:
            scores: (B, T) 每个时间步的异常分数
        """
        # 全连接层
        scores = self.fc(x)  # (B, T, 1)
        # 压缩最后一个维度
        scores = scores.squeeze(-1)  # (B, T)
        # Sigmoid激活
        return torch.sigmoid(scores)