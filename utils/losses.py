import torch
import torch.nn as nn
import torch.nn.functional as F

class DMIL_CenterLoss(nn.Module):
    def __init__(self, k=4, lambda_center=20):
        super().__init__()
        self.k = k
        self.lambda_center = lambda_center

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, T) 各片段的异常分数，经过sigmoid处理
            targets: (B,) 视频级别的标签 (0=正常，1=异常)
        
        Returns:
            loss: 组合损失值
        """
        # 动态多示例学习损失
        dmil_loss = self.dynamic_mil_loss(predictions, targets)
        
        # 中心损失
        center_loss = self.center_loss(predictions, targets)
        
        return dmil_loss + self.lambda_center * center_loss

    def dynamic_mil_loss(self, s, y):
        # 对每个视频的预测分数排序（降序）
        sorted_s, _ = torch.sort(s, dim=1, descending=True)  # (B, T)
        
        # 取前k个分数
        topk_s = sorted_s[:, :self.k]  # (B, k)
        
        # 计算每个样本的平均交叉熵损失
        loss_per_sample = []
        for i in range(s.size(0)):
            if y[i] == 1:  # 异常视频
                losses = -torch.log(topk_s[i] + 1e-8)
            else:          # 正常视频
                losses = -torch.log(1 - topk_s[i] + 1e-8)
            
            loss_per_sample.append(torch.mean(losses))
        
        return torch.mean(torch.stack(loss_per_sample))

    def center_loss(self, s, y):
        # 仅计算正常视频的损失
        normal_mask = (y == 0)
        if normal_mask.sum() == 0:
            return torch.tensor(0.0, device=s.device)
        
        # 获取正常视频的预测
        normal_s = s[normal_mask]  # (N_normal, T)
        
        # 计算每个视频的分数中心
        centers = torch.mean(normal_s, dim=1, keepdim=True)  # (N_normal, 1)
        
        # 计算均方差
        diff = normal_s - centers
        squared_diff = torch.pow(diff, 2)
        
        return torch.mean(squared_diff)
