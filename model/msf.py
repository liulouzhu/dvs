import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, functional

# R1

class LIFNeuron(neuron.LIFNode):
    def __init__(self, tau=0.625, v_threshold=1.0, v_reset=0.0):
        super().__init__(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset

class LocalSpikingFeature(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilations=[1, 2, 3]):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            padding = (k - 1) * d // 2
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k, 
                             dilation=d, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    LIFNeuron(tau=0.625)
                )
            )
    
    def forward(self, x):
        # 输入形状: [T, B, C]
        x = x.permute(1, 2, 0)  # [B, C, T]
        features = []
        for conv in self.conv_blocks:
            out = conv(x)
            features.append(out)
        concat_feat = torch.cat(features, dim=1)  # [B, 3*C, T]
        return concat_feat.permute(2, 0, 1)  # [T, B, 3*C]

class GlobalSpikingFeature(nn.Module):
    def __init__(self, in_channels, out_channels, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.channel_reduce = nn.Conv1d(in_channels, out_channels, 1)
        self.gcn_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.lif = LIFNeuron(tau=0.625)
        
    def build_adjacency(self, x):
        # 特征相似度分支
        norm_x = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.bmm(norm_x.transpose(1,2), norm_x)  # [B, T, T]
        
        # 位置距离分支
        T = x.size(2)
        pos = torch.arange(T, device=x.device, dtype=torch.float)
        pos_diff = pos.view(1, T, 1) - pos.view(1, 1, T)
        dis_matrix = torch.exp(-torch.abs(pos_diff) / self.sigma)  # [1, T, T]
        
        return F.softmax(sim_matrix, dim=-1) + F.softmax(dis_matrix, dim=-1)

    def forward(self, x):
        # 输入形状: [T, B, C]
        x = x.permute(1, 2, 0)  # [B, C, T]
        reduced = self.channel_reduce(x)  # [B, C', T]
        
        adj = self.build_adjacency(reduced)  # [B, T, T]
        gcn_out = self.gcn_conv(reduced)     # [B, C', T]
        gcn_out = torch.bmm(gcn_out, adj)    # [B, C', T]
        
        return self.lif(gcn_out.permute(2, 0, 1))  # [T, B, C']

class TemporalInteractionModule(nn.Module):
    def __init__(self, channels, alpha=0.6):
        super().__init__()
        self.alpha = alpha
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            LIFNeuron(tau=0.625)
        )
        
    def forward(self, x):
        # 输入形状: [T, B, C]
        T, B, C = x.shape
        x = x.permute(1, 2, 0)  # [B, C, T]
        
        outputs = []
        prev_state = torch.zeros(B, C, 1, device=x.device)
        for t in range(T):
            current = x[:, :, t].unsqueeze(-1)  # [B, C, 1]
            
            # 时间交互
            conv_state = self.temporal_conv(prev_state)
            new_state = (1-self.alpha)*current + self.alpha*conv_state
            outputs.append(new_state.squeeze(-1))
            prev_state = new_state
            
        return torch.stack(outputs, dim=0)  # [T, B, C]

class MultiScaleSpikingFusion(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4, alpha=0.6):
        super().__init__()
        self.lsf = LocalSpikingFeature(in_channels, in_channels//3)
        self.gsf = GlobalSpikingFeature(in_channels, in_channels//reduction_ratio)
        self.tim = TemporalInteractionModule(in_channels + in_channels//reduction_ratio, alpha)
        self.fc = nn.Linear(in_channels + in_channels//reduction_ratio, 1)
        
    def forward(self, x):
        # 输入形状: [T, B, C]
        lsf_feat = self.lsf(x)  # [T, B, 3C]
        gsf_feat = self.gsf(x)  # [T, B, C/4]
        
        # 特征拼接
        fused = torch.cat([lsf_feat, gsf_feat], dim=-1)  # [T, B, 3C + C/4]
        
        # 时间交互
        tim_out = self.tim(fused)  # [T, B, C_total]
        
        # 异常评分
        scores = torch.sigmoid(self.fc(tim_out))  # [T, B, 1]
        
        return scores.squeeze(-1)  # [T, B]

        # return tim_out  # [T, B, C_total]
