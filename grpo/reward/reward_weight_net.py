import torch
import torch.nn as nn

class RewardWeightNet(nn.Module):
    """奖励权重网络，用于学习各个维度的权重"""
    def __init__(self, num_dimensions=4):
        super().__init__()
        self.weight_layer = nn.Sequential(
            nn.Linear(num_dimensions, 64),
            nn.ReLU(),
            nn.Linear(64, num_dimensions),
            nn.Softmax(dim=-1)  # 使用Softmax确保权重和为1
        )
        
    def forward(self, reward_features):
        """
        输入: reward_features (batch_size, num_dimensions) - 各个维度的原始奖励值
        输出: weights (batch_size, num_dimensions) - 对应的权重
        """
        return self.weight_layer(reward_features) 