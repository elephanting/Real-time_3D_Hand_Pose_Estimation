import torch
import torch.nn as nn
from einops import rearrange

class dense_bn(nn.Module):
    def __init__(self, inc, ouc):
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(inc,ouc,True), nn.BatchNorm1d(ouc), nn.Sigmoid())
    def forward(self, x):
        return self.dense(x)
    

class IKNet(nn.Module):
    def __init__(self, inc, depth, width, joints=16):
        super().__init__()        
        self.dense = dense_bn(inc, width)

        self.dense_1 = dense_bn(width, width)
        self.dense_2 = dense_bn(width, width)
        self.dense_3 = dense_bn(width, width)
        self.dense_4 = dense_bn(width, width)
        self.dense_5 = dense_bn(width, width)

        # joints * 6D rotation and 1 for shape estimation
        self.dense_6 = nn.Linear(width, joints*6+1)
    def forward(self, x):
        x = rearrange(x,'b j c -> b (j c)', c=3)
        x = self.dense(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        x = self.dense_6(x)

        theta_raw = x[:, :-1]
        shape = x[:, -1]
        theta_raw = rearrange(theta_raw, 'b (j n) -> b j n', n=6)
        return theta_raw, shape