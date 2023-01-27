from inspect import isfunction # inspect模块https://www.cnblogs.com/yaohong/p/8874154.html主要提供了四种用处：1.对是否是模块、框架、函数进行类型检查 2.获取源码 3.获取类或者函数的参数信息 4.解析堆栈
import torch.nn as nn

# x是否为None，不是None则返回True，是None则返回False
def exists(x):
    return x is not None

# 如果val非None则返回val，否则(如果d为函数则返回d(),否则返回d)
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# 上采样
def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

# 下采样
def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)
