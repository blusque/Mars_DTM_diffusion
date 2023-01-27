import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math
from einops import rearrange  # einops把张量的维度操作具象化，让开发者“想出即写出
from einops import reduce
from functools import partial

from utils import *

# 一种位置编码，前一半sin后一半cos
# eg：维数dim=5，time取1和2两个时间
# layer = SinusoidalPositionEmbeddings(5)
# embeddings = layer(torch.tensor([1,2]))
# return embeddings的形状是(2,5),第一行是t=1时的位置编码，第二行是t=2时的位置编码
# 额外连接(transformer原作位置编码实现)：https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb


class SinusoidalPositionEmbeddings(nn.Module):
    '''一种位置编码，前一半sin后一半cos
      eg：维数dim=5，time取1和2两个时间
      layer = SinusoidalPositionEmbeddings(5)
      embeddings = layer(torch.tensor([1,2]))
      return embeddings的形状是(2,5),第一行是t=1时的位置编码，第二行是t=2时的位置编码
      额外连接(transformer原作位置编码实现)：https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb'''

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1",
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    '''Block类，先卷积后GN归一化后siLU激活函数，若存在scale_shift则进行一定变换'''

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        # GN归一化 https://zhuanlan.zhihu.com/p/177853578
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

# 例：dim=8，dim_out=16,time_emb_dim=2, groups=8
# Block = ResnetBlock(8, 16, time_emb_dim=2, groups=8)
# a = torch.ones(1, 8, 64, 64)
# b = torch.ones(1, 2)
# result = Block(a, b)


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        # 如果time_emb_dim存在则有mlp层
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        # nn.Identity()有 https://blog.csdn.net/artistkeepmonkey/article/details/115067356
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)  # torch.Size([1, 16, 64, 64])

        if exists(self.mlp) and exists(time_emb):
            # time_emb为torch.Size([1, 2])
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)  # torch.Size([1, 16])
            # rearrange(time_emb, "b c -> b c 1 1")为torch.Size([1, 16, 1, 1])
            h = rearrange(condition, "b c -> b c 1 1") + h  # torch.Size([1, 16, 64, 64])

        h = self.block2(h)  # torch.Size([1, 16, 64, 64])
        # return最后补了残差连接 # torch.Size([1, 16, 64, 64])
        return h + self.res_conv(x)

# 可以参考class ResnetBlock进行理解


class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        # 如果time_emb_dim存在则有mlp层
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),  # Gaussian Error Linear Unit
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # qkv为一个元组，其中每一个元素的大小为torch.Size([b, hidden_dim, h, w])
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )  # qkv中每个元素从torch.Size([b, hidden_dim, h, w])变为torch.Size([b, heads, dim_head, h*w])
        q = q * self.scale  # q扩大dim_head**-0.5倍

        # sim有torch.Size([b, heads, h*w, h*w])
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)  # attn有torch.Size([b, heads, h*w, h*w])

        # [b, heads, h*w, h*w]和[b, heads, dim_head, h*w] 得 out为[b, heads, h*w, dim_head]
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y",
                        x=h, y=w)  # 得out为[b, hidden_dim, h, w]
        return self.to_out(out)  # 得 [b, dim, h, w]

# 和class Attention几乎一致


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y",
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,  # 下例中，dim=image_size=28
        init_dim=None,  # 默认为None，最终取dim // 3 * 2
        out_dim=None,  # 默认为None，最终取channels
        dim_mults=(1, 2, 4, 8),
        channels=1,  # 通道数默认为1
        with_time_emb=True,  # 是否使用embeddings
        resnet_block_groups=8,  # 如果使用ResnetBlock，groups=resnet_block_groups
        use_convnext=True,  # 是True使用ConvNextBlock，是Flase使用ResnetBlock
        convnext_mult=2,  # 如果使用ConvNextBlock，mult=convnext_mult
        ref_type='cat',  # 参考类型, 如果是'cat'为拼接, 如果是'add'为加和
    ):
        super().__init__()
        self.channels = channels
        self.ref_type = ref_type

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        self.ref_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        if self.ref_type == 'cat':
            dim = dim * 2
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]  # 从头到尾dim组成的列表
        if self.ref_type == 'cat':
            dims[0] = dims[0] * 2
        in_out = list(zip(dims[:-1], dims[1:]))  # dim对组成的列表
        # 使用ConvNextBlock或ResnetBlock
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])  # 初始化下采样网络列表
        self.ups = nn.ModuleList([])  # 初始化上采样网络列表
        num_resolutions = len(in_out)  # dim对组成的列表的长度

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)  # 是否到了最后一对

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in,
                                    time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, ref, time):
        assert ref.shape == x.shape, 'x and ref should have the same shape'

        x = self.init_conv(x)
        ref = self.ref_conv(ref)
        if self.ref_type == 'cat':
            x = torch.cat([x, ref], dim=1)
        elif self.ref_type == 'add':
            x = x + ref
        else:
            raise ValueError(
                f"ref_type should be 'cat' or 'add', while it is {self.ref_type} now.")

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
