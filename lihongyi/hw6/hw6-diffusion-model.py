import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

import torchvision
from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
import matplotlib.pyplot as plt
import os

torch.backends.cudnn.benchmark = True
torch.manual_seed(4096)

if torch.cuda.is_available():
  torch.cuda.manual_seed(4096)

# 定义线性beta调度函数，用于生成噪声的时间步长
def linear_beta_schedule(timesteps):
    """
    线性调度函数，用于生成每个时间步的beta值。
    beta值控制每一步添加的噪声量。
    """
    scale = 1000 / timesteps  # 根据时间步数调整比例
    beta_start = scale * 0.0001  # 起始beta值
    beta_end = scale * 0.02  # 结束beta值
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

# 从张量中提取特定时间步的值
def extract(a, t, x_shape):
    """
    从张量a中提取与时间步t对应的值，并调整形状以匹配输入张量x的形状。
    """
    b, *_ = t.shape
    out = a.gather(-1, t)  # 根据时间步t提取值
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # 调整形状

# 自定义数据集类，用于加载和预处理图像数据
class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size
    ):
        """
        初始化数据集，加载指定文件夹中的图像，并定义图像的预处理操作。
        """
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for p in Path(f'{folder}').glob(f'**/*.jpg')]  # 获取所有图像路径
        #################################
        ## TODO: Data Augmentation ##
        #################################
        self.transform = T.Compose([
            T.Resize(image_size),  # 调整图像大小
            T.ToTensor()  # 转换为张量
        ])

    def __len__(self):
        """
        返回数据集中图像的数量。
        """
        return len(self.paths)

    def __getitem__(self, index):
        """
        根据索引加载图像并应用预处理。
        """
        path = self.paths[index]
        img = Image.open(path)  # 打开图像
        return self.transform(img)  # 应用预处理

# 检查变量是否存在
def exists(x):
    return x is not None

# 返回默认值或调用默认函数
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 返回输入张量本身
def identity(t, *args, **kwargs):
    return t

# 无限循环数据加载器
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 检查数字是否有整数平方根
def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

# 将数字分组
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 图像归一化到[-1, 1]
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# 图像反归一化到[0, 1]
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# 残差模块，添加输入到输出
class Residual(nn.Module):
    """
    残差模块：将输入直接添加到输出，便于梯度流动。
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # 包装的子模块

    def forward(self, x, *args, **kwargs):
        """
        前向传播：计算子模块的输出并加上输入。
        """
        return self.fn(x, *args, **kwargs) + x

# 上采样模块
def Upsample(dim, dim_out = None):
    """
    上采样模块：将输入的分辨率扩大一倍。
    """
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),  # 最近邻插值上采样
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)  # 卷积调整通道数
    )

# 下采样模块
def Downsample(dim, dim_out = None):
    """
    下采样模块：将输入的分辨率缩小一半。
    """
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),  # 重排张量
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)  # 卷积调整通道数
    )

# 权重标准化卷积层
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# 层归一化模块
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

# 前归一化模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# 正弦位置嵌入模块
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# 构建块模块
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

# 残差网络块
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

# 线性注意力模块
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

# 注意力模块
class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# 模型
class Unet(nn.Module):
    """
    UNet模型：用于图像生成的主网络。
    """
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        """
        初始化UNet模型，定义网络结构。
        """
        super().__init__()

        # 确定维度

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # 时间嵌入

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 层

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        """
        前向传播：执行下采样、中间层和上采样操作。
        """
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
model = Unet(64)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            beta_schedule='linear',
            auto_normalize=True
    ):
        """
        初始化扩散模型，定义正向和逆向过程的参数。
        
        参数:
            model: UNet模型，用于预测噪声
            image_size: 图像大小
            timesteps: 扩散过程的时间步数
            beta_schedule: beta值的调度方式
            auto_normalize: 是否自动将图像归一化到[-1, 1]
        """
        super().__init__()
        # 验证模型的输入和输出通道是否一致
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels

        self.image_size = image_size

        # 根据调度类型选择beta调度函数
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 计算beta和相关参数
        betas = beta_schedule_fn(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # 累积乘积
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)  # 前一个累积乘积

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # 采样相关参数

        self.sampling_timesteps = timesteps  # 默认采样时间步数为训练时的时间步数

        # 注册缓冲区，用于存储扩散过程的参数
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 计算扩散过程q(x_t | x_{t-1})和其他参数

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 计算后验分布q(x_{t-1} | x_t, x_0)的参数

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # 上述公式等价于1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # 下面的log计算被截断，因为在扩散链的开始，后验方差为0

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 派生损失权重
        # snr - 信噪比

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()

        register_buffer('loss_weight', maybe_clipped_snr / snr)

        # 自动归一化数据到[-1, 1] - 可以通过设置为False来关闭

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        """
        根据噪声样本x_t和预测的噪声，重建原始图像x_0
        
        参数:
            x_t: t时刻的噪声图像
            t: 时间步
            noise: 预测的噪声
            
        返回:
            预测的原始图像x_0
        """
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """
        给定t时刻的图像x_t和原始图像x_0，计算添加的噪声
        
        参数:
            x_t: t时刻的噪声图像
            t: 时间步
            x0: 原始图像
            
        返回:
            添加的噪声
        """
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        """
        计算后验分布q(x_{t-1} | x_t, x_0)的参数
        
        参数:
            x_start: 原始图像x_0
            x_t: t时刻的噪声图像
            t: 时间步
            
        返回:
            后验分布的均值、方差和对数方差
        """
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, clip_x_start=False, rederive_pred_noise=False):
        """
        使用模型预测噪声和原始图像
        
        参数:
            x: t时刻的噪声图像
            t: 时间步
            clip_x_start: 是否将预测的原始图像限制在[-1, 1]范围内
            rederive_pred_noise: 是否从裁剪后的x_start重新计算噪声
            
        返回:
            预测的噪声和原始图像
        """
        model_output = self.model(x, t)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        """
        计算逆向过程p(x_{t-1} | x_t)的均值和方差
        
        参数:
            x: t时刻的噪声图像
            t: 时间步
            clip_denoised: 是否将去噪后的图像限制在[-1, 1]范围内
            
        返回:
            模型预测的均值、方差、对数方差和预测的原始图像
        """
        noise, x_start = self.model_predictions(x, t)

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int):
        """
        从t时刻采样x_{t-1}，执行单步去噪
        
        参数:
            x: t时刻的噪声图像
            t: 时间步
            
        返回:
            t-1时刻的图像和预测的原始图像
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0.  # t=0时不添加噪声
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        """
        从纯噪声开始，通过逐步去噪生成图像
        
        参数:
            shape: 生成图像的形状
            return_all_timesteps: 是否返回所有时间步的图像
            
        返回:
            生成的图像或所有时间步的图像
        """
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)  # 初始化为随机噪声
        imgs = [img]

        x_start = None

        ###########################################
        ## TODO: plot the sampling process ##
        ###########################################
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(img, t)  # 逐步去噪
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)  # 反归一化到[0, 1]
        return ret

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False):
        """
        生成指定批次大小的图像
        
        参数:
            batch_size: 批次大小
            return_all_timesteps: 是否返回所有时间步的图像
            
        返回:
            生成的图像或所有时间步的图像
        """
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps=return_all_timesteps)

    def q_sample(self, x_start, t, noise=None):
        """
        正向过程：向原始图像添加t步噪声
        
        参数:
            x_start: 原始图像
            t: 时间步
            noise: 添加的噪声，如果为None则随机生成
            
        返回:
            添加了t步噪声的图像
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        """
        损失函数：使用均方误差(MSE)
        
        返回:
            MSE损失函数
        """
        return F.mse_loss

    def p_losses(self, x_start, t, noise=None):
        """
        计算预测噪声的损失
        
        参数:
            x_start: 原始图像
            t: 时间步
            noise: 添加的噪声，如果为None则随机生成
            
        返回:
            加权的MSE损失
        """
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 添加噪声
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 预测噪声并计算损失
        model_out = self.model(x, t)

        loss = self.loss_fn(model_out, noise, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        # 对损失进行加权
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        """
        模型前向传播：随机选择时间步，添加噪声并计算损失
        
        参数:
            img: 原始图像
            
        返回:
            预测噪声的损失
        """
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()  # 随机选择时间步

        img = self.normalize(img)  # 归一化图像到[-1, 1]
        return self.p_losses(img, t, *args, **kwargs)  # 计算损失


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            train_batch_size=16,          # 训练批次大小
            gradient_accumulate_every=1,  # 梯度累积步数，可以增大有效批量大小
            train_lr=1e-4,                # 学习率
            train_num_steps=100000,       # 总训练步数
            ema_update_every=10,          # EMA模型更新频率
            ema_decay=0.995,              # EMA衰减率
            adam_betas=(0.9, 0.99),       # Adam优化器参数
            save_and_sample_every=1000,   # 保存模型和生成样本的频率
            num_samples=25,               # 每次采样生成的图像数量
            results_folder='./results',   # 结果保存路径
            split_batches=True,           # 是否拆分批次
            inception_block_idx=2048
    ):
        """
        训练器初始化：设置加速器、模型、数据集、优化器和其他训练参数
        """
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = Dataset(folder, self.image_size)  # 加载数据集
        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)  # 无限循环数据加载器

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)  # 优化器

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, ckpt):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(ckpt, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        """
        训练扩散模型的主要流程
        """
        accelerator = self.accelerator  # 获取加速器
        device = accelerator.device     # 获取设备

        # 使用tqdm创建进度条
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            # 训练循环，直到达到指定的训练步数
            while self.step < self.train_num_steps:

                total_loss = 0.

                # 梯度累积循环
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)  # 获取一批数据并移动到设备上

                    # 使用混合精度训练
                    with self.accelerator.autocast():
                        loss = self.model(data)  # 计算损失（这里调用的是GaussianDiffusion的forward方法）
                        loss = loss / self.gradient_accumulate_every  # 根据梯度累积次数缩放损失
                        total_loss += loss.item()  # 累加损失值

                    self.accelerator.backward(loss)  # 反向传播

                # 梯度裁剪，防止梯度爆炸
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 更新进度条描述，显示当前损失
                pbar.set_description(f'loss: {total_loss:.4f}')

                # 确保所有进程同步
                accelerator.wait_for_everyone()

                self.opt.step()  # 优化器更新参数
                self.opt.zero_grad()  # 清空梯度

                # 再次确保所有进程同步
                accelerator.wait_for_everyone()

                self.step += 1  # 增加步数计数
                
                # 主进程负责EMA更新和模型保存
                if accelerator.is_main_process:
                    self.ema.update()  # 更新EMA模型

                    # 定期保存模型和生成样本
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()  # 设置EMA模型为评估模式

                        # 生成样本图像
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim=0)

                        # 保存生成的图像
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'),
                                         nrow=int(math.sqrt(self.num_samples)))

                        # 保存模型检查点
                        self.save(milestone)

                # 更新进度条
                pbar.update(1)

        # 训练完成
        accelerator.print('training complete')

    def inference(self, num=1000, n_iter=5, output_path='./submission'):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        with torch.no_grad():
            for i in range(n_iter):
                batches = num_to_groups(num // n_iter, 200)
                all_images = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))[0]
                for j in range(all_images.size(0)):
                    torchvision.utils.save_image(all_images[j], f'{output_path}/{i * 200 + j + 1}.jpg')



path = '/kaggle/input/diffusion/faces/faces'
IMG_SIZE = 64             # Size of images, do not change this if you do not know why you need to change
batch_size = 16
train_num_steps = 10000        # total training steps
lr = 1e-3
grad_steps = 1            # gradient accumulation steps, the equivalent batch size for updating equals to batch_size * grad_steps = 16 * 1
ema_decay = 0.995           # exponential moving average decay

channels = 16             # Numbers of channels of the first layer of CNN
dim_mults = (1, 2, 4)        # The model size will be (channels, 2 * channels, 4 * channels, 4 * channels, 2 * channels, channels)

timesteps = 100            # Number of steps (adding noise)
beta_schedule = 'linear'

model = Unet(
    dim = channels,
    dim_mults = dim_mults
)

diffusion = GaussianDiffusion(
    model,
    image_size = IMG_SIZE,
    timesteps = timesteps,
    beta_schedule = beta_schedule
)

trainer = Trainer(
    diffusion,
    path,
    train_batch_size = batch_size,
    train_lr = lr,
    train_num_steps = train_num_steps,
    gradient_accumulate_every = grad_steps,
    ema_decay = ema_decay,
    save_and_sample_every = 1000
)

trainer.train()



ckpt = '/content/drive/MyDrive/ML 2023 Spring/model-55.pt'
trainer.load(ckpt)
trainer.inference()





