import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange, repeat
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import numpy as np


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)



class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8
        
        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        
        x=x[:,:,:h,:w]
        
        return x

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2), out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')

        return out



##########################################################################
class EVS(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False, LayerNorm_type='WithBias', att=False, idx=3, patch=128):
        super(EVS, self).__init__()

        self.att = att
        self.idx = idx
        if self.att:
            self.norm1 = LayerNorm(dim)
            self.attn = SS2D(d_model=dim, patch=patch)

        self.norm2 = LayerNorm(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

        self.kernel_size = (patch, patch)

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def forward(self, x):
        if self.att:

            if self.idx % 2 == 1:
                x = torch.flip(x, dims=(-2, -1)).contiguous()
            if self.idx % 2 == 0:
                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
#            if self.idx % 4 == 3:
#                x = torch.flip(x, dims=(-2, -1)).contiguous()
#            if self.idx % 4 == 0:
#                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()

            x = self.grids(x)
            x = x + self.attn(self.norm1(x))
            x = self.grids_inverse(x)

        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- EVSSM -----------------------
class EVSSM(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[6, 6, 12],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 ):
        super(EVSSM, self).__init__()

        self.encoder = True

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = EVS(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                                     patch=384)
            self.encoder_level1.add_module(f"block{i}", block)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = EVS(dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                                     patch=192)
            self.encoder_level2.add_module(f"block{i}", block)

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = EVS(dim=dim * 4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                                     patch=96)
            self.encoder_level3.add_module(f"block{i}", block)

        self.decoder_level3 = nn.Sequential()
        for i in range(num_blocks[2]):
            block = EVS(dim=dim * 4, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                                     patch=96)
            self.decoder_level3.add_module(f"block{i}", block)

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.decoder_level2 = nn.Sequential()
        for i in range(num_blocks[1]):
            block = EVS(dim=dim * 2, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                                     patch=192)
            self.decoder_level2.add_module(f"block{i}", block)

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential()
        for i in range(num_blocks[0]):
            block = EVS(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias, att=True, idx=i,
                                     patch=384)
            self.decoder_level1.add_module(f"block{i}", block)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, img1=None):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = inp_dec_level2 + out_enc_level2

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = inp_dec_level1 + out_enc_level1
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

#
##
###
# from torchvision.models import resnet50
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from thop import clever_format
# from thop import profile
#
#
# model = fftformer().cuda()
# device = torch.device("cuda")
## 创建输入网络的tensor
# tensor = (torch.rand((1, 3, 256, 256),device=device),)
#
# input = torch.randn((1, 3, 256, 256),device=device)
# flops, params = profile(model, inputs=(input, ))
# print(flops/1e9,params/1e6)
## 创建resnet50网络
#
#
## 分析FLOPs
# flops = FlopCountAnalysis(model, tensor)
# print("FLOPs: ", flops.total())
#
## 分析parameters
# print(parameter_count_table(model))
