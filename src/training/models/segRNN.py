"""
Code taken from: https://github.com/hughxx/tsf-new-paper-taste/blob/master/models/PatchMixer.py
"""

import torch
import torch.nn as nn
import torch.fft


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mean = None
        self.stdev = None
        self.last = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Backbone(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super(Backbone, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len

        # Patching
        self.patch_len = patch_len  # 16
        self.stride = stride  # 8
        self.patch_num = patch_num = int((seq_len - patch_len) / stride + 1)
        self.padding_patch = padding_patch
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num = patch_num = patch_num + 1

        # 1
        d_model = patch_len * patch_len
        self.embed = nn.Linear(patch_len, d_model)
        self.dropout_embed = nn.Dropout(0.3)

        # 2
        # self.lin_res = nn.Linear(seq_len, pred_len) # direct res, seems bad
        self.lin_res = nn.Linear(patch_num * d_model, pred_len)
        self.dropout_res = nn.Dropout(0.3)

        # 3.1
        self.depth_conv = nn.Conv1d(
            patch_num, patch_num, kernel_size=patch_len, stride=patch_len, groups=patch_num)
        self.depth_activation = nn.GELU()
        self.depth_norm = nn.BatchNorm1d(patch_num)
        self.depth_res = nn.Linear(d_model, patch_len)
        # 3.2
        # self.point_conv = nn.Conv1d(patch_len,patch_len,kernel_size=1, stride=1)
        # self.point_activation = nn.GELU()
        # self.point_norm = nn.BatchNorm1d(patch_len)
        self.point_conv = nn.Conv1d(
            patch_num, patch_num, kernel_size=1, stride=1)
        self.point_activation = nn.GELU()
        self.point_norm = nn.BatchNorm1d(patch_num)
        # 4
        self.mlp = Mlp(patch_len * patch_num, pred_len * 2, pred_len)

    def forward(self, x):  # B, L, D -> B, H, D
        B, _, D = x.shape
        L = self.patch_num
        P = self.patch_len

        # z_res = self.lin_res(x.permute(0, 2, 1)) # B, L, D -> B, H, D
        # z_res = self.dropout_res(z_res)

        # 1
        if self.padding_patch == 'end':
            # B, L, D -> B, D, L -> B, D, L
            z = self.padding_patch_layer(x.permute(0, 2, 1))
        z = z.unfold(dimension=-1, size=self.patch_len,
                     step=self.stride)  # B, D, L, P
        z = z.reshape(B * D, L, P, 1).squeeze(-1)
        z = self.embed(z)  # B * D, L, P -> # B * D, L, d
        z = self.dropout_embed(z)

        # 2
        # B * D, L, d -> B, D, L * d -> B, D, H
        z_res = self.lin_res(z.reshape(B, D, -1))
        z_res = self.dropout_res(z_res)

        # 3.1
        res = self.depth_res(z)  # B * D, L, d -> B * D, L, P
        z_depth = self.depth_conv(z)  # B * D, L, d -> B * D, L, P
        z_depth = self.depth_activation(z_depth)
        z_depth = self.depth_norm(z_depth)
        z_depth = z_depth + res
        # 3.2
        z_point = self.point_conv(z_depth)  # B * D, L, P -> B * D, L, P
        z_point = self.point_activation(z_point)
        z_point = self.point_norm(z_point)
        z_point = z_point.reshape(B, D, -1)  # B * D, L, P -> B, D, L * P

        # 4
        z_mlp = self.mlp(z_point)  # B, D, L * P -> B, D, H

        return (z_res + z_mlp).permute(0, 2, 1)


class PatchMixer(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len, patch_len, stride, padding_patch):
        super(PatchMixer, self).__init__()
        self.rev = RevIN(enc_in)

        self.backbone = Backbone(
            seq_len, pred_len, patch_len, stride, padding_patch)

        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x):
        z = self.rev(x, 'norm')  # B, L, D -> B, L, D
        z = self.backbone(z)  # B, L, D -> B, H, D
        z = self.rev(z, 'denorm')  # B, L, D -> B, H, D
        return z
