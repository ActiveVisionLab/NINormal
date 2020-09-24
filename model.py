import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F

# We copied the TransformerEncoder, TransformerEncoderLayer and MultiheadAttention code from pytorch 1.3 code base
# so we can run with PyTorch >= 1.1.
from transplant_attn.transformer_from_torch import TransformerEncoder, TransformerEncoderLayer
from transplant_attn.MultiheadAttention_from_torch import MultiheadAttention


def run_attn(attn, x, use_ffn):
    """
    :param attn:        Attention functions, currently support nn.Multihead and TransformerEncoder
    :param x:           Input embeddings in shape (B, K, N, C), C denotes the No. of channels for each point, e.g. 512.
    :param use_ffn:     if choose use_ffn, the input is only x
    :return:            Soft attn output () and weights. The returned weights is None if use TransformerEncoder
    """
    attn_out_list = []
    weights = None
    B = x.shape[0]

    if use_ffn:
        for b in range(B):
            attn_out, weights = attn(x[b])  # x: (K, N, 512), attn_out: (K, N, 512), weights: (N, K, K)
            attn_out_list.append(attn_out)
    else:
        for b in range(B):
            attn_out, weights = attn(x[b], x[b], x[b])  # x: (K, N, 512), attn_out: (K, N, 512), weights: (N, K, K)
            attn_out_list.append(attn_out)

    # The weights are only for debug and visualisation.
    # We just return the (N, K) matrix, not the full (N, K, K) tensor.
    if weights is not None:
        weights = weights[:, 0, :]

    x = torch.stack(attn_out_list)
    return x, weights


class NINormalNet(nn.Module):
    def __init__(self):
        super(NINormalNet, self).__init__()

        self.conv0 = nn.Conv2d(3, 64, kernel_size=(1, 1))
        self.conv1 = nn.Conv2d(64, 256, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(1, 1))

        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)

        encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.0)
        self.attn = TransformerEncoder(encoder_layer, num_layers=1)

        self.fc0 = nn.Conv1d(512, 256, kernel_size=1)
        self.fc1 = nn.Conv1d(256, 64, kernel_size=1)
        self.fc2 = nn.Conv1d(64, 3, kernel_size=1)

        self.bn_fc0 = nn.BatchNorm1d(256)
        self.bn_fc1 = nn.BatchNorm1d(64)

        # the temperature is just a scalar learnable that controls the softmax strength
        self.temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)  # (1, )

    def forward(self, pts):
        """
        :param pts:     (B, K, N, 3)    input points
        :return:        (B, N, 3)       normals
        """
        x = pts.permute(0, 3, 1, 2)  # (B, 3, K, N)
        x = F.relu(self.bn0(self.conv0(x)))  # (B, C, K, N)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, C, K, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, C, K, N)

        # learn a temperature
        x = x / self.temp
        x, weights = run_attn(attn=self.attn, x=x.permute(0, 2, 3, 1), use_ffn=True)

        x, _ = torch.max(x, dim=1)  # (B, K, N, 512) -> (B, N, 512)
        x = x.transpose(1, 2)  # (B, C, N)
        x = F.relu(self.bn_fc0(self.fc0(x)))  # (B, C, N)
        x = F.relu(self.bn_fc1(self.fc1(x)))  # (B, C, N)
        x = self.fc2(x)  # (B, 3, N)
        x = x.transpose(1, 2)  # (B, N, 3)

        # normalise all normal predictions to unit length, we only care about angle in normal estimation task.
        x = F.normalize(x, dim=2)

        return x, weights
