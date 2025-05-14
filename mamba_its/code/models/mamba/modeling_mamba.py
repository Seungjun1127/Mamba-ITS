import math
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from .configuration_mamba import MambaConfig

# Output dataclass (Swin/ViT 스타일)
@dataclass
class MambaImageClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# PreTrainedModel 상속
class MambaPreTrainedModel(PreTrainedModel):
    config_class = MambaConfig
    base_model_prefix = "mamba"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

# Patch Embedding
class MambaPatchEmbedding(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, hidden, H', W')
        B, hidden, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden)
        x = self.norm(x)
        return x, (H, W)

# SSM Block (Placeholder)
class MambaSSMBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
    def forward(self, x):
        # x: (B, N, C)
        return self.norm(self.linear(x))

# Patch Merging (계층적 다운샘플링)
class MambaPatchMerging(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.reduction = nn.Linear(input_dim * 4, output_dim)
        self.norm = nn.LayerNorm(input_dim * 4)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        # 2x2 패치 병합
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, (H // 2, W // 2)

# 2D Positional Encoding
class Mamba2DPositionalEncoding(nn.Module):
    def __init__(self, config: MambaConfig, H, W):
        super().__init__()
        self.row_embed = nn.Parameter(torch.zeros(1, H, 1, config.hidden_size))
        self.col_embed = nn.Parameter(torch.zeros(1, 1, W, config.hidden_size))
    def forward(self, x, H, W):
        # x: (B, N, C), N=H*W
        pos = self.row_embed[:, :H] + self.col_embed[:, :, :W]
        pos = pos.view(1, H * W, -1)
        return x + pos

# Mamba Vision Backbone
class MambaModel(MambaPreTrainedModel):
    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.patch_embed = MambaPatchEmbedding(config)
        self.pos_embed = None  # 동적 생성
        self.layers = nn.ModuleList([
            MambaSSMBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.patch_merge = MambaPatchMerging(config.hidden_size, config.hidden_size * 2) if config.patch_merge else None
        self.norm = nn.LayerNorm(config.hidden_size * (2 if config.patch_merge else 1))
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, pixel_values):
        x, (H, W) = self.patch_embed(pixel_values)
        if self.pos_embed is None or self.pos_embed.row_embed.shape[1] != H or self.pos_embed.col_embed.shape[2] != W:
            self.pos_embed = Mamba2DPositionalEncoding(self.config, H, W).to(x.device)
        x = self.pos_embed(x, H, W)
        for layer in self.layers:
            x = layer(x)
        if self.patch_merge is not None:
            x, (H, W) = self.patch_merge(x, H, W)
        x = self.norm(x)
        # Global average pooling
        x = x.transpose(1, 2)  # (B, C, N)
        x = self.pool(x).squeeze(-1)  # (B, C)
        return x

# Image Classification Head
class MambaForImageClassification(MambaPreTrainedModel):
    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.num_labels = config.num_classes
        self.mamba = MambaModel(config)
        self.classifier = nn.Linear(self.mamba.norm.normalized_shape[0], self.num_labels)
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> Union[Tuple, MambaImageClassifierOutput]:
        features = self.mamba(pixel_values)
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif labels.dtype == torch.long or labels.dtype == torch.int:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        return MambaImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
