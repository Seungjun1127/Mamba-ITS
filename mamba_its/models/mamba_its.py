import torch
import torch.nn as nn

# 블록별 임포트 (아직 미구현)
from .blocks.local_ssm import LocalSSMBlock
from .blocks.global_ssm import GlobalSSMBlock
from .blocks.patch_merge import PatchMerging
from .blocks.pos_embed import PositionalEncoding2D
from .blocks.window_interact import WindowInteraction

class MambaITSModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 패치 임베딩
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size
        )
        # 2D 위치 인코딩
        self.pos_embed = PositionalEncoding2D(config)
        # 계층적 stage별 블록 쌓기
        self.stages = nn.ModuleList()
        in_dim = config.embed_dim
        h, w = config.img_size // config.patch_size, config.img_size // config.patch_size
        for i, depth in enumerate(config.depths):
            stage_blocks = []
            for d in range(depth):
                if config.use_local_ssm:
                    stage_blocks.append(LocalSSMBlock(in_dim, config))
                if config.use_global_ssm:
                    stage_blocks.append(GlobalSSMBlock(in_dim, config))
                if config.window_interaction:
                    stage_blocks.append(WindowInteraction(in_dim, config))
            self.stages.append(nn.Sequential(*stage_blocks))
            # 패치 머징
            if i < len(config.depths) - 1:
                self.stages.append(PatchMerging((h, w), in_dim))
                h, w = h // 2, w // 2
                in_dim = in_dim * 2

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        for stage in self.stages:
            x = stage(x)
        return x
