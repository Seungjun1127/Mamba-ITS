# coding=utf-8
# Copyright 2024 NVIDIA. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Mamba Vision model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mamba import MambaConfig

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "MambaConfig"
_FEAT_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "nvidia/MambaVision-T-1K"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "nvidia/MambaVision-T-1K"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/MambaVision-T-1K",
    # See all mambaVision models at https://huggingface.co/models?filter=mambaVision
]

@dataclass
class MambaImageClassifierOutput(ModelOutput):
    """
    Mamba outputs for image classification.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class MambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MambaConfig
    base_model_prefix = "mamba"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class MambaPatchEmbedding(nn.Module):
    def __init__(self, config):
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

class MambaSSMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.hidden_size
        self.d_state = getattr(config, 'd_state', 16)
        self.dropout = getattr(config, 'dropout', 0.1)
        
        # SSM layer
        self.ssm = SelectiveScan(
            d_model=self.d_model,
            d_state=self.d_state,
            dropout=self.dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, 4 * self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.d_model, self.d_model),
            nn.Dropout(self.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        
    def forward(self, x):
        # SSM block with residual connection
        residual = x
        x = self.norm1(x)
        x = self.ssm(x)
        x = residual + x
        
        # Feed-forward network with residual connection
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, None  # None for attention (not used in Mamba)

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

class SelectiveScan(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(d_model, d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state, 1))
        self.C = nn.Parameter(torch.randn(d_model, 1, d_state))
        self.D = nn.Parameter(torch.randn(d_model, 1, 1))
        
        # Selective mechanism
        self.selective_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        
        # Selective gating
        gate = self.selective_gate(x)  # (B, L, D)
        x = x * gate
        
        # Initialize state
        state = torch.zeros(B, D, self.d_state, device=x.device)
        
        # Process sequence
        outputs = []
        for t in range(L):
            # Update state
            state = torch.matmul(state, self.A) + torch.matmul(x[:, t:t+1].transpose(1, 2), self.B)
            # Compute output
            output = torch.matmul(state, self.C.transpose(1, 2)) + self.D * x[:, t:t+1]
            outputs.append(output)
        
        # Stack outputs
        output = torch.cat(outputs, dim=1)  # (B, L, D)
        
        # Residual connection and normalization
        output = self.norm(output + x)
        output = self.dropout(output)
        
        return output

class MambaModel(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.num_features = config.hidden_size
        
        # Patch Embedding
        self.patch_embed = MambaPatchEmbedding(config)
        
        # SSM layers
        self.layers = nn.ModuleList([
            MambaSSMBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Patch Merging (if enabled)
        self.patch_merge = MambaPatchMerging(config.hidden_size, config.hidden_size * 2) if config.patch_merge else None
        
        # Layer Norm
        self.norm = nn.LayerNorm(config.hidden_size * (2 if config.patch_merge else 1))
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MambaModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Patch Embedding
        x, (H, W) = self.patch_embed(pixel_values)
        
        # Process through SSM layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
                
            layer_outputs = layer(x)
            x = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Optional patch merging
        if self.patch_merge is not None:
            x, (H, W) = self.patch_merge(x, H, W)
            
        x = self.norm(x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (B, C, N)
        pooled_output = self.pool(x).squeeze(-1)  # (B, C)
        
        if not return_dict:
            return (x, pooled_output) + (all_hidden_states,) + (all_attentions,)
            
        return MambaModelOutput(
            last_hidden_state=x,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

class MambaForImageClassification(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mamba = MambaModel(config)
        
        # Classifier head
        self.classifier = (
            nn.Linear(self.mamba.num_features, config.num_labels) 
            if config.num_labels > 0 
            else nn.Identity()
        )
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MambaImageClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mamba(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MambaImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )