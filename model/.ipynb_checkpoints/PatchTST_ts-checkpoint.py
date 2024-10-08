# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Tuple

import numpy as np
import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)  # type: ignore

    @staticmethod
    def _init_weight(out: torch.Tensor) -> torch.Tensor:
        """
        Features are not interleaved. The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        # set early to avoid an error in pytorch-1.8+
        out.requires_grad = False

        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(  # type: ignore
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen x ...]."""
        _, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class PatchTST_ts(nn.Module):
    """
    Module implementing the PatchTST model for forecasting as described in
    https://arxiv.org/abs/2211.14730 extended to be probabilistic.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
        Default: ``StudentTOutput()``.
    """

    def __init__(self,configs):
        super().__init__()

        self.pred_len = configs.pred_len
        self.seq_len= configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.hidden_dim = configs.hidden_dim
        self.padding_patch = configs.padding_patch
        self.n_heads = configs.n_heads
        #self.d_linear = configs.d_linear
        self.drop_out = configs.drop_out
        self.activation = configs.activation
        self.norm_first = configs.norm_first
        self.num_layers = configs.num_layers
        self.subtract_last = configs.subtract_last
        self.features = configs.features
        self.individual = configs.individual
        self.d_ff = configs.d_ff
        self.norm_first = configs.norm_first

        
        self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)
        
        if self.padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num += 1

        # project from patch_len + 2 features (loc and scale) to d_model
        self.patch_proj = nn.Linear(self.patch_len , self.hidden_dim)

        self.positional_encoding = SinusoidalPositionalEmbedding(
            self.patch_num, self.hidden_dim 
        )

        layer_norm_eps: float = 1e-5
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim ,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.drop_out,
            activation=self.activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=self.norm_first,
        )
        encoder_norm = nn.LayerNorm(self.hidden_dim , eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, self.num_layers, encoder_norm
        )

        self.flatten = nn.Linear(
            self.hidden_dim  * self.patch_num, self.pred_len 
        )


    def forward( self, inputs) :
        # x: [Batch, Channel,  Input length]
        batch_size = inputs.shape[0]
        inputs = inputs.permute(0,2,1)
        n_vars = inputs.shape[1]
        
        if self.padding_patch == "end":
            inputs  = self.padding_patch_layer(inputs ) # x: [Batch, Channel, patch_num, patch_len]    
        inputs = inputs.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # x: [Batch, Channel, patch_num, patch_len]        
        # project patches
        enc_in = self.patch_proj(inputs)
        enc_in  = torch.reshape(enc_in , (enc_in .shape[0]*enc_in .shape[1],enc_in .shape[2],enc_in .shape[3])) 
        embed_pos = self.positional_encoding(enc_in.size())

        # transformer encoder with positional encoding
        enc_out = self.encoder(enc_in + embed_pos)
        enc_out  = torch.reshape(enc_out, (batch_size, n_vars, enc_out.shape[-2], enc_out.shape[-1])) # x: [bs x nvars x patch_num x hidden_dim]

        # flatten and project to prediction length * d_model
        flatten_out = self.flatten(enc_out.flatten(start_dim=2))
        return flatten_out.permute(0,2,1)