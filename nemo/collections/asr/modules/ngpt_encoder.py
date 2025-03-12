# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.distributed
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, open_dict
import random
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging
from nemo.collections.asr.models.configs import CacheAwareStreamingConfig
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
try:
    from flash_attn import flash_attn_func
except ImportError:

    def flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=1.0,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    ):
        """Quick and dirty implementation for prototyping."""
        return nn.functional.softmax(q @ (k * softmax_scale).transpose(2, 3), dim=-1) @ v


__all__ = ['NGPTEncoder']


class NGPTEncoder(NeuralModule,StreamingEncoder, Exportable, AccessMixin):
    """
    Transformer encoder based on nGPT for ASR.
    Based on this paper:
    'nGPT: Normalized Transformer with Representation Learning on the Hypersphere' by Ilya Loshchilov et al.
    https://github.com/NVIDIA/ngpt
    """

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim, device=dev)
        input_example_length = torch.randint(max_dim // 4, max_dim, (max_batch,), device=dev, dtype=torch.int64)
        all_input_example = tuple([input_example, input_example_length])
        return all_input_example

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def input_types_for_export(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types_for_export(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def disabled_deployment_input_names(self):
        return set()

    @property
    def disabled_deployment_output_names(self):
        return set()

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        base_scale: float = 1 / (1024**0.5),  # 1/sqrt(d_model)
        n_heads=4,
        causal_downsampling=False,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=256,
        att_context_size=None,
        att_context_probs=None,
        att_context_style='chunked_limited',
        use_bias=False,
        dropout=0.1,
        use_nGPT=True,
    ):
        super().__init__()
        (
            self.att_context_size_all,
            self.att_context_size,
            self.att_context_probs,
            self.conv_context_size,
        ) = self._calc_context_sizes(
            att_context_style=att_context_style,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
        )
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor
        self._feat_out = d_model
        if subsampling == "ngpt-frame-stack":
            self.pre_encode = NGPTStackingSubsampling(
                subsampling_factor=subsampling_factor,
                feat_in=feat_in,
                feat_out=d_model,
                use_bias=use_bias,
                base_scale=base_scale,
            )
        else:  # temporary back-compat with 1st expts
            self.pre_encode = ConvSubsampling(
                subsampling=subsampling,
                subsampling_factor=subsampling_factor,
                feat_in=feat_in,
                feat_out=d_model,
                conv_channels=subsampling_conv_channels,
                subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
                activation=nn.ReLU(True),
                is_causal=causal_downsampling,
            )
        self.ngpt = GPT(
            config=GPTConfig(
                n_layer=n_layers,
                n_head=n_heads,
                n_embd=d_model,
                base_scale=base_scale,
                use_nGPT=use_nGPT,
                dropout=dropout,
                bias=use_bias,
                att_context_size=self.att_context_size,
            )
        )

        self.setup_streaming_params()

    def get_initial_cache_state(self, batch_size=1, dtype=torch.float32, device=None, max_dim=0):
        print()
        print("************************************************************************************************************")
        print(f"get_initial_cache_state ")

        print(f"get_initial_cache_state ")
        print(f"get_initial_cache_state ")
        print(f"get_initial_cache_state ")
        print(f"get_initial_cache_state ")
        print(f"max_dim {max_dim}")
        if device is None:
            device = next(self.parameters()).device
        if max_dim > 0:
            create_tensor = torch.randn
        else:
            create_tensor = torch.zeros
        print(f"max_dim")

        
        cache_last_channel = create_tensor(
            (
                len(self.layers),
                batch_size,
                self.streaming_cfg.last_channel_cache_size,
                self.d_model,
            ),
            device=device,
            dtype=dtype,
        )
        print(f"cache_last_channel  {cache_last_channel.shape}")
    
        if max_dim > 0:
            cache_last_channel_len = torch.randint(
                0,
                min(max_dim, self.streaming_cfg.last_channel_cache_size),
                (batch_size,),
                device=device,
                dtype=torch.int64,
            )
            for i in range(batch_size):
                cache_last_channel[:, i, cache_last_channel_len[i] :, :] = 0    
        else:
            cache_last_channel_len = torch.zeros(batch_size, device=device, dtype=torch.int64)

        print(f"cache_last_channel_len  {cache_last_channel_len}")
        print()
        print("************************************************************************************************************")
        return cache_last_channel,None, cache_last_channel_len


    def setup_streaming_params(
        self,
        chunk_size: int = None,
        shift_size: int = None,
        left_chunks: int = None,
        att_context_size: list = None,
        max_context: int = 10000,
    ):
        """
        This function sets the needed values and parameters to perform streaming. The configuration would be stored in self.streaming_cfg.
        The streaming configuration is needed to simulate streaming inference.

        Args:
            chunk_size (int): overrides the chunk size
            shift_size (int): overrides the shift size for chunks
            left_chunks (int): overrides the number of left chunks visible to each chunk
            max_context (int): the value used for the cache size of last_channel layers if left context is set to infinity (-1)
                Defaults to -1 (means feat_out is d_model)
        """
        streaming_cfg = CacheAwareStreamingConfig()
        print(f"chunk_size {chunk_size}")
        print(f" shift_size {shift_size}")
        # When att_context_size is not specified, it uses the default_att_context_size
        if att_context_size is None:
            att_context_size = self.att_context_size

        if chunk_size is not None:
            if chunk_size < 1:
                raise ValueError("chunk_size needs to be a number larger or equal to one.")
            lookahead_steps = chunk_size - 1
            streaming_cfg.cache_drop_size = chunk_size - shift_size
        elif self.att_context_style == "chunked_limited":
            lookahead_steps = att_context_size[1]
            streaming_cfg.cache_drop_size = 0
        elif self.att_context_style == "regular":
            lookahead_steps = att_context_size[1] * self.n_layers #+ self.conv_context_size[1] * self.n_layers
            streaming_cfg.cache_drop_size = lookahead_steps
        else:
            streaming_cfg.cache_drop_size = 0
            lookahead_steps = None
        print(f"lookahead_steps {lookahead_steps}")
        if chunk_size is None:
            streaming_cfg.last_channel_cache_size = att_context_size[0] if att_context_size[0] >= 0 else max_context
            print(f"max_context  {max_context}")
            print(f"streaming_cfg.last_channel_cache_size = {streaming_cfg.last_channel_cache_size} ")
        else:
            if left_chunks is None:
                raise ValueError("left_chunks can not be None when chunk_size is set.")
            streaming_cfg.last_channel_cache_size = left_chunks * chunk_size
            print(f"streaming_cfg.last_channel_cache_size  {streaming_cfg.last_channel_cache_size} ")

        
        if hasattr(self.pre_encode, "get_sampling_frames"):
            sampling_frames = self.pre_encode.get_sampling_frames()
        else:
            sampling_frames = 0
        print(f"sampling_frames {sampling_frames}")
        print(f"self.subsampling_factor {self.subsampling_factor}")

        if isinstance(sampling_frames, list):
            streaming_cfg.chunk_size = [
                sampling_frames[0] + self.subsampling_factor * lookahead_steps,
                sampling_frames[1] + self.subsampling_factor * lookahead_steps,
            ]
        else:
            streaming_cfg.chunk_size = sampling_frames * (1 + lookahead_steps)
        print(f"streaming_cfg.chunk_size  {streaming_cfg.chunk_size}")
        if isinstance(sampling_frames, list):
            streaming_cfg.shift_size = [
                sampling_frames[0] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
                sampling_frames[1] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
            ]
        else:
            streaming_cfg.shift_size = sampling_frames * (1 + lookahead_steps - streaming_cfg.cache_drop_size)
        print(f"streaming_cfg.shift_size  {streaming_cfg.shift_size}")

        if isinstance(streaming_cfg.shift_size, list):
            streaming_cfg.valid_out_len = (
                streaming_cfg.shift_size[1] - sampling_frames[1]
            ) // self.subsampling_factor + 1
        else:
            streaming_cfg.valid_out_len = streaming_cfg.shift_size // self.subsampling_factor

        print(f"streaming_cfg.valid_out_len {streaming_cfg.valid_out_len}")
        if hasattr(self.pre_encode, "get_streaming_cache_size"):
            streaming_cfg.pre_encode_cache_size = self.pre_encode.get_streaming_cache_size()
        else:
            streaming_cfg.pre_encode_cache_size = 0
        print(f"streaming_cfg.pre_encode_cache_size {streaming_cfg.pre_encode_cache_size}")
        if isinstance(streaming_cfg.pre_encode_cache_size, list):
            if streaming_cfg.pre_encode_cache_size[1] >= 1:
                streaming_cfg.drop_extra_pre_encoded = (
                    1 + (streaming_cfg.pre_encode_cache_size[1] - 1) // self.subsampling_factor
                )
            else:
                streaming_cfg.drop_extra_pre_encoded = 0
        else:
            streaming_cfg.drop_extra_pre_encoded = streaming_cfg.pre_encode_cache_size // self.subsampling_factor

        print(f"streaming_cfg.drop_extra_pre_encoded {streaming_cfg.drop_extra_pre_encoded}") 
        # for m in self.layers.modules():
        #     if hasattr(m, "_max_cache_len"):
        #         if isinstance(m, GPT):
        #             m.cache_drop_size = streaming_cfg.cache_drop_size
        #             print(f" m.cache_drop_size  { m.cache_drop_size }")
        self.streaming_cfg = streaming_cfg   
    


    def _calc_context_sizes(
        self, att_context_size, att_context_probs, att_context_style,
    ):
        # convert att_context_size to a standard list of lists
        if att_context_size:
            att_context_size_all = list(att_context_size)
            if isinstance(att_context_size_all[0], int):
                att_context_size_all = [att_context_size_all]
            for i, att_cs in enumerate(att_context_size_all):
                if isinstance(att_cs, ListConfig):
                    att_context_size_all[i] = list(att_cs)
                if att_context_style == "chunked_limited":
                    if att_cs[0] > 0 and att_cs[0] % (att_cs[1] + 1) > 0:
                        raise ValueError(f"att_context_size[{i}][0] % (att_context_size[{i}][1] + 1) should be zero!")
                    if att_cs[1] < 0 and len(att_context_size_all) <= 1:
                        raise ValueError(
                            f"Right context (att_context_size[{i}][1]) can not be unlimited for chunked_limited style!"
                        )
        else:
            att_context_size_all = [[-1, -1]]

        if att_context_probs:
            if len(att_context_probs) != len(att_context_size_all):
                raise ValueError("The size of the att_context_probs should be the same as att_context_size.")
            att_context_probs = list(att_context_probs)
            if sum(att_context_probs) != 1:
                raise ValueError(
                    "The sum of numbers in att_context_probs should be equal to one to be a distribution."
                )
        else:
            att_context_probs = [1.0 / len(att_context_size_all)] * len(att_context_size_all)

        print(f"att_context_size_all {att_context_size_all}")
        print(f"att_context_size_all[0] {att_context_size_all[0]}")
        print(f"att_context_probs {att_context_probs}")

        return att_context_size_all, att_context_size_all[0], att_context_probs, None
    
    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        if self.self_attention_model != "rel_pos_local_attn":
            att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)

            if self.att_context_style == "regular":
                if att_context_size[0] >= 0:
                    att_mask = att_mask.triu(diagonal=-att_context_size[0])
                if att_context_size[1] >= 0:
                    att_mask = att_mask.tril(diagonal=att_context_size[1])
                    
            elif self.att_context_style == "chunked_limited":
                # When right context is unlimited, just the left side of the masking need to get updated
                if att_context_size[1] == -1:
                    if att_context_size[0] >= 0:
                        att_mask = att_mask.triu(diagonal=-att_context_size[0])
                else:
                    chunk_size = att_context_size[1] + 1
                    # left_chunks_num specifies the number of chunks to be visible by each chunk on the left side
                    if att_context_size[0] >= 0:
                        left_chunks_num = att_context_size[0] // chunk_size
                    else:
                        left_chunks_num = 10000

                    chunk_idx = torch.arange(0, max_audio_length, dtype=torch.int, device=att_mask.device)
                    chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
                    diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
                    chunked_limited_mask = torch.logical_and(
                        torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
                    )
                    att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))
        else:
            att_mask = None

        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        if att_mask is not None:
            # pad_mask_for_att_mask is the mask which helps to ignore paddings
            pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
            pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
            # att_mask is the masking to be used by the MHA layers to ignore the tokens not supposed to be visible
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            # paddings should also get ignored, so pad_mask_for_att_mask is used to ignore their corresponding scores
            att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
            att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask
    

    def forward_for_export(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        
        raise NotImplementedError()

    def streaming_post_process(self, rets, keep_all_outputs=True):
        raise NotImplementedError()

    @typecheck()
    def forward(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        
        return self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

    def forward_internal(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        print(f"length befor {length}")
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )
        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size


        #print(f"audio_signal {audio_signal.size(1)}")
        audio_signal = audio_signal.transpose(1, 2)
        #print(f"audio_signal after transposes {audio_signal.size(1)}")
        x, length = self.pre_encode(x=audio_signal, lengths=length)
        length = length.to(torch.int64)
            # self.streaming_cfg is set by setup_streaming_cfg(), called in the init
        if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
            audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
            length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

        print(f"length after   {x.size(1)}")
        max_audio_length = x.size(1)

        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None
        
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )
        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_channel_next = []
        if cache_last_channel is not None:
            print(f"cache_last_channel is not none")
            print(f" cache_last_channel {cache_last_channel.shape}")
            print(f"  cache_last_channel[lth] {cache_last_channel.shape}")
            cache_last_channel_cur = cache_last_channel
            cache_last_time_cur = cache_last_time
        else:
            cache_last_channel_cur = None
            cache_last_time_cur = None

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []    
        x = self.ngpt(x, att_mask, pad_mask,cache_last_channel=cache_last_channel_cur, cache_keep_size=cache_keep_size,padding_length = padding_length )
        cache_last_channel_next,  x =  x
        x = x.transpose(1, 2)

        if cache_last_channel is not None:
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                None, # for cache time 
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return x, length

    def normalize_matrices(self):
        if hasattr(self.pre_encode, "normalize_matrices"):
            self.pre_encode.normalize_matrices()
        self.ngpt.normalize_matrices()


def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
    # Split the sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # Apply the rotary embeddings to the query and key
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1] // 2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1] // 2, 2)) * torch.stack((cos, sin), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape)
    k_rot = torch.reshape(k_rot, k.shape)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


def get_sinusoidal_embeddings(n_positions, dim, device):
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(n_positions, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.empty((n_positions, dim), device=device)
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb


def justnorm(x, fp32: bool = False, idim: int = -1):
    if fp32:
        dtype = x.dtype
        x = x.float()
        res = (x / x.norm(p=2, dim=idim, keepdim=True)).to(dtype=dtype)
    else:
        res = x / x.norm(p=2, dim=idim, keepdim=True)
    return res


def justnorm_fp32(x, idim: int = -1):
    return justnorm(x, idim=idim, fp32=True)


class Block(nn.Module):

    def __init__(self, config, iblock):
        super().__init__()
        self.config = config

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        if config.use_nGPT == 0:
            self.rmsnorm_att = RMSNorm(config.n_embd)
            self.rmsnorm_mlp = RMSNorm(config.n_embd)

        if config.use_nGPT == 1:
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(
                self.attn_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(
                self.mlp_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.sqk_init_value = 1.0
            self.sqk_init_scaling = config.base_scale
            self.sqk = torch.nn.Parameter(self.sqk_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32))

            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(
                self.suv_init_scaling * torch.ones(2 * 4 * config.n_embd, dtype=torch.float32)
            )

    def forward(self, h, mask, pad_mask, cache_last_channel_cur):

        B, T, C = h.size()
        key, value, query, cache = self.update_cache(key=h,value=h,query=h,cache=cache_last_channel_cur)

        # hin = h
        # if self.config.use_nGPT == 0:
        #     hin = self.rmsnorm_att(h)

        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        k = k.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        v = v.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)

        sinusoidal_pos = get_sinusoidal_embeddings(T, self.config.n_embd // self.config.n_head, device=q.device)
        q, k = apply_rotary_position_embeddings(sinusoidal_pos, q.transpose(1, 2), k.transpose(1, 2))
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)

        if self.config.use_nGPT == 1:
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
                1, 1, self.config.n_head, self.config.n_embd // self.config.n_head
            )
            q = sqk * justnorm(q)
            k = sqk * justnorm(k)

        sqrt_head_dim = (self.config.n_embd / self.config.n_head) ** 0.5
        if self.config.use_nGPT == 0:
            softmax_scale = 1.0 / sqrt_head_dim
        if self.config.use_nGPT == 1:
            softmax_scale = sqrt_head_dim


        y = flash_attn_func(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
        y = y.to(dtype=q.dtype)
        y = y.contiguous().view(B, T, self.config.n_embd)

        h_att = self.att_c_proj(y)

        if self.config.use_nGPT == 0:
            h = h + h_att
        if self.config.use_nGPT == 1:
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = justnorm(h)  # normally, normalization is not needed
            B_norm = justnorm(h_att)

            # res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = justnorm(res)

        hin = h
        if self.config.use_nGPT == 0:
            hin = self.rmsnorm_mlp(h)
        uv = self.c_fc(hin)
        if self.config.use_nGPT == 1:
            suv = self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd**0.5))
            uv = suv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)

        if self.config.use_nGPT == 0:
            h = h + h_mlp
        if self.config.use_nGPT == 1:
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = justnorm(h)  # normally, normalization is not needed
            B_norm = justnorm(h_mlp)

            # res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = justnorm(res)

        if cache is None:
            return h
        else:
            return h, cache

    def update_cache(self, key, value, query, cache):
        if cache is not None:
            key = value = torch.cat([cache, key], dim=1)
            q_keep_size = query.shape[1] - self.cache_drop_size
            cache = torch.cat([cache[:, q_keep_size:, :], query[:, :q_keep_size, :]], dim=1)
        return key, value, query, cache


@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0**0.5)  # 1 / sqrt(n_embd)
    use_nGPT: int = 0
    dropout: float = 0.0
    bias: bool = False
    att_context_size: int = 0


class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # wte=nn.Embedding(config.vocab_size, config.n_embd),
                # drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config, il) for il in range(config.n_layer)])
            )
        )
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # *we don't use it becuase in the nGPT paper there was no weight tying of weights*
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale / math.sqrt(2 * config.n_layer))
        # report number of parameters
        logging.info("[nGPT] number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        if config.use_nGPT == 0:
            self.rmsnorm_f = RMSNorm(config.n_embd)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

    def forward(self, x, att_mask=None, pad_mask=None, cache_last_channel=None):


        cache_last_time_next = []
        cache_last_channel_next = []
        for idx, block in enumerate(self.transformer.h):

            
            original_signal = x

            if cache_last_channel is not None:
                print(f"cache_last_channel is not none")
                print(f" cache_last_channel {cache_last_channel.shape}")                
                cache_last_channel_cur = cache_last_channel[idx]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None

            x = block(
                x, 
                mask=att_mask,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur
                )
            
            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, cache_last_time_cur) = x
                cache_last_channel_next.append(cache_last_channel_cur)
                #cache_last_time_next.append(cache_last_time_cur)
            
        if self.config.use_nGPT == 0:
            x = self.rmsnorm_f(x)
        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
  
            return (
                x,
                #length,
                cache_last_channel_next,
                cache_last_time_next,
              #  torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return x #, length
#return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logging.info(
            f"[nGPT] num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logging.info(
            f"[nGPT] num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        use_fused = False  # fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        logging.info(f"[nGPT] using fused AdamW: {use_fused}")
        return optimizer

    def normalize_matrices(self):
        if not self.config.use_nGPT:
            return

        transformer = self.transformer
        module = self

        for layer_idx in range(0, module.config.n_layer):
            block = transformer["h"][layer_idx]

            block.query.weight.data.copy_(justnorm_fp32(block.query.weight.data, 1))  # n_proj, n_embd
            block.key.weight.data.copy_(justnorm_fp32(block.key.weight.data, 1))  # n_proj, n_embd
            block.value.weight.data.copy_(justnorm_fp32(block.value.weight.data, 1))  # n_proj, n_embd
            block.att_c_proj.weight.data.copy_(justnorm_fp32(block.att_c_proj.weight.data, 0))  # n_embd, n_proj

            block.c_fc.weight.data.copy_(justnorm_fp32(block.c_fc.weight.data, 1))  # n_proj, n_embd
            block.mlp_c_proj.weight.data.copy_(justnorm_fp32(block.mlp_c_proj.weight.data, 0))  # n_embd, n_proj



class NGPTStackingSubsampling(torch.nn.Module):
    """Stacking subsampling which simply stacks consecutive frames to reduce the sampling rate
    Args:
        subsampling_factor (int): The subsampling factor
        feat_in (int): size of the input features
        feat_out (int): size of the output features
    """

    def __init__(
        self,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        use_bias: bool = False,
        base_scale: float = 1 / (1024**0.5),
    ):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = torch.nn.Linear(subsampling_factor * feat_in, feat_out, bias=use_bias)
        self.pad_frame = nn.Parameter(torch.ones(feat_in, dtype=torch.float32))

    def _init_weights(self):
        torch.nn.init.normal_(self.proj_out.weight, mean=0.0, std=self.config.base_scale)
        if self.proj_out.bias is not None:
            torch.nn.init.zeros_(self.proj_out.bias)

    def normalize_matrices(self):
        self.proj_out.weight.data.copy_(justnorm_fp32(self.proj_out.weight.data, 0))
    def get_sampling_frames(self):
        return self.subsampling_factor

    def get_streaming_cache_size(self):
        return 0

    def forward(self, x, lengths):
        b, t, h = x.size()
        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        lengths = torch.div(lengths + pad_size, self.subsampling_factor, rounding_mode='floor')

        # Pad and fill padding frames (all-zero) with a learnable padding 'embedding'
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        x[(x == 0).all(dim=-1)] = self.pad_frame

        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        x = justnorm(x)
        x = self.proj_out(x)
        x = justnorm(x)

        return x, lengths
