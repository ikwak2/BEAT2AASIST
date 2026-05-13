# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------


import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as ta_kaldi
from torch.nn import LayerNorm
import torchaudio.transforms as T

from .backbone import TransformerEncoder

logger = logging.getLogger(__name__)


class BEATsConfig:
    def __init__(self, cfg=None):
        self.input_patch_size: int = 16  # path size of patch embedding
        self.embed_dim: int = 512  # patch embedding dimension
        self.conv_bias: bool = False  # include bias in conv encoder

        self.encoder_layers: int = 12  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072  # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use

        self.layer_wise_gradient_decay_ratio: float = (
            1.0  # ratio for layer-wise gradient decay
        )
        self.layer_norm_first: bool = False  # apply layernorm first in the transformer
        self.deep_norm: bool = False  # apply deep_norm first in the transformer

        # dropouts
        self.dropout: float = 0.1  # dropout probability for the transformer
        self.attention_dropout: float = 0.1  # dropout probability for attention weights
        self.activation_dropout: float = (
            0.0  # dropout probability after activation in FFN
        )
        self.encoder_layerdrop: float = (
            0.0  # probability of dropping a tarnsformer layer
        )
        self.dropout_input: float = (
            0.0  # dropout to apply to the input (after feat extr)
        )

        # positional embeddings
        self.conv_pos: int = (
            128  # number of filters for convolutional positional embeddings
        )
        self.conv_pos_groups: int = (
            16  # number of groups for convolutional positional embedding
        )

        # relative position embedding
        self.relative_position_embedding: bool = (
            False  # apply relative position embedding
        )
        self.num_buckets: int = 320  # number of buckets for relative position embedding
        self.max_distance: int = (
            1280  # maximum distance for relative position embedding
        )
        self.gru_rel_pos: bool = False  # apply gated relative position embedding

        # label predictor
        self.finetuned_model: bool = False  # whether the model is a fine-tuned model.
        self.predictor_dropout: float = 0.1  # dropout probability for the predictor
        self.predictor_class: int = 527  # target class number for the predictor

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class BEATs(nn.Module):
    def __init__(
        self,
        args,
        cfg: BEATsConfig,
    ) -> None:
        super().__init__()
        logger.info(f"BEATs Config: {cfg.__dict__}")

        self.cfg = cfg
        self.args = args
        self.train_mode = False

        self.embed = cfg.embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(
            1,
            self.embed,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=cfg.conv_bias,
        )

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        if cfg.finetuned_model:
            self.predictor_dropout = nn.Dropout(cfg.predictor_dropout)
            self.predictor = nn.Linear(cfg.encoder_embed_dim, cfg.predictor_class)
        else:
            self.predictor = None

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        tgt_layer: Optional[int] = None,
    ):
        fbank = source  # (B, 402, 128)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)
        
        if self.train_mode:
            ### Masking Augmentation
            if self.args.time_mask=='True' and np.random.rand()<0.5:
                time_masking = T.TimeMasking(time_mask_param= self.args.max_time_mask, iid_masks=self.args.mask_iid, p=self.args.max_mask_ratio)
                fbank = time_masking(fbank)

            if self.args.freq_mask=='True' and np.random.rand()<0.5:   
                freq_masking = T.FrequencyMasking(freq_mask_param= self.args.max_freq_mask, iid_masks=self.args.mask_iid)
                fbank = freq_masking(fbank)

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)  # (B, emb_dim, W, H)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)  # (B, H*W, emb_dim)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=tgt_layer,
        )

        return x, padding_mask, layer_results


class BEATsModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # load the pre-trained checkpoint
        if args.pre_trained_path:
            checkpoint = torch.load(args.pre_trained_path)
            cfg = BEATsConfig(checkpoint["cfg"])
            BEATs_model = BEATs(args, cfg)
            BEATs_model.load_state_dict(checkpoint["model"])
            print('PRE-TRAINED BEATs')
        else:
            cfg = BEATsConfig()
            BEATs_model = BEATs(args, cfg)
            print('Train BEATs FROM-SCRATCH ')

        self.model = BEATs_model
        self.args = args

    def forward(self, x, return_hidden: bool = False, last_k: Optional[int] = None):
        
        if next(self.model.parameters()).device != x.device \
           or next(self.model.parameters()).dtype != x.dtype:
            self.model.to(x.device, dtype=x.dtype)
        
        # when returning hidden states, set tgt_layer to the last encoder layer to collect all
        tgt_layer = self.model.cfg.encoder_layers - 1 if return_hidden else None
        out = self.model.extract_features(x, tgt_layer=tgt_layer)

        # out is (features, padding_mask, layer_results)
        features = out[0]
        layer_results = out[2] if len(out) > 2 else []

        if not return_hidden:
            return features

        # layer_results is a list of tuples (x_tbC, z) captured before-first and after each layer when tgt_layer is not None
        # Convert to B x T x C and keep outputs AFTER each transformer layer
        hidden_states = []
        if layer_results:
            # skip the very first tuple (input to the first layer)
            for x_tbc, _ in layer_results[1:]:
                hidden_states.append(x_tbc.transpose(0, 1))  # T x B x C -> B x T x C

        if isinstance(last_k, int) and last_k > 0:
            hidden_states = hidden_states[-last_k:]

        return features, hidden_states 



