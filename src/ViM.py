#!/usr/bin/env python3

import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange, Reduce
from src.VisionMambaEncoder import VisionMambaEncoderBlock

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
    
    def forward(self, x):
        return x + self.pos_embed


class ViM(nn.Module):
    def __init__(
        self,
        input_dim,
        state_dim,
        d_conv,
        num_blocks,
        image_size,
        patch_size,
        num_classes,
        channels,
        dropout,
    ):
        super().__init__()
        image_h, image_w = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_h, patch_w = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        # Patchify
        self.patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_h,
                p2=patch_h,
            ),
            nn.Linear(channels * patch_h * patch_w, input_dim),
        )

        # Positional Encoding
        self.pos = PositionalEncoding((image_h // image_w) ** 2, input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

        # Latent
        self.latent = nn.Identity()

        # Encoder Layers
        self.layers = nn.ModuleList()
        
        for _ in range(num_blocks):
            self.layers.append(
                VisionMambaEncoderBlock(
                    input_dim=input_dim,
                    state_dim=state_dim,
                    d_conv=d_conv
                )
            )

        # Output Head
        self.output_head = nn.Sequential(
            Reduce("b m e -> b e", "mean"),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes)
        )
    
    def forward(self, x):
        # Patch Dimensions
        b, c, h, w = x.shape

        # Patch Embeddings
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        # Positional Encoding
        x = self.pos(x)

        # Dropout
        x = self.dropout(x)

        # Encode
        for block in self.layers:
            x = block(x)

        # Project to a latent variable
        x = self.latent(x)

        return self.output_head(x)