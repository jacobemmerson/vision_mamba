#!/usr/bin/env python3

import torch
import torch.nn as nn
from mamba_ssm import Mamba

class VisionMambaEncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        state_dim,
        d_conv,
    ):
        super().__init__()

        ## Store Shapes
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.d_conv = d_conv

        ## 1D Convolutions
        self.forwardConv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=1
        )

        # Forward Process Weights
        #self.forwardLinearB = nn.Linear(input_dim, state_dim)
        #self.forwardLinearC = nn.Linear(input_dim, state_dim)
        #self.forwardLinearD = nn.Linear(input_dim, input_dim)
        #self.forwardParam = nn.Parameter(torch.randn(input_dim, state_dim))

        # Backward Process Weights
        self.backwardConv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=1
        )

        #self.backwardLinearB = nn.Linear(input_dim, state_dim)
        #self.backwardLinearC = nn.Linear(input_dim, state_dim)
        #self.backwardLinearD = nn.Linear(input_dim, input_dim)
        #self.backwardParam = nn.Parameter(torch.randn(input_dim, state_dim))

        ## SSM Modules
        self.forwardSSM = Mamba(
            d_model=input_dim,
            d_state=state_dim,
            d_conv=d_conv
        )

        self.backwardSSM = Mamba(
            d_model=input_dim,
            d_state=state_dim,
            d_conv=d_conv
        )

        ## Last Linear Layer
        self.linear = nn.Linear(input_dim, input_dim)

        ## Norm and Activation
        self.norm = nn.LayerNorm(input_dim)
        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        b,m,d = x.shape

        ## Skip Connection
        skip = x

        ## Layer Norm
        x = self.norm(x)

        ## Split Input
        x = self.linear(x)
        z = self.linear(x)

        ## Activation
        z = self.silu(z)

        ## Convolutions
        yf = self.process(
            x,
            self.forwardConv1d,
            self.forwardSSM
        )

        yb = self.process(
            x,
            self.backwardConv1d,
            self.backwardSSM
        )

        out = (yf * z) + (yb * z) + skip
        return out
    
    def process(
        self,
        x,
        conv1d,
        ssm
    ): 
        x = x.permute(0, 2, 1) # (B, E, M)
        x = self.silu(conv1d(x)) 
        x = x.permute(0, 2, 1) # (B, M, E)
        return ssm(x)