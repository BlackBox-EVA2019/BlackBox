#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import math


# Loss functions

# Definition of loss functions for L1 and L2 norm

def masked_ABS(prediction, target, mask):
    batch_SE = torch.abs(prediction - target).mul_(mask)
    batch_MSE = torch.sum(batch_SE, [-1, -2]).div_(torch.sum(mask, [-1, -2]))
    return batch_MSE.mean()


def masked_MSE(prediction, target, mask):
    batch_SE = (prediction - target).pow_(2).mul_(mask)
    batch_MSE = torch.sum(batch_SE, [-1, -2]).div_(torch.sum(mask, [-1, -2]))
    return batch_MSE.mean()


# Model


# Deep convolutional encoder-decoder

def dconv_nonlin(in_ch, out_ch, kernel_size, stride=2, output_padding=0, nonlin=None):
    if nonlin is None:
        nonlin = nn.ReLU()
    return [nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               output_padding=output_padding
                               ),
            nonlin,
            nn.BatchNorm2d(out_ch)
            ]


def conv_nonlin(in_ch, out_ch, kernel_size, stride=2, nonlin=None):
    if nonlin is None:
        nonlin = nn.ReLU()
    return [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nonlin,
            nn.BatchNorm2d(out_ch)
            ]


# Neural network model class definition

class ConvEncDec(nn.Module):
    def __init__(self, height=384, width=96, window_days=1, out_channels=1, dim=32, enc_dropout=0.1,
                 input_layers=3, reducing_layers=5, output_layers=0, reduction_exponent=2, kernel_size=5,
                 encode_position=False):
        super().__init__()

        (self.height, self.width) = (height, width)
        self.reduction_factor = 2 ** reducing_layers
        self.window_days = window_days
        self.out_channels = out_channels
        self.dim = dim
        self.reduction_exponent = reduction_exponent
        self.reducing_layers = reducing_layers
        self.kernel_size = kernel_size
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.encode_position = encode_position
        if self.encode_position:
            self.register_buffer("h_pos_encoding", torch.linspace(-1, 1,
                                                                  height).reshape(1, 1, height, 1).expand(1, 1, height, width))
            self.register_buffer("w_pos_encoding", torch.linspace(-1, 1,
                                                                  width).reshape(1, 1, 1, width).expand(1, 1, height, width))
            # data + mask + horiz pos enc + vert pos enc
            self.in_channels = window_days * 2 + 2
        else:
            self.in_channels = window_days * 2  # data + mask

        assert height % self.reduction_factor == 0
        assert width % self.reduction_factor == 0
        assert input_layers >= 1
        #assert channels == 1

        # Choice of type of nonlinearity in model
        nonlin = nn.SELU(inplace=True)

        Layers = conv_nonlin(self.in_channels, dim,
                             kernel_size, stride=1, nonlin=nonlin)

        for _ in range(input_layers):
            Layers += conv_nonlin(dim, dim, kernel_size,
                                  stride=1, nonlin=nonlin)

        for i in range(reducing_layers):
            Layers += conv_nonlin(dim * math.floor(reduction_exponent ** i),
                                  dim *
                                  math.floor(reduction_exponent ** (i+1)),
                                  kernel_size, stride=2, nonlin=nonlin)

        for _ in range(output_layers):
            Layers += conv_nonlin(dim * math.floor(reduction_exponent ** reducing_layers),
                                  dim *
                                  math.floor(reduction_exponent **
                                             reducing_layers),
                                  kernel_size, stride=1, nonlin=nonlin)

        Layers += [nn.Dropout(enc_dropout)]

        for _ in range(output_layers):
            Layers += dconv_nonlin(dim * math.floor(reduction_exponent ** reducing_layers),
                                   dim *
                                   math.floor(reduction_exponent **
                                              reducing_layers),
                                   kernel_size, stride=1, output_padding=0, nonlin=nonlin)

        for i in range(reducing_layers-1, -1, -1):
            Layers += dconv_nonlin(dim * math.floor(reduction_exponent ** (i+1)),
                                   dim * math.floor(reduction_exponent ** i),
                                   kernel_size, stride=2, output_padding=1, nonlin=nonlin)

        for _ in range(input_layers):
            Layers += dconv_nonlin(dim, dim, kernel_size,
                                   stride=1, output_padding=0, nonlin=nonlin)

        Layers += dconv_nonlin(dim, self.in_channels, kernel_size=kernel_size, stride=1,
                               output_padding=0, nonlin=nonlin
                               )

        Layers += [nn.ConvTranspose2d(self.in_channels, out_channels, kernel_size=kernel_size, stride=1,
                                      padding=kernel_size // 2, output_padding=0
                                      )
                   ]

        self.Layers = nn.Sequential(*Layers)

    def forward(self, x, xmask, xtime):
        x = self.prepare(x, xmask, xtime)
        x = self.Layers(x).squeeze()
        return x

    def prepare(self, x, xmask, xtime):
        if x.dim() == 3:
            # make it a batch of one
            x = x.unsqueeze(0)
            xmask = xmask.unsqueeze(0)
            xtime = xtime.unsqueeze(0)
        if self.encode_position:
            bs = x.shape[0]
            input_list = (x,
                          xmask,
                          self.h_pos_encoding.expand(bs, -1, -1, -1),
                          self.w_pos_encoding.expand(bs, -1, -1, -1)
                          )
        else:
            input_list = (x, xmask)
        return torch.cat(input_list, dim=1)

    def get_latent_vector_size(self):
        return [self.height // self.reduction_factor, self.width // self.reduction_factor,
                self.dim * math.floor(self.reduction_exponent ** self.reducing_layers)]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    pass
