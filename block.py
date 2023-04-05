# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, hidden, num_dbl=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers += [nn.Sequential(nn.Conv2d(in_channels, hidden // 2, kernel_size=1),
                                      nn.BatchNorm2d(hidden // 2),
                                      nn.SiLU()),
                        nn.Sequential(nn.Conv2d(hidden // 2, hidden, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(hidden),
                                      nn.SiLU())]

        for dbl in range(num_dbl - 1):
            self.layers += [nn.Sequential(nn.Conv2d(hidden, hidden // 2, kernel_size=1),
                                          nn.BatchNorm2d(hidden // 2),
                                          nn.SiLU()),
                            nn.Sequential(nn.Conv2d(hidden // 2, hidden, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(hidden),
                                          nn.SiLU())]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DBLBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.dbl = nn.Sequential(nn.Conv2d(in_channels, out_channels, **kwargs),
                                 nn.BatchNorm2d(out_channels),
                                 nn.SiLU())

    def forward(self, x):
        x = self.dbl(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    DBLBlock(channels, channels // 2, kernel_size=1),
                    DBLBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            DBLBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            DBLBlock(
                2 * in_channels, (num_classes + 5) * 3, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )