# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import torch.nn as nn
import torch
from torchinfo import summary
from block import DBLBlock, ResidualBlock, ScalePrediction, ConvBlock

class Yolo_Model(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()

        # Build back-bone dark-net 53
        self.Block1 = nn.Sequential(DBLBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
                                    DBLBlock(32, 64, kernel_size=3, stride=2, padding=1),
                                    ResidualBlock(64),
                                    DBLBlock(64, 128, kernel_size=3, stride=2, padding=1),
                                    ResidualBlock(128, True, 2),
                                    DBLBlock(128, 256, kernel_size=3, stride=2, padding=1),
                                    ResidualBlock(256, True, 8))

        self.Block2 = nn.Sequential(DBLBlock(256, 512, kernel_size=3, stride=2, padding=1),
                                    ResidualBlock(512, True, 8))

        self.Block3 = nn.Sequential(DBLBlock(512, 1024, kernel_size=3, stride=2, padding=1),
                                    ResidualBlock(1024, True, 4),
                                    ConvBlock(1024, 1024, 3))

        # For big objects (13x13)
        self.ScalePredictBig = nn.Sequential(DBLBlock(1024, 512, kernel_size=1),
                                             ScalePrediction(512, num_classes))

        # For medium objects (26x26)

        self.Block4 = nn.Sequential(DBLBlock(1024, 256, kernel_size=1),
                                    nn.Upsample(scale_factor=2.0))

        self.Block5 = nn.Sequential(ConvBlock(768, 512, 3))

        self.ScalePredictMedium = nn.Sequential(DBLBlock(512, 256, kernel_size=1),
                                                ScalePrediction(256, num_classes))

        # For small objects (52x52)
        self.Block6 = nn.Sequential(DBLBlock(512, 128, kernel_size=1),
                                    nn.Upsample(scale_factor=2.0))

        self.ScalePredictSmall = nn.Sequential(ConvBlock(384, 256, 3),
                                               DBLBlock(256, 128, kernel_size=1),
                                               ScalePrediction(128, num_classes))

    def forward(self, x):
        # List store 3 head
        outputs = []

        # Back-bone
        route1 = self.Block1(x)
        route2 = self.Block2(route1)
        route3 = self.Block3(route2)

        # For detect big objects
        scale_big = self.ScalePredictBig(route3)
        outputs.append(scale_big)

        # For detect medium objects
        route3 = self.Block4(route3)
        route2 = torch.cat([route2, route3], dim=1)
        route2 = self.Block5(route2)
        scale_medium = self.ScalePredictMedium(route2)
        outputs.append(scale_medium)

        # For detect small objects
        route2 = self.Block6(route2)
        route1 = torch.cat([route1, route2], dim=1)
        scale_small = self.ScalePredictSmall(route1)
        outputs.append(scale_small)

        return outputs

def test():
    model = Yolo_Model()
    x = torch.randn((1, 3, 416, 416))
    y = model(x)
    print(y[2].shape)
    summary(model, (1, 3, 416, 416))

if __name__ == '__main__':
    test()
