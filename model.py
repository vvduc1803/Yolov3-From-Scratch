import torch.nn as nn
import torch
from torchinfo import summary
from block import DBLBlock, ResidualBlock, ScalePrediction

class Yolo_Model(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()

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
                                    DBLBlock(1024, 512, kernel_size=1),
                                    DBLBlock(512, 1024, kernel_size=3, padding=1))

        self.Block7_ = nn.Sequential(DBLBlock(1024, 256, kernel_size=1),
                                     nn.Upsample(scale_factor=2.0))

        self.Block8_ = nn.Sequential(DBLBlock(768, 256, kernel_size=1),
                                     DBLBlock(256, 512, kernel_size=3, padding=1))

        self.Block9_ = nn.Sequential(DBLBlock(512, 128, kernel_size=1),
                                     nn.Upsample(scale_factor=2.0))

        self.Block10_ = nn.Sequential(DBLBlock(384, 128, kernel_size=1),
                                      DBLBlock(128, 256, kernel_size=3, padding=1))

        self.ScalePredictBig = nn.Sequential(ResidualBlock(1024, use_residual=False),
                                             DBLBlock(1024, 512, kernel_size=1),
                                             ScalePrediction(512, num_classes)
                                             )

        self.ScalePredictMedium = nn.Sequential(ResidualBlock(512, use_residual=False),
                                                DBLBlock(512, 256, kernel_size=1),
                                                ScalePrediction(256, num_classes)
                                                )

        self.ScalePredictSmall = nn.Sequential(ResidualBlock(256, use_residual=False),
                                               DBLBlock(256, 128, kernel_size=1),
                                               ScalePrediction(128, num_classes)
                                               )


    def forward(self, x):
        outputs = []
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        route3 = self.Block4_(x)
        x = self.Block4_(x)
        route2 = self.Block5_(x)
        x = self.Block5_(x)
        route1 = self.Block6_(x)
        x = self.Block6_(x)
        detec_big = self.ScalePredictBig(x)
        outputs.append(detec_big)

        route1 = self.Block7_(route1)
        route1 = torch.cat([route1, route2], dim=1)
        x = self.Block8_(route1)
        route1 = self.Block8_(route1)
        detec_medium = self.ScalePredictMedium(route1)
        outputs.append(detec_medium)

        x = self.Block9_(x)
        x = torch.cat([x, route3], dim=1)
        x = self.Block10_(x)
        detec_small = self.ScalePredictSmall(x)
        outputs.append(detec_small)

        return outputs

def test():
    model = Yolo_Model()
    summary(model, (1, 3, 416, 416))

if __name__ == '__main__':
    test()
