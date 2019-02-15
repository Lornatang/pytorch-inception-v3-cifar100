import torch
from torch import nn


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch_0 = nn.Conv2d(in_channels, 96, kernel_size=1, stride=1, padding=0)

        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        return torch.cat([branch_0, branch_1, branch_2, branch_3], axis=3)


class ReductionA(nn.Module):

    def __init__(self, in_channels):
        super(ReductionA, self).__init__()
        self.branch_0 = nn.Conv2d(in_channels, 384, kernel_size=3, stride=1, padding=1)

        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(192, 224, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(224, 256, kernel_size=3, stride=1, padding=1)
        )
        self.branch_2 = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        return torch.cat([branch_0, branch_1, branch_2], axis=3)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch_0 = nn.Conv2d(in_channels, 384, kernel_size=1, stride=1, padding=0)

        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(192, 224, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(224, 256, kernel_size=1, stride=1, padding=0)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(192, 192, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(192, 224, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(224, 224, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(224, 256, kernel_size=1, stride=1, padding=0)
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        return torch.cat([branch_0, branch_1, branch_2, branch_3], axis=3)


class ReductionB(nn.Module):

    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(256, 320, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
        )

        self.branch_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        return torch.cat([branch_0, branch_1, branch_2], axis=3)


class InceptionC(nn.Module):

    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
        self.branch_0 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=1, stride=1, padding=0),
            torch.cat([
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
            ], axis=3)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, 384, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(384, 448, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(448, 512, kernel_size=1, stride=1, padding=0),
            torch.cat([
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
            ], axis=3)
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        return torch.cat([branch_0, branch_1, branch_2, branch_3], axis=3)
