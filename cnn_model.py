from abc import abstractmethod, ABC
from typing import List

import torch.nn as nn


class USStem(nn.Module, ABC):
    out_channels: int

    @abstractmethod
    def __init__(self, inp_channels):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, x):
        pass


class USProgressiveStem(USStem):
    out_channels = 64

    def __init__(self, inp_channels: int = 1):
        super().__init__(inp_channels)
        self.net = nn.Sequential(nn.Conv2d(inp_channels, 32, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),

                                 nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU(),

                                 nn.Conv2d(32, USProgressiveStem.out_channels, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(USProgressiveStem.out_channels),
                                 nn.ReLU()
                                 )

    def forward(self, x):
        return self.net(x)


class USBody(nn.Module, ABC):
    @abstractmethod
    def __init__(self, channels: List[int]):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, x):
        pass


class USBaseBody(USBody):
    def __init__(self, inp_channels = 64, channels=None, dropout=0.2):
        super().__init__(channels)
        if channels is None:
            channels = [128, 256, 512]
        self.channels = channels
        self.inp_channels = inp_channels
        self.dropout = dropout

        self.net = nn.Sequential(
            *self.init_body_layers()
        )

    def init_body_layers(self) -> List[nn.Module]:
        prev_out_n = self.inp_channels
        body_layers = []
        for channel_n in self.channels:
            body_layers.extend(self.init_body_layer(prev_out_n, channel_n))
            prev_out_n = channel_n

        return body_layers

    def init_body_layer(self, inp_ch, out_ch) -> List[nn.Module]:
        layer = [nn.Conv2d(inp_ch, out_ch, kernel_size=3, padding=1),
                 nn.BatchNorm2d(out_ch),
                 nn.ReLU(),
                 nn.MaxPool2d(kernel_size=2),
                 nn.Dropout2d(self.dropout)
                 ]
        return layer

    def forward(self, x):
        return self.net(x)


class USVGGBody(USBody):
    def __init__(self, channels=None, inp_channels = 64, dropout=0.2):
        super().__init__(channels)
        if channels is None:
            channels = [128, 256, 512]
        self.dropout = dropout
        self.channels = channels
        self.inp_channels = inp_channels
        self.net = nn.Sequential(
            *self.init_body_layers()
        )

    def init_body_layers(self) -> List[nn.Module]:
        prev_out_n = self.inp_channels
        body_layers = []
        for channel_n in self.channels:
            body_layers.extend(self.init_body_layer(prev_out_n, channel_n))
            prev_out_n = channel_n

        return body_layers

    def init_body_layer(self, inp_ch, out_ch) -> List[nn.Module]:
        layer = [
            nn.Conv2d(inp_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout),
        ]
        return layer

    def forward(self, x):
        return self.net(x)


class USHead(nn.Module, ABC):
    inp_channels: int

    @abstractmethod
    def __init__(self, num_classes: int):
        super().__init__()
        pass

    @abstractmethod
    def forward(self, x):
        pass


class USGAPHead(USHead):
    def __init__(self, inp_channels: int = 512, num_classes: int = 10):
        super().__init__(num_classes)
        self.num_classes = num_classes
        self.inp_channels = inp_channels
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(),
                                 nn.Linear(self.inp_channels, self.num_classes)
                                 )

    def forward(self, x):
        return self.net(x)


class USFCHead(USHead):
    def __init__(self, inp_channels: int = 512, fc_dim: int = 128, dropout=0.2, num_classes: int = 10):
        super().__init__(num_classes)
        self.num_classes = num_classes
        self.inp_channels = inp_channels
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(),
                                 nn.Linear(self.inp_channels, self.fc_dim),
                                 nn.ReLU(),
                                 nn.Dropout(self.dropout),
                                 nn.Linear(self.fc_dim, self.num_classes))

    def forward(self, x):
        return self.net(x)


class UrbanSoundCNN(nn.Module):
    def __init__(self, stem: USStem, body: USBody, head: USHead):
        super().__init__()
        self.stem = stem
        self.body = body
        self.head = head

    def forward(self, x):
        y = self.stem(x)
        y = self.body(y)
        y = self.head(y)
        return y
