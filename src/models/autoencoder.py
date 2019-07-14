import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 2, stride=(2,2))
        self.conv3 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 64, 2, stride=(2,2))
        self.conv5 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv6 = nn.Conv2d(64, out_channels, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.conv_transpose2d(x, self.conv6.weight, padding=2)
        x = F.conv_transpose2d(x, self.conv5.weight, padding=2)
        x = F.conv_transpose2d(x, self.conv4.weight, stride=(2,2))
        x = F.conv_transpose2d(x, self.conv3.weight, padding=2)
        x = F.conv_transpose2d(x, self.conv2.weight, stride=(2,2))
        x = F.conv_transpose2d(x, self.conv1.weight, padding=2)
        return x
