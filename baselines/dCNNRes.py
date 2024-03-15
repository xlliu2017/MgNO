import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import *

class DilResNet(nn.Module):
    def __init__(self, in_channels=1, hid_channels=48, out_channels=1, stride=1, normalizer=None):
        super(DilResNet, self).__init__()
        
        # Encoder CNN
        self.encoder_conv = nn.Conv2d(in_channels=in_channels, out_channels=hid_channels, kernel_size=3, padding=1, stride=stride)
        
        # Processor
        self.processor = nn.Sequential(
            dCNN(hid_channels),
            dCNN(hid_channels),
            dCNN(hid_channels),
            dCNN(hid_channels)
        )
        
        # Decoder CNN
        self.decoder_conv = nn.Conv2d(in_channels=hid_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.normalizer = normalizer

    def forward(self, x):
        # Encode
        x = self.encoder_conv(x)
        
        # Process with residual connections
        for i in range(4):
            identity = x
            x = self.processor[i](x)
            x += identity
            x = F.relu(x)
        
        # Decode
        x = self.decoder_conv(x)
        
        return self.normalizer.decode(x) if self.normalizer is not None else x


class dCNN(nn.Module):
    def __init__(self, num_channels):
        super(dCNN, self).__init__()
        
        # Dilation rates
        dilation_rates = [1, 2, 4, 8, 4, 2, 1]
        
        # 7 dilated CNN layers
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, dilation=dilation_rates[0], padding=dilation_rates[0])
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, dilation=dilation_rates[1], padding=dilation_rates[1])
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, dilation=dilation_rates[2], padding=dilation_rates[2])
        self.conv4 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, dilation=dilation_rates[3], padding=dilation_rates[3])
        self.conv5 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, dilation=dilation_rates[4], padding=dilation_rates[4])
        self.conv6 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, dilation=dilation_rates[5], padding=dilation_rates[5])
        self.conv7 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, dilation=dilation_rates[6], padding=dilation_rates[6])
        

    def forward(self, x):

        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))

        x = F.relu(self.conv6(x))

        x = self.conv7(x)
        return x


if __name__ == '__main__':
    model = DilResNet(hid_channels=64)
    print(count_params(model))
    x = torch.randn(10, 1, 256, 256)
    out = model(x)
    print(out.shape)