import torch
import torch.nn as nn

class DownSampling(nn.Module):
    
    def __init__(self, in_channels, out_channels, max_pool):
        """
        DownSampling block in the U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            max_pool (bool): Whether to use max pooling.
        """
        super(DownSampling, self).__init__()
        self.max_pool = max_pool
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.relu(self.batchnorm2d(x))
        skip_connection = x
        
        if self.max_pool:
            next_layer = self.maxpool2d(x)
        else:
            return x
        return next_layer, skip_connection

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        UpSampling block in the U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(UpSampling, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x, prev_skip):
        x = self.up(x)
        x = torch.cat((x, prev_skip), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        next_layer = self.relu(self.batchnorm(x))
        return next_layer

class UNet(nn.Module):
    
    """
        U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            features (list): List of feature sizes for downsampling and upsampling.
    """
    def __init__(self, in_channels, out_channels, features):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        for feature in features:
            self.downs.append(DownSampling(in_channels, feature, True))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(UpSampling(2 * feature, feature))

        self.bottleneck = DownSampling(features[-1], 2 * features[-1], False)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x, skip_connection = down(x)
            skip_connections.append(skip_connection)
        skip_connections = skip_connections[::-1]
        x = self.bottleneck(x)
        for i, up in enumerate(self.ups):
            x = up(x, skip_connections[i])

        return self.final_conv(x)
    
if __name__ == "__main__":
    #Example Usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = [64, 128, 256, 512]
    model = UNet(1, 1, features=features).to(device)
    print(model(torch.rand(1, 1, 512, 512)).shape)