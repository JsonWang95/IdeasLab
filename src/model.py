import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, encoder_name='resnet18', in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, features_only=True, out_indices=(0, 1, 2, 3, 4), pretrained=True)
        encoder_channels = self.encoder.feature_info.channels()

        self.conv0 = DoubleConv(in_channels, 64)
        self.conv1 = DoubleConv(encoder_channels[0], encoder_channels[0])
        self.conv2 = DoubleConv(encoder_channels[1], encoder_channels[1])
        self.conv3 = DoubleConv(encoder_channels[2], encoder_channels[2])
        self.conv4 = DoubleConv(encoder_channels[3], encoder_channels[3])

        self.up_conv4 = DoubleConv(encoder_channels[4] + encoder_channels[3], encoder_channels[3])
        self.up_conv3 = DoubleConv(encoder_channels[3] + encoder_channels[2], encoder_channels[2])
        self.up_conv2 = DoubleConv(encoder_channels[2] + encoder_channels[1], encoder_channels[1])
        self.up_conv1 = DoubleConv(encoder_channels[1] + encoder_channels[0], encoder_channels[0])
        self.up_conv0 = DoubleConv(encoder_channels[0] + 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.conv0(x)
        e1, e2, e3, e4, e5 = self.encoder(x)
        
        e1 = self.conv1(e1)
        e2 = self.conv2(e2)
        e3 = self.conv3(e3)
        e4 = self.conv4(e4)

        # Decoder
        d4 = self.up_conv4(torch.cat([F.interpolate(e5, size=e4.shape[2:], mode='bilinear', align_corners=True), e4], dim=1))
        d3 = self.up_conv3(torch.cat([F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True), e3], dim=1))
        d2 = self.up_conv2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True), e2], dim=1))
        d1 = self.up_conv1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True), e1], dim=1))
        d0 = self.up_conv0(torch.cat([F.interpolate(d1, size=x0.shape[2:], mode='bilinear', align_corners=True), x0], dim=1))

        return self.final_conv(d0)

# For debugging purposes
if __name__ == "__main__":
    model = UNet(encoder_name='resnet18', in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(f"Output shape: {output.shape}")