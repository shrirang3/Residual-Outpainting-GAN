import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Define constants for patch size (used by discriminator)
patch_h, patch_w = 8, 8
patch = (1, patch_h, patch_w)

def weights_init_normal(m):
    """Initialize network weights with normal distribution."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

# Modified Generator with Residual Blocks and U-Net Skip Connections
class CEGenerator(nn.Module):
    def __init__(self, channels=3, extra_upsample=False):
        super(CEGenerator, self).__init__()
        self.extra_upsample = extra_upsample

        # Initial convolution block
        self.init_conv = nn.Sequential(
            nn.Conv2d(channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(1024),
            ResidualBlock(1024),
            ResidualBlock(1024)
        )
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        if self.extra_upsample:
            self.up5 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

        # Final layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(64 if extra_upsample else 64, channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.init_conv(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)

        # Residual blocks
        res = self.res_blocks(d5)

        # Decoder with skip connections
        u1 = self.up1(res)
        u1 = torch.cat([u1, d4], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)

        if self.extra_upsample:
            u5 = self.up5(u4)
            return self.final_conv(u5)

        return self.final_conv(u4)


class CEDiscriminator(nn.Module):
    def __init__(self, channels=3):
        super(CEDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


# VGG Perceptual Loss
class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:36].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        vgg_input = self.vgg(input)
        vgg_target = self.vgg(target).detach()
        return self.criterion(vgg_input, vgg_target)