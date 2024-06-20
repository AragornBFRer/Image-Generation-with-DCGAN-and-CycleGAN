# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator       --> Used in the vanilla GAN in Part 1
#   - CycleGenerator    --> Used in the CycleGAN in Part 2
#   - DCDiscriminator   --> Used in both the vanilla GAN and CycleGAN (Parts 1 and 2)

import torch
import torch.nn as nn
import torch.nn.functional as F


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim):
        super(DCGenerator, self).__init__()

        # deconv dimension calculation:
        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        # parameters for deconv:
        # in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True
        self.deconv1 = deconv(noise_size, conv_dim * 4, 4, 1, 0)            # 1x1 -> 4x4
        self.deconv2 = deconv(conv_dim * 4, conv_dim * 2, 4)                   # 4x4 -> 8x8
        self.deconv3 = deconv(conv_dim * 2, conv_dim, 4)                    # 8x8 -> 16x16
        self.deconv4 = deconv(conv_dim, 3, 4, batch_norm=False)   # 16x16 -> 32x32


    def forward(self, z):
        """Generates an image given a sample of random noise.

            Input
            -----
                z: BS x noise_size x 1 x 1   -->  16x100x1x1

            Output
            ------
                out: BS x channels x image_width x image_height  -->  16x3x32x32
        """

        out = F.relu(self.deconv1(z))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = F.tanh(self.deconv4(out))
        return out


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, conv_dim, init_zero_weights):
        super(CycleGenerator, self).__init__()

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv(3, conv_dim // 2, 4,
                          init_zero_weights=init_zero_weights)
        self.conv2 = conv(conv_dim // 2, conv_dim, 4,
                          init_zero_weights=init_zero_weights)

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(conv_dim)

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv1 = deconv(conv_dim, conv_dim // 2, 4)
        self.deconv2 = deconv(conv_dim // 2, 3, 4, batch_norm=False)


    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.relu(self.resnet_block(out))

        out = F.relu(self.deconv1(out))
        out = F.tanh(self.deconv2(out))

        return out


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim):
        super(DCDiscriminator, self).__init__()

        # Output dimension calculation [(W-K+2P)/S]+1. 
        # W is the input volume, K is the Kernel size, P is the padding, S is the stride
        # stride=2, padding=1, batch_norm=True, init_zero_weights=False   
        self.conv1 = conv(3, conv_dim, 4)   # 32x32 -> 16x16
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)   # 16x16 -> 8x8
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)   # 8x8 -> 4x4
        self.conv4 = conv(conv_dim * 4, 1, 4, 1, 0, batch_norm=False)  # 4x4 -> 1x1


    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = self.conv4(out).squeeze()
        out = F.sigmoid(out)
        return out

