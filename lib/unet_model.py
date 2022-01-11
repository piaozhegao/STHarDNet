""" Full assembly of the parts to form the complete network """
import logging

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)

        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        logging.debug("input： {}".format(x.shape))
        x1 = self.inc(x)
        logging.debug(x1 .shape)
        x2 = self.down1(x1)
        logging.debug(x2.shape)
        x3 = self.down2(x2)
        logging.debug(x3.shape)
        x4 = self.down3(x3)
        logging.debug(x4.shape)
        x5 = self.down4(x4)
        logging.debug(x5.shape)

        logging.debug("up sampling ")
        x = self.up1(x5, x4)
        logging.debug("cnn_decoder_1:{}".format(x.shape))
        x = self.up2(x, x3)
        logging.debug("cnn_decoder_2:{}".format(x.shape))
        x = self.up3(x, x2)
        logging.debug("cnn_decoder_3:{}".format(x.shape))
        x = self.up4(x, x1)
        logging.debug("cnn_decoder_4:{}".format(x.shape))
        logits = self.outc(x)
        logging.debug("cnn_out:{}".format(logits.shape))
        return logits
