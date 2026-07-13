import torch.nn as nn
import torch
from torch import autograd
from functools import partial
import torch.nn.functional as F
from torchvision import models



class Unet3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters=32): 
        super(Unet3D, self).__init__()
        base = num_filters
        
        self.conv1 = DoubleConv3D(in_channels, base)          
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = DoubleConv3D(base, base*2)               
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = DoubleConv3D(base*2, base*4)             
        self.pool3 = nn.MaxPool3d(2)
        self.conv4 = DoubleConv3D(base*4, base*8)             
        self.pool4 = nn.MaxPool3d(2)
        self.conv5 = DoubleConv3D(base*8, base*16)            
        self.up6 = nn.ConvTranspose3d(base*16, base*8, kernel_size=2, stride=2)
        self.conv6 = DoubleConv3D(base*16, base*8)
        self.up7 = nn.ConvTranspose3d(base*8, base*4, kernel_size=2, stride=2)
        self.conv7 = DoubleConv3D(base*8, base*4)
        self.up8 = nn.ConvTranspose3d(base*4, base*2, kernel_size=2, stride=2)
        self.conv8 = DoubleConv3D(base*4, base*2)
        self.up9 = nn.ConvTranspose3d(base*2, base, kernel_size=2, stride=2)
        self.conv9 = DoubleConv3D(base*2, base)
        self.conv10 = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        
        c5 = self.conv5(p4)
        
        up_6 = self.up6(c5)
        merge6 = self.match_and_concat(c4, up_6)
        c6 = self.conv6(merge6)
        
        up_7 = self.up7(c6)
        merge7 = self.match_and_concat(c3, up_7)
        c7 = self.conv7(merge7)
        
        up_8 = self.up8(c7)
        merge8 = self.match_and_concat(c2, up_8)
        c8 = self.conv8(merge8)
        
        up_9 = self.up9(c8)
        merge9 = self.match_and_concat(c1, up_9)
        c9 = self.conv9(merge9)
        
        c10 = self.conv10(c9)
        
        return c10, c1


    def match_and_concat(self, encoder_feat, decoder_feat):
        if encoder_feat.shape[2:] != decoder_feat.shape[2:]:
            decoder_feat = F.interpolate(
                decoder_feat, 
                size=encoder_feat.shape[2:], 
                mode='trilinear', 
                align_corners=True
            )
        return torch.cat([encoder_feat, decoder_feat], dim=1)




class DoubleConv3D(nn.Module):
    #(Convolution3D -> BatchNorm3D -> ReLU) * 2
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def Uentropy3D(logits, c):
    # logits: [Batch, Channel, Depth, Height, Width]
    # softmax prob & log_softmax (1, 4, D, H, W)
    pc = F.softmax(logits, dim=1)  
    logpc = F.log_softmax(logits, dim=1) 
    # entropy: -p * log(p) / log(c) normalization
    u_all = -pc * logpc / math.log(c)
    
    NU = torch.sum(u_all[:, 1:, ...], dim=1)
    return NU


