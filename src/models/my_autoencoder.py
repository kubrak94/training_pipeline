'''
    A simple autoencoder model that repeats previous works
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias = True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                              padding, bias = bias)
    def forward(self, x):
        return F.relu(self.conv(x))
        
class ConvTransposeBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias = True):
        super(ConvTransposeBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 
                                              padding, bias=bias)
    def forward(self, x):
        return F.relu(self.conv(x))

class EncoderBlock(nn.Sequential):
    
    def __init__(self, in_channels, features_num, expand_rate, kernel_sizes=[5,2,5,2,5], 
                 strides = [1,2,1,2,1], paddings = [2,0,2,0,2], bias = True):
        super(EncoderBlock, self).__init__()
        for i in range(len(kernel_sizes)):
            alias = "_".join(["Conv2d", str(kernel_sizes[i]), str(features_num)])
            self.add_module(alias,  ConvBlock(in_channels, features_num, kernel_sizes[i], strides[i], 
                                              paddings[i], bias))
            in_channels = features_num
            if strides[i] == 1:
                features_num *= expand_rate
            
class DecoderBlock(nn.Sequential):
    
    def __init__(self, features_num, in_channels, collaps_rate, kernel_sizes=[5,2,5,2,5], 
                 strides = [1,2,1,2,1], paddings = [2,0,2,0,2], bias = True):
        super(DecoderBlock, self).__init__()
        for i in range(len(kernel_sizes)):
            alias = "_".join(["Conv2d_transpose", str(kernel_sizes[i]), str(features_num)])
            if i == (len(kernel_sizes) - 1):
                self.add_module(alias,  ConvTransposeBlock(features_num, in_channels, kernel_sizes[i], strides[i], 
                                              paddings[i], bias))
            elif strides[i] == 2:
                self.add_module(alias,  ConvTransposeBlock(features_num, features_num//collaps_rate, kernel_sizes[i], strides[i], paddings[i], bias))
                features_num//=collaps_rate
            else:
                self.add_module(alias,  ConvTransposeBlock(features_num, features_num, kernel_sizes[i], strides[i], paddings[i], bias))

        
class MyAutoencoder(nn.Module):
    def __init__(self, in_channels=1, start_features_num=16, expand_rate=2, kernel_sizes=[5,2,5,2,5], 
                 strides=[1,2,1,2,1], paddings=[2,0,2,0,2], bias=True, out_channels=0):
        super(MyAutoencoder, self).__init__()
        self.encoder = EncoderBlock(in_channels, start_features_num, expand_rate, kernel_sizes, strides,
                                    paddings, bias)
        
        max_features = start_features_num * expand_rate*len([stride for stride in strides if stride!=1])
        self.decoder = DecoderBlock(max_features, in_channels, expand_rate, kernel_sizes, 
                 strides, paddings, bias)
        
        
    def forward(self, x):

        encoder_pass = self.encoder(x)
        decoder_pass = self.decoder(encoder_pass)
        
        return decoder_pass


def myAutoencoder():
    return MyAutoencoder(1, 16, 2)
