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

        
class MyAutoencoderTied(nn.Module):
    def __init__(self, in_channels=1, start_features_num=16, expand_rate=2, kernel_sizes=[5,2,5,2,5], 
                 strides = [1,2,1,2,1], paddings = [2,0,2,0,2], bias = False):
        super(MyAutoencoderTied, self).__init__()
        
        self.encoder = EncoderBlock(in_channels, start_features_num, expand_rate, kernel_sizes, strides,
                                    paddings, bias)
        
        max_features = start_features_num * expand_rate*len([stride for stride in strides if stride!=1])
        
        
    def forward(self, x):

        x = self.encoder(x)
        for i in range(len(self.encoder)-1,-1,-1):
            conv_layer = self.encoder[i].conv
            x = F.conv_transpose2d(input=x, weight=conv_layer.weight, padding=conv_layer.padding, stride=conv_layer.stride,bias=conv_layer.bias)
            x = F.relu(x)
        return x


def myAutoencoderTied():
    return MyAutoencoderTied(1, 16, 2)
