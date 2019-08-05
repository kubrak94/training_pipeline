'''
    A simple autoencoder model that repeats previous works
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from src.models.common_blocks import ConvBlock, ConvTransposeBlock,ResBlock

class ResnetEncoderBlock(nn.Sequential):
    
    def __init__(self, in_channels, features_num, expand_rate, kernel_sizes=[5,3,3,3,3,3], 
                 strides = [1,1,1,1,1,1], paddings = [2,1,1,1,1,1], bias = False):
        super(ResnetEncoderBlock, self).__init__()
        alias = "_".join(["Conv2d", str(kernel_sizes[0]), str(features_num)])
        self.add_module(alias,  ConvBlock(in_channels, features_num, kernel_sizes[0], strides[0], 
                                              paddings[0], bias))
        in_channels = features_num
        for i in range(1,len(kernel_sizes)):
            alias = "_".join(["ResBlock", str(i), str(kernel_sizes[i]), str(features_num)])
            if i % 2 == 0:
                isDownsample = True
                features_num *= expand_rate
            else:
                isDownsample = False
            self.add_module(alias,  ResBlock(in_channels, features_num, kernel_sizes[i], strides[i], 
                                              paddings[i], bias, isDownsample))
            in_channels = features_num
            
class DecoderBlock(nn.Sequential):
    
    def __init__(self, features_num, in_channels, collaps_rate, kernel_sizes=[5,2,5,2,5], 
                 strides = [1,2,1,2,1], paddings = [2,0,2,0,2], bias = True, final_activation='relu'):
        super(DecoderBlock, self).__init__()
        for i in range(len(kernel_sizes)):
            if i !=  len(kernel_sizes)-1:
                activation = 'relu'
            else:
                activation = final_activation
            alias = "_".join(["Conv2d_transpose", str(kernel_sizes[i]), str(features_num)])
            if i == (len(kernel_sizes) - 1):
                self.add_module(alias,  ConvTransposeBlock(features_num, in_channels, kernel_sizes[i], strides[i], 
                                              paddings[i], bias, activation))
            elif strides[i] == 2:
                self.add_module(alias,  ConvTransposeBlock(features_num, features_num//collaps_rate, kernel_sizes[i], strides[i], paddings[i], bias, activation))
                features_num//=collaps_rate
            else:
                self.add_module(alias,  ConvTransposeBlock(features_num, features_num, kernel_sizes[i], strides[i], paddings[i], bias, activation))

        
class MyResnetAutoencoder(nn.Module):
    def __init__(self, in_channels=1, start_features_num=16, expand_rate=2, kernel_sizes=[5,3,3,3,3,3], 
                 strides = [1,1,1,1,1,1], paddings = [2,1,1,1,1,1], bias = False, final_activation='relu'):
        super(MyResnetAutoencoder, self).__init__()
        self.encoder = ResnetEncoderBlock(in_channels, start_features_num, expand_rate, kernel_sizes, strides,
                                    paddings, bias)
        
        max_features = start_features_num * expand_rate * (len(kernel_sizes) // 2 - 1)
        self.decoder = DecoderBlock(max_features, in_channels, expand_rate,
                                    bias = bias, final_activation=final_activation)
        
        
    def forward(self, x):

        encoder_pass = self.encoder(x)
        decoder_pass = self.decoder(encoder_pass)
        
        return decoder_pass


def myResnetAutoencoder(in_channels, out_channels, final_activation):
    return MyResnetAutoencoder(in_channels, 16, 2, final_activation=final_activation)
