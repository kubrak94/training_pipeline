'''
    A simple plain U-net model that improves the autoencoder model
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
            
class DecoderBlock(nn.Module):
    
    def __init__(self, in_features, out_features, collaps_rate, kernel_sizes=[2,5], 
                 strides = [2,1], paddings = [0,2], bias = False):
        super(DecoderBlock, self).__init__()

        self.conv_transpose = ConvTransposeBlock(in_features, out_features, kernel_sizes[0], strides[0], paddings[0], bias)
        self.conv = ConvBlock(in_features, out_features, kernel_sizes[1], strides[1], paddings[1], bias)
        
    def forward(self, lower_dim, higher_dim):
        upsample = self.conv_transpose(lower_dim)
        conv = self.conv(torch.cat((upsample, higher_dim),1))
        return conv
    
class UnetEncoder(nn.Module):
    
    def __init__(self, in_channels=1, start_features_num=16, expand_rate=2, kernel_sizes=[5,2,5,2,5], 
                 strides = [1,2,1,2,1], paddings = [2,0,2,0,2], bias = False):
        super(UnetEncoder, self).__init__()
        self.encoder_blocks = []
        features_num = start_features_num
        start_in_channels = in_channels
        for i in range(len(kernel_sizes)):
            encoder_block = ConvBlock(in_channels, features_num, kernel_sizes[i], strides[i], 
                                                  paddings[i], bias)
            in_channels = features_num
            if strides[i] == 2:
                features_num*=expand_rate
            self.encoder_blocks.append(encoder_block)
        self.encoder_blocks = nn.Sequential(*self.encoder_blocks)
    
    def forward(self,x):
        encoder_passes = []
        for idx,encoder_block in enumerate(self.encoder_blocks.children()):
            x = encoder_block(x)
            if idx % 2 == 0:
                encoder_passes.append(x)
        return encoder_passes
    
class UnetDecoder(nn.Module):
    
    def __init__(self, in_channels=1, start_features_num=16, expand_rate=2, kernel_sizes=[5,2,5,2,5], 
                 strides = [1,2,1,2,1], paddings = [2,0,2,0,2], bias = False):
        super(UnetDecoder, self).__init__()
        self.decoder_blocks = []
        
        self.num_downsampling = len([stride for stride in strides if stride!=1])
        max_features = start_features_num * expand_rate*self.num_downsampling
        
        for i in range(self.num_downsampling):
            decoder_block = DecoderBlock(max_features, max_features//expand_rate, expand_rate, kernel_sizes[-2:], 
                 strides[-2:], paddings[-2:], bias)
            max_features=max_features//expand_rate
            self.decoder_blocks.append(decoder_block)
        self.decoder_blocks = nn.Sequential(*self.decoder_blocks)
        
    def forward(self,encoder_passes):
        x = encoder_passes[-1]
        for idx,decoder_block in enumerate(self.decoder_blocks.children()):
            x = decoder_block(x,encoder_passes[-(idx+2)])

        return x

class MyPlainUnet(nn.Module):
    def __init__(self, in_channels=1, start_features_num=16, expand_rate=2, kernel_sizes=[5,2,5,2,5], 
                 strides = [1,2,1,2,1], paddings = [2,0,2,0,2], bias = False):
        super(MyPlainUnet, self).__init__()
        
        self.encoder = []
        self.decoder_blocks = []
        features_num = start_features_num
        start_in_channels = in_channels
        
        self.encoder = UnetEncoder(in_channels, start_features_num, expand_rate, kernel_sizes, 
                                  strides, paddings , bias)
        
        self.decoder = UnetDecoder(in_channels, start_features_num, expand_rate, kernel_sizes, 
                                  strides, paddings , bias)
        
        self.final_conv = ConvBlock(start_features_num,start_in_channels, kernel_sizes[0],
                                     strides[0], paddings[0], bias)
        
        
    def forward(self, x):
        encoder_passes = self.encoder(x)

        x = self.decoder(encoder_passes)
        x = self.final_conv(x)
        return x


def myPlainUnet(in_channels=1, out_channels=1):
    return MyPlainUnet(1, 16, 2)
