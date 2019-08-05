'''
    A simple plain U-net model that improves the autoencoder model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from src.models.common_blocks import ConvBlock, ConvTransposeBlock
        
class UpsampleBlock(nn.Module):
    
    def __init__(self, in_channels, scale_factor = 2, kernel_size = 5, stride = 1, padding = 2, bias = False):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.smooth_conv = ConvBlock(in_channels, in_channels, kernel_size, stride, padding, bias)
        
    def forward(self, x):
        return self.smooth_conv(self.upsample(x))
    
class FPNEncoder(nn.Module):
    
    def __init__(self, in_channels=1, start_features_num=16, expand_rate=2, kernel_sizes=[5,2,5,2,5], 
                 strides = [1,2,1,2,1], paddings = [2,0,2,0,2], bias = False):
        super(FPNEncoder, self).__init__()
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
            
class DecoderBlock(nn.Module):
    
    def __init__(self, in_features, out_features, kernel_size=5, 
                 stride = 1, padding = 2, bias = False, num_layers = 3, layer_num = 2):
        super(DecoderBlock, self).__init__()
        self.num_layers = num_layers
        self.layer_num = layer_num
        self.conv_equalize = nn.Conv2d(in_features, out_features, 1, 1, 0, bias=bias)
        self.conv_reduce = nn.Conv2d(out_features, out_features // 2, kernel_size, stride, padding, bias=bias)
        self.fmap_upsample = UpsampleBlock(out_features, 2, kernel_size, stride, padding, bias)
        if layer_num != 0:
            self.pred_upsample = UpsampleBlock(out_features // 2, 2*(layer_num),kernel_size,stride, padding, bias)
        
        
    def forward(self, fmap_encoder, fmap_decoder):
        
        fmap_output = None
        pred_output = None
        conv_equalize = self.conv_equalize(fmap_encoder)
        if self.layer_num != self.num_layers:
            fmap_decoder = fmap_decoder + conv_equalize
        else:
            fmap_decoder = conv_equalize
        conv_reduce = self.conv_reduce(fmap_decoder)
        
        if self.layer_num != 0:
            fmap_output = self.fmap_upsample(fmap_decoder)
            pred_output = self.pred_upsample(conv_reduce)
        else:
            pred_output = conv_reduce
        return fmap_output, pred_output
    
class FPNDecoder(nn.Module):
    
    def __init__(self, in_channels=1, start_features_num=16, expand_rate=2, kernel_sizes=[5,2,5,2,5], 
                 strides = [1,2,1,2,1], paddings = [2,0,2,0,2], bias = False):
        super(FPNDecoder, self).__init__()
        self.decoder_blocks = []
        
        self.num_downsampling = len([stride for stride in strides if stride!=1])
        max_features = start_features_num * expand_rate*self.num_downsampling
        
        for i in range(self.num_downsampling + 1):
            decoder_block = DecoderBlock(max_features, start_features_num, kernel_sizes[0], 
                 strides[0], paddings[0], bias, self.num_downsampling, self.num_downsampling - i)
            max_features=max_features//expand_rate
            self.decoder_blocks.append(decoder_block)
        self.decoder_blocks = nn.Sequential(*self.decoder_blocks)
        
    def forward(self,encoder_passes):
        fmap_output = None
        pred_outputs = None
        for idx,decoder_block in enumerate(self.decoder_blocks.children()):
            fmap_output, pred_output = decoder_block(encoder_passes[-(idx+1)], fmap_output)
            if idx==0:
                shape = list(pred_output.shape)
                shape[1] = shape[1] * len(self.decoder_blocks)
                pred_outputs = torch.zeros(shape,dtype=torch.float32, device=pred_output.device)
            pred_outputs[:,idx*pred_output.shape[1]:(idx+1)*pred_output.shape[1]] = pred_output
        
        return pred_outputs

class MyPlainFPN(nn.Module):
    def __init__(self, in_channels=1, start_features_num=16, expand_rate=2, kernel_sizes=[5,2,5,2,5], 
                 strides = [1,2,1,2,1], paddings = [2,0,2,0,2], bias = False, final_activation='relu'):
        super(MyPlainFPN, self).__init__()
        
        self.encoder = []
        self.decoder_blocks = []
        features_num = start_features_num
        start_in_channels = in_channels
        self.num_downsampling = len([stride for stride in strides if stride!=1])
        
        self.encoder = FPNEncoder(in_channels, start_features_num, expand_rate, kernel_sizes, 
                                  strides, paddings , bias)
        
        self.decoder = FPNDecoder(in_channels, start_features_num, expand_rate, kernel_sizes, 
                                  strides, paddings , bias)
        
        self.final_conv = ConvBlock(start_features_num//2*(self.num_downsampling+1),start_in_channels, kernel_sizes[0],
                                     strides[0], paddings[0], bias, final_activation)
        
        
    def forward(self, x):
        encoder_passes = self.encoder(x)

        x = self.decoder(encoder_passes)
        x = self.final_conv(x)
        return x


def myPlainFPN(in_channels, out_channels, final_activation):
    return MyPlainFPN(in_channels, 16, 2, final_activation=final_activation)
