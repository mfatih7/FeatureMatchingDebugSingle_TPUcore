import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

# Checkpointing can be used with a module or a function of a module

class Conv2d_N_REL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn_or_gn):
        super(Conv2d_N_REL, self).__init__()
        
        self.GNseperation = 8
        
        self.cnn = nn.Conv2d(    in_channels = in_channels,
                                 out_channels = out_channels,
                                 kernel_size = kernel_size,
                                 stride = stride,
                                 padding = padding,
                                 bias = False,)           
        if(bn_or_gn == 'bn'):
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.GroupNorm( int(out_channels/self.GNseperation), out_channels)
            
        self.non_lin = nn.ReLU()
            
    def forward(self, x):        
        x = self.non_lin( self.norm( self.cnn(x) ) )
        return x
    
class Squeeze_Excite(nn.Module):
    def __init__(self, in_channels, squeeze_excite_channels, out_channels ):
        super(Squeeze_Excite, self).__init__()
            
        self.squeeze_excite = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_excite_channels, 1),
            nn.SiLU(),
            nn.Conv2d(squeeze_excite_channels, out_channels, 1),
            nn.Sigmoid(),            
        )
            
    def forward(self, x):
        
        return x * self.squeeze_excite(x)

class Inverted_Residual_with_Linear_Bottleneck_and_Squeeze_Excite(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor, squeeze_and_excite_reduction, h_swish_or_relu ):
        super(Inverted_Residual_with_Linear_Bottleneck_and_Squeeze_Excite, self).__init__()

        self.padding_mode = 'circular'  
        
        self.expansion_conv = nn.Conv2d( 
                                          in_channels = in_channels,
                                          out_channels = in_channels*expansion_factor,
                                          kernel_size = (1,1),
                                          stride = (1,1),
                                          bias = False,
                                        )        
        self.batchnorm1 = nn.BatchNorm2d( in_channels*expansion_factor)
        
        depth_wise_conv_with_padding_list = []
            
        if(stride==(1,1)):
            depth_wise_conv_with_padding_list.append( nn.ReflectionPad2d( (0, 0, 0, 1) ) ) # padding_left, padding_right, padding_top, padding_bottom
        
        depth_wise_conv_with_padding_list.append( 
                                                   nn.Conv2d( 
                                                                 in_channels = in_channels*expansion_factor,
                                                                 out_channels = in_channels*expansion_factor,
                                                                 kernel_size = kernel_size,
                                                                 stride = stride,
                                                                 groups = in_channels*expansion_factor,
                                                                 bias = False,
                                                            )
                                                )
        self.depth_wise_conv_with_padding = nn.Sequential(*depth_wise_conv_with_padding_list)
        
        self.batchnorm2 = nn.BatchNorm2d(in_channels*expansion_factor)
        
        self.squeeze_Excite = Squeeze_Excite( 
                                                in_channels*expansion_factor,
                                                math.ceil(in_channels*expansion_factor/squeeze_and_excite_reduction),
                                                in_channels*expansion_factor, )
        
        self.point_wise_conv = nn.Conv2d( 
                                          in_channels = in_channels*expansion_factor,
                                          out_channels = out_channels,
                                          kernel_size = (1,1),
                                          stride = (1,1),
                                          padding = (0,0),
                                          padding_mode = self.padding_mode,
                                          bias = False,
                                        )
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        
        if(h_swish_or_relu == 'RE'):
            self.activation_non_lin = nn.ReLU6()
        elif(h_swish_or_relu == 'HS'):
            self.activation_non_lin = nn.Hardswish()
        else:
            assert(0)       
        
    def forward(self, x):
        
        x = self.activation_non_lin( self.batchnorm1( self.expansion_conv( x ) ) )
        x = self.activation_non_lin( self.squeeze_Excite( self.batchnorm2( self.depth_wise_conv_with_padding(x) ) ) )
        x = self.batchnorm3( self.point_wise_conv(x) )
            
        return x
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, size_reduction, bn_or_gn):
        super(Block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size_reduction = size_reduction
        
        layers = []
        
        if( size_reduction == 1 ):        
            layers.append( Inverted_Residual_with_Linear_Bottleneck_and_Squeeze_Excite(    
                                                                                        in_channels=in_channels,
                                                                                        out_channels=out_channels,
                                                                                        kernel_size=(2,1),
                                                                                        stride=(2,1),
                                                                                        expansion_factor=6,
                                                                                        squeeze_and_excite_reduction=4,
                                                                                        h_swish_or_relu='HS', ) )

        elif( size_reduction == 0 ):
            layers.append( Inverted_Residual_with_Linear_Bottleneck_and_Squeeze_Excite(
                                                                                        in_channels=in_channels,
                                                                                        out_channels=out_channels,
                                                                                        kernel_size=(2,1),
                                                                                        stride=(1,1),
                                                                                        expansion_factor=6,
                                                                                        squeeze_and_excite_reduction=4,
                                                                                        h_swish_or_relu='HS', ) )
        self.net = nn.Sequential(*layers)
            
    def forward(self, x):        
        
        if(self.in_channels == self.out_channels and self.size_reduction == 0):
            shortcut = x
            x = self.net( x )      
            x = shortcut + x
        else:
            x = self.net( x )
            
        return x

class MobileNetV3(nn.Module):
    def __init__(self, N, bn_or_gn, en_grad_checkpointing ):
        super(MobileNetV3, self).__init__()
        
        self.N = N
        
        self.en_grad_checkpointing = en_grad_checkpointing
        
        self.n_blocks = int(math.log2(N))
        
        layers = []
        
        layers.append( Conv2d_N_REL( in_channels=2, out_channels=32, kernel_size=(1,4), stride=(1,1), padding=(0,0), bn_or_gn=bn_or_gn ) )
        layers.append( Conv2d_N_REL( in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), bn_or_gn=bn_or_gn ) )
        
        layer_count_prev = layers[-1].cnn.out_channels
        
        for block_no in range(self.n_blocks):
            
            if((block_no+2)%2 < 1):
                in_channels = layer_count_prev
                out_channels = layer_count_prev
            else:
                in_channels = layer_count_prev
                out_channels = layer_count_prev*2
                layer_count_prev = layer_count_prev*2
            
            layers.append( Block( in_channels=in_channels, out_channels=out_channels, size_reduction=1, bn_or_gn=bn_or_gn ) )
            if( block_no < self.n_blocks-1 ):
                layers.append( Block( in_channels=out_channels, out_channels=out_channels, size_reduction=0, bn_or_gn=bn_or_gn ) )

        self.fully_connected_size = layers[-1].net[-1].point_wise_conv.out_channels
            
        self.fc1 = nn.Linear(self.fully_connected_size * 1 * 1, int(self.fully_connected_size/4) )
        
        self.fc2 = nn.Linear(int(self.fully_connected_size/4) * 1 * 1, 10)
        
        self.non_lin = nn.ReLU()  
        
        self.net = nn.Sequential(*layers)

    def block_1(self, x):
        
        x = self.net( x )
        
        return x        
        
    def forward(self, x):
        
        if(self.en_grad_checkpointing == False or self.training == False):
            
            x = self.block_1( x )
        else:
            
            x = checkpoint( self.block_1, x )
                
        x = x.view( -1, self.fully_connected_size * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
        
def get_model( N, bn_or_gn, en_checkpointing ):   

    if( N == 512 or N == 1024 or N == 2048 or N == 4096 ):
    
        return MobileNetV3( N, bn_or_gn, en_checkpointing )
    
    else:
        print( 'Not Valid N ' + str(N) )
        assert(0)
