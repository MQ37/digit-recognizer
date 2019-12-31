import torch
from torch import nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # start_channels
        # kernel_size
        # nonlin
        # num_layers
        # stride
        # norm
        layers = []
        sc = config["start_channels"]
        kernel_size = config["kernel_size"]
        stride = config["stride"]
        nonlin = config["nonlin"]
        norm = config["norm"]
        
        layers.append( nn.Conv2d(1, sc, kernel_size, stride=1) )
        layers.append( nonlin() )
        for i in range(config["num_layers"]):
            layers.append( nn.Conv2d(sc, sc*2, kernel_size, stride=stride) )
            layers.append( nonlin() )
            if norm:
                layers.append( norm(sc*2, affine=True, track_running_stats=True) )
            sc *= 2
        
        self.cnn = nn.Sequential(*layers)
        self.fc1 = nn.Linear( self.cnn(torch.rand(1, 1, 28, 28)).numel(), 10 )
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return torch.softmax(x, dim=1)


class ResBlock(nn.Module):
    
    def __init__(self, channel, kernel_size, use_reflection, dropout, norm, output_relu):
        super().__init__()
        self.output_relu = output_relu
        
        conv1 = []
        if use_reflection:
            conv1.append( nn.ReflectionPad2d( kernel_size // 2 ) )
            conv1.append( nn.Conv2d( channel, channel, kernel_size) )
        else:
            conv1.append( nn.Conv2d( channel, channel, kernel_size, padding=kernel_size // 2 ) )
        conv1.append( nn.ReLU(inplace=True) )
        if norm:
            conv1.append( getattr(nn, norm)(channel) )
            
        self.conv1 = nn.Sequential(*conv1)
        
        conv2 = []
        if use_reflection:
            conv2.append( nn.ReflectionPad2d( kernel_size // 2 ) )
            conv2.append( nn.Conv2d( channel, channel, kernel_size) )
        else:
            conv2.append( nn.Conv2d( channel, channel, kernel_size, padding=kernel_size // 2 ) )
        conv2.append( nn.ReLU(inplace=True) )
        if norm:
            conv2.append( getattr(nn, norm)(channel) )
            
        self.conv2 = nn.Sequential(*conv2)
        
    def forward(self, x):
        x = self.conv2(self.conv1(x)) + x
        if self.output_relu:
            x = F.relu(x)
        return x

class ResNet(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        #conv0_channel
        #conv0_kernel_size
        
        #down_repeat
        #down_kernel_size
        #down_dropout
        #down_use_reflection
        #down_norm
        
        #res_repeat
        #res_kernel_size
        #res_norm
        #res_use_reflection
        #res_dropout
        #res_output_relu
        
        
        
        layers = []
        
        layers.append( nn.ReflectionPad2d( config["conv0_kernel_size"] // 2 ) )
        layers.append( nn.Conv2d( 1, config["conv0_channel"], config["conv0_kernel_size"] ) )
        
        chan = config["conv0_channel"]
        for i in range(config["down_repeat"]):
            if config["down_use_reflection"]:
                layers.append( nn.ReflectionPad2d( config["down_kernel_size"] // 2 ) )
                layers.append( nn.Conv2d( chan, chan*2, config["down_kernel_size"], stride=2 ) )
            else:
                layers.append( nn.Conv2d( chan, chan*2, config["down_kernel_size"], stride=2, padding=config["down_kernel_size"] // 2 ) )
            layers.append( nn.ReLU(inplace=True) )
            
            if config["down_norm"]:
                layers.append( getattr(nn, config["down_norm"])(chan*2) )
            if config["down_dropout"]:
                layers.append( nn.Dropout2d(config["down_dropout"]) )
            chan *= 2
                
        for i in range( config["res_repeat"] ):
            res_layer = ResBlock( chan, config["res_kernel_size"], config["res_use_reflection"],
                                  config["res_dropout"], config["res_norm"], config["res_output_relu"])
            layers.append(res_layer)
            
        self.conv_net = nn.Sequential(*layers)
        
        self.fc1 = nn.Linear( self.conv_net( torch.rand(1, 1, 28, 28) ).numel(), config["lin_units"] )
        self.fc2 = nn.Linear( config["lin_units"], 10 )
            
        
        
    def forward(self, x):
        x = self.conv_net(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x