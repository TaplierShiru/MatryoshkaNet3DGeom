import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, floor


class ResNet(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, num_penultimate_channels, \
            input_resolution, output_resolution, num_initial_channels=16, num_inner_channels=64, \
            num_downsampling=3, num_blocks=6):

        assert num_blocks >= 0
        super(ResNet, self).__init__()
        model = [nn.BatchNorm2d(num_input_channels)]
        
        # additional down and upsampling blocks to account for difference in input/output resolution
        num_additional_down   = int(log(input_resolution / output_resolution,2)) if output_resolution <= input_resolution else 0
        num_additional_up     = int(log(output_resolution / input_resolution,2)) if output_resolution >  input_resolution else 0
        # number of channels to add during downsampling
        num_channels_down     = int(floor(float(num_inner_channels - num_initial_channels)/(num_downsampling+num_additional_down)))
        # adjust number of initial channels
        num_initial_channels += (num_inner_channels-num_initial_channels) % num_channels_down

        # initial feature block
        model += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_input_channels, num_initial_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_initial_channels),
            nn.ReLU(True)
        ]
        model += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_initial_channels, num_initial_channels, kernel_size=3, padding=0)
        ]
        # downsampling
        for i in range(num_downsampling+num_additional_down):                        
            model += [ResDownBlock(num_initial_channels, num_initial_channels+num_channels_down)]
            model += [ResSameBlock(num_initial_channels+num_channels_down)]
            num_initial_channels += num_channels_down
        # inner blocks at constant resolution
        for i in range(num_blocks):
            model += [ResSameBlock(num_initial_channels)]

        num_channels_up = int(floor(float(num_initial_channels - num_penultimate_channels)/(num_downsampling+num_additional_up)))
        # upsampling
        for i in range(num_downsampling+num_additional_up):
            model += [ResUpBlock(num_initial_channels, num_initial_channels-num_channels_up)]
            model += [ResSameBlock(num_initial_channels-num_channels_up)]
            num_initial_channels -= num_channels_up
        model += [nn.Conv2d(num_initial_channels, num_output_channels, kernel_size=3,padding=1)]
        self.model = nn.Sequential(*model)

        
    def forward(self, input):
        return self.model(input)        


class ResSameBlock(nn.Module):
    """ 
    ResNet block for constant resolution.
    
    """
    
    def __init__(self, dim):
        super(ResSameBlock, self).__init__()

        self.model = nn.Sequential(*[
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),            
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        return self.model(x) + x


class ResUpBlock(nn.Module):
    """ 
    ResNet block for upsampling.
    
    """

    def __init__(self, dim, output_dim):
        super(ResUpBlock, self).__init__()
        self.model = nn.Sequential(*[
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, output_dim, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        ])
        self.project = nn.Conv2d(dim, output_dim, kernel_size=1)

    def forward(self, x):
        xu = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.project(xu) + self.model(x)


class ResDownBlock(nn.Module):
    """ 
    ResNet block for downsampling.
    
    """
    
    def __init__(self, dim, output_dim):
        super(ResDownBlock, self).__init__()
        self.num_down = int(output_dim - dim)
        assert self.num_down > 0 # TODO: Replace operation if num lower zero
        
        self.model = nn.Sequential(*[
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, output_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        xu = x[:,:,::2,::2]
        bs,_,h,w = xu.size()
        sparse_x = torch.cat([xu, x.new_zeros(bs, self.num_down, h, w, requires_grad=False)], 1)
        return self.model(x) + sparse_x

