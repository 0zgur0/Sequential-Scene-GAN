import torch
from torch import nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm
import numpy as np

channels = 3
GEN_SIZE=128
DISC_SIZE=128

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        
        
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class ResBlockGenerator2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )

    def forward(self, x):
        return self.model(x) + x
    
    
class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            
    def forward(self, x):
        return self.model(x) + self.bypass(x)

class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Discriminator(nn.Module):
    def __init__(self, channels=channels, input_size=128):
        super(Discriminator, self).__init__()
        self.avg_kernel_size = int(input_size/32)
        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                nn.AvgPool2d(self.avg_kernel_size)
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.model(x).view(-1,DISC_SIZE)
        x = self.fc(x)
        return self.activation( x )


class Generator_BG(nn.Module):
    def __init__(self, z_dim, label_channel=0, num_res_blocks=2, num_res_blocks_fg=0, num_res_blocks_bg=0):
        super(Generator_BG, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.num_res_blocks_fg = num_res_blocks_fg
        self.num_res_blocks_bg = num_res_blocks_bg
        self.z_dim = z_dim
        
        self.cond_64 = nn.Conv2d(int(label_channel), GEN_SIZE//2, 4, 2, 1, bias=False)
        self.cond_64_BN = nn.BatchNorm2d(GEN_SIZE//2) 
        self.cond_16 = nn.AvgPool2d(4)
        nn.init.xavier_uniform(self.cond_64.weight.data, 1.)
        
        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE*3//2, channels, 3, stride=1, padding=1)
        self.final_bg = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)
        nn.init.xavier_uniform(self.final_bg.weight.data, 1.)

        self.model1 = nn.Sequential(
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
             )
        
        self.model2 = nn.Sequential()
        for j in range(self.num_res_blocks):
            self.model2.add_module(str(j), ResBlockGenerator2(GEN_SIZE, GEN_SIZE))
            
        
        self.model3_res = nn.Sequential()
        for j in range(self.num_res_blocks_fg):
            self.model3_res.add_module(str(j), ResBlockGenerator2(GEN_SIZE*3//2, GEN_SIZE*3//2))
        
        self.model3 = nn.Sequential(
            ResBlockGenerator(GEN_SIZE*3//2, GEN_SIZE*3//2, stride=2),
            ResBlockGenerator(GEN_SIZE*3//2, GEN_SIZE*3//2, stride=2),
            ResBlockGenerator(GEN_SIZE*3//2, GEN_SIZE*3//2, stride=2),
            nn.BatchNorm2d(GEN_SIZE*3//2),
            nn.ReLU(),
            self.final,
            nn.Tanh())
        
        
        self.model3_bg_res = nn.Sequential()
        for j in range(self.num_res_blocks_bg):
            self.model3_bg_res.add_module(str(j), ResBlockGenerator2(GEN_SIZE, GEN_SIZE))     
        
        self.model3_bg = nn.Sequential(
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final_bg,
            nn.Tanh())

    def forward(self, z, y):
        y = self.cond_16( F.relu(self.cond_64_BN(self.cond_64(y) )))
        x = self.model1( self.dense(z).view(-1, GEN_SIZE, 4, 4) )
        x = self.model2(x)
        x_non_bg = torch.cat([x,y],1)
        
        x = self.model3_bg_res(x)
        x = self.model3_bg(x)
        
        x_non_bg = self.model3_res(x_non_bg)
        x_non_bg = self.model3(x_non_bg)
        return x_non_bg, x
    


class Generator_FG(nn.Module):
    #Noise summary and mask concat at 16x16 level and start generating image
    def __init__(self, z_dim, label_channel=0, num_res_blocks=2):
        super(Generator_FG, self).__init__()
        self.z_dim = z_dim
        self.num_res_blocks = num_res_blocks
        
        self.dense = nn.Linear(self.z_dim, 16 * 16 * GEN_SIZE)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        
        self.final = nn.Conv2d(GEN_SIZE*7//2, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.final.weight.data, 1.)
        
        self.cond_64 = nn.Conv2d(int(label_channel), GEN_SIZE//2, 4, 2, 1, bias=False)
        #self.cond_64_BN = nn.BatchNorm2d(GEN_SIZE//2) 
        nn.init.xavier_uniform(self.cond_64.weight.data, 1.)
        self.cond_16 = nn.AvgPool2d(4)
        
        self.recon_64 = nn.Conv2d(3, GEN_SIZE//2, 4, 2, 1, bias=False)
        self.recon_64_BN = nn.BatchNorm2d(GEN_SIZE//2) 
        self.recon_32 = nn.Conv2d(GEN_SIZE//2, GEN_SIZE//2, 4, 2, 1, bias=False)
        self.recon_32_BN = nn.BatchNorm2d(GEN_SIZE//2) 
        self.recon_16 = nn.Conv2d(GEN_SIZE//2, GEN_SIZE//2, 4, 2, 1, bias=False)
        self.recon_16_BN = nn.BatchNorm2d(GEN_SIZE//2) 
        
        nn.init.xavier_uniform(self.recon_64.weight.data, 1.)
        nn.init.xavier_uniform(self.recon_32.weight.data, 1.)     
        nn.init.xavier_uniform(self.recon_16.weight.data, 1.)

        self.model2 = nn.Sequential()
        for j in range(self.num_res_blocks):
            self.model2.add_module(str(j), ResBlockGenerator2(GEN_SIZE*2, GEN_SIZE*2))
        
        
        self.model3 = nn.Sequential(
            ResBlockGenerator(GEN_SIZE*3, GEN_SIZE*3, stride=2),
            ResBlockGenerator(GEN_SIZE*3, GEN_SIZE*3, stride=2)
            )
        
        self.model4 = nn.Sequential(
            ResBlockGenerator(GEN_SIZE*7//2, GEN_SIZE*7//2, stride=2),
            nn.BatchNorm2d(GEN_SIZE*7//2),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z, y, prev_img):
        z = self.dense(z).view(-1, GEN_SIZE, 16, 16) 
        #y = self.cond_16( F.relu(self.cond_64_BN(self.cond_64(y) )))
        y = self.cond_16( self.cond_64(y) )
        prev_img = F.relu(self.recon_64_BN(self.recon_64(prev_img)))
        prev_img_late = F.relu(self.recon_32_BN(self.recon_32(prev_img)))
        prev_img_late = F.relu(self.recon_16_BN(self.recon_16(prev_img_late)))
        
        x = torch.cat([z,y,prev_img_late],1)
        x = self.model2(x)
        
        x = torch.cat([z,x ],1)
        x = self.model3(x)
     
        x = torch.cat([x,prev_img],1)
        x = self.model4(x)
        return x            



class Generator_Baseline(nn.Module):
    def __init__(self, z_dim, label_channel=0, num_res_blocks=2):
        super(Generator_Baseline, self).__init__()
        self.z_dim = z_dim
        self.num_res_blocks = num_res_blocks
        
        self.dense = nn.Linear(self.z_dim, 16 * 16 * GEN_SIZE)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        
        self.final = nn.Conv2d(GEN_SIZE*5//2, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.final.weight.data, 1.)
        
        self.cond_64 = nn.Conv2d(int(label_channel), GEN_SIZE//2, 4, 2, 1, bias=False)
        self.cond_64_BN = nn.BatchNorm2d(GEN_SIZE//2) 
        nn.init.xavier_uniform(self.cond_64.weight.data, 1.)
        self.cond_16 = nn.AvgPool2d(4)
        

        self.model1 = nn.Sequential()
        for j in range(self.num_res_blocks):
            self.model1.add_module(str(j), ResBlockGenerator2(GEN_SIZE*3//2, GEN_SIZE*3//2))
        
        
        self.model2 = nn.Sequential(
            ResBlockGenerator(GEN_SIZE*5//2, GEN_SIZE*5//2, stride=2),
            ResBlockGenerator(GEN_SIZE*5//2, GEN_SIZE*5//2, stride=2),
            ResBlockGenerator(GEN_SIZE*5//2, GEN_SIZE*5//2, stride=2),
            nn.BatchNorm2d(GEN_SIZE*5//2),
            nn.ReLU(),
            self.final,
            nn.Tanh())
        
    def forward(self, z, y):
        z = self.dense(z).view(-1, GEN_SIZE, 16, 16) 
        y = self.cond_16( F.relu(self.cond_64_BN(self.cond_64(y) )))

        x = torch.cat([z,y],1)
        x = self.model1(x)
        
        x = torch.cat([z,x],1)
        x = self.model2(x)
     
        return x            
