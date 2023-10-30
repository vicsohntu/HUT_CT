import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from unfoldNd import UnfoldNd, FoldNd
from vit_pytorch.cross_vit import CrossViTCLS2D

class ConvUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 5, 2, padding=2, output_padding=1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = self.model(x)
        return x

class ConvPass(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_size, out_size, 3, 1, 1, bias=False),
        ]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = self.model(x)
        return x

class ConvUp2d(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 5, 2, padding=2, output_padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = self.model(x)
        return x

class ConvPass2d(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU(inplace=True))
        #layers.append(nn.LeakyReLU(0.1))
        #layers.append(nn.GELU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = self.model(x)
        return x

class UD(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv3d(in_size, out_size, 3, 1, 1, bias=False)]
        layers.append(nn.Conv3d(out_size, out_size, 5, 2, 2, bias=False))
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class UU(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 5, 2, padding=2, output_padding=1, bias=False),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        self.outlayer=nn.Conv3d(out_size*2, out_size, 3, 1, 1, bias=False)
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        x = self.outlayer(x)
        return x

class UD2d(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False)]
        layers.append(nn.Conv2d(out_size, out_size, 5, 2, 2, bias=False))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class UU2d(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 5, 2, padding=2, output_padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        self.outlayer=nn.Conv2d(out_size*2, out_size, 3, 1, 1, bias=False)
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        x = self.outlayer(x)
        return x

class SConvDown(nn.Module):
    def __init__(self, in_size, out_size, passthrough=True, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv3d(in_size, out_size, 3, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size, affine=True))
        layers.append(nn.LeakyReLU(0.1))
        if dropout:
            layers.append(nn.Dropout(dropout))
        if passthrough:
            layers.append(nn.Conv3d(out_size, out_size, 3, 1, 1, bias=False))
            layers.append(nn.InstanceNorm3d(out_size, affine=True))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class SConvPass(nn.Module):
    def __init__(self, in_size, out_size, passthrough=True, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv3d(in_size, out_size, 3, 1, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm3d(out_size, affine=True))
            layers.append(nn.LeakyReLU(0.1))
        if dropout:
            layers.append(nn.Dropout(dropout))
        if passthrough:
            layers.append(nn.Conv3d(out_size, out_size, 3, 1, 1, bias=False))
            layers.append(nn.InstanceNorm3d(out_size, affine=True))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class SConvUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, 3, 2, 1, 1, bias=False),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class SConvDown2d(nn.Module):
    def __init__(self, in_size, out_size, passthrough=True, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        layers.append(nn.LeakyReLU(0.1))
        if dropout:
            layers.append(nn.Dropout(dropout))
        if passthrough:
            layers.append(nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False))
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class SConvPass2d(nn.Module):
    def __init__(self, in_size, out_size, passthrough=True, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
            layers.append(nn.LeakyReLU(0.1))
        if dropout:
            layers.append(nn.Dropout(dropout))
        if passthrough:
            layers.append(nn.Conv2d(out_size, out_size, 3, 1, 1, bias=False))
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class SConvUp2d(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 3, 2, 1, 1, bias=False),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
    
class DualRes2d(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        layers = [nn.BatchNorm2d(in_size, affine=True)]
        layers.append(nn.PReLU())
        layers.append(nn.Conv2d(in_size, in_size, 3, 1, 1, bias=False))
        layers.append(nn.Dropout(0.2))
        layers.append(nn.BatchNorm2d(in_size, affine=True))
        layers.append(nn.PReLU())
        layers.append(nn.Conv2d(in_size, in_size, 3, 1, 1, bias=False))
        self.pipe1 = nn.Sequential(*layers)
    def forward(self, x):
        return x+self.pipe1(x)

class MonoRes2d(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        layers = [nn.BatchNorm2d(in_size, affine=True)]
        layers.append(nn.PReLU())        
        layers.append(nn.Conv2d(in_size, in_size, 3, 1, 1, bias=False))
        self.pipe1 = nn.Sequential(*layers)
    def forward(self, x):
        return x+self.pipe1(x)

class DownConv2d(nn.Module):
    def __init__(self, in_size, dropout=0.0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        layers = [nn.BatchNorm2d(in_size, affine=True)]
        layers.append(nn.PReLU())
        layers.append(nn.Conv2d(in_size, in_size, 3, 2, 1, bias=False))
        self.pipe1 = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.drop = dropout
    def forward(self, x):
        if self.drop>0.0:
            return self.dropout(torch.concat((self.maxpool(x), self.pipe1(x)),dim=1))
        else:
            return torch.concat((self.maxpool(x), self.pipe1(x)),dim=1)

class UpConv2d(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 3, 2, 1, 1, bias=False)]
        if dropout>0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class ConvPass2d2(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False)]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)


class UNetDown2d(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 5, 2, 2, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp2d(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 5, 2, padding=2, output_padding=1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class FCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.25):
        super().__init__()
        self.pass0 = ConvPass2d2(in_channels, 8)
        self.dres0 = DualRes2d(8)
        self.down0 = DownConv2d(8) #out: 16
        self.dres1 = DualRes2d(16)
        self.down1 = DownConv2d(16, dropout=dropout) #out: 32
        self.dres2 = DualRes2d(32)
        self.down2 = DownConv2d(32, dropout=dropout) #out: 64
        self.dres3 = DualRes2d(64)
        self.down3 = DownConv2d(64, dropout=dropout) #out: 128
        self.dres4 = DualRes2d(128)
        
        self.up0 = UpConv2d(128, 64)
        self.mres0= MonoRes2d(64)
        self.up1 = UpConv2d(64, 32)
        self.mres1= MonoRes2d(32)
        self.up2 = UpConv2d(32, 16)        
        self.mres2= MonoRes2d(16)
        self.up3 = UpConv2d(16, 8)
        self.mres3= MonoRes2d(8)
        self.pass1 = ConvPass2d2(8, 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        p0 = self.pass0(x)
        s0 = self.dres0(p0)
        d0 = self.down0(s0)
        s1 = self.dres1(d0)
        d1 = self.down1(s1)
        s2 = self.dres2(d1)
        d2 = self.down2(s2)
        s3 = self.dres3(d2)
        d3 = self.down3(s3)
        s4 = self.dres4(d3)
        u0 = self.up0(s4)
        m0 = self.mres0(self.dropout(s3+u0))
        u1 = self.up1(m0)
        m1 = self.mres1(self.dropout(s2+u1))
        u2 = self.up2(m1)
        m2 = self.mres2(s1+u2)        
        u3 = self.up3(m2)
        m3 = self.mres3(s0+u3)
        p1 = self.pass1(m3)
        return p1


class MSConv2d(nn.Module):
    def __init__(self, in_size, dim, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, in_size, 3, 1, 1, bias=False)]
        layers.append(nn.InstanceNorm2d(in_size))
        layers.append(nn.ReLU(inplace=True))
        self.pipe0 = nn.Sequential(*layers)
        self.pipe33 = nn.Conv2d(dim, dim, 3, 1, dilation=1, padding=1, bias=False)
        self.pipe11H = nn.Conv2d(in_size, dim//2, 1, 1, dilation=1, bias=False)
        self.pipe11F = nn.Conv2d(in_size, dim, 1, 1, dilation=1, bias=False)
        self.pipe11F2 = nn.Conv2d(dim, dim, 1, 1, dilation=1, bias=False)
        self.pipe33D4 = nn.Conv2d(in_size, dim, 3, 1, dilation=4, padding=4, bias=False)
        self.pipe33D2 = nn.Conv2d(dim, dim, 3, 1, dilation=2, padding=2, bias=False)
        self.output = nn.Conv2d(dim//2+dim*3, in_size, 1, 1, dilation=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.drop = dropout
    def forward(self, x):
        x=self.pipe0(x)
        x1=self.pipe11H(x)
        x2=self.pipe33(self.pipe11F(x))
        x3=self.pipe33D2(self.pipe11F(x))
        x4=self.pipe11F2(self.pipe33D4(x))
        o0=torch.concat((x1,x2,x3,x4), dim=1)
        o1=self.output(o0)+x
        return o1

class DownUSSLConv2d(nn.Module):
    def __init__(self, in_size, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, in_size, 3, 1, 1, bias=False)]
        layers.append(nn.InstanceNorm2d(in_size))
        layers.append(nn.ReLU(inplace=True))
        self.pipe1 = nn.Sequential(*layers)
        self.down = nn.Conv2d(in_size, in_size*2, 3, 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.drop = dropout
    def forward(self, x):
        skip=self.pipe1(x)
        if self.drop>0.0:
            return self.dropout(self.down(skip)), skip
        else:
            return self.down(skip), skip

class UpUSSLConv2d(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 3, 2, 1, 1, bias=False)]
        if dropout>0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class UpUSSLConv2d2(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 3, 4, 1, 3, bias=False)]
        if dropout>0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
    
class USSLNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.25):
        super().__init__()
        self.pass0 = ConvPass2d2(in_channels, 16)
        self.down0 = DownUSSLConv2d(16,dropout=dropout) #out: 32
        self.msconv1 = MSConv2d(32,128, dropout=dropout) # out 32
        self.down1 = DownUSSLConv2d(32,dropout=dropout) #out: 64
        self.msconv2 = MSConv2d(64,128, dropout=dropout) # out 64
        self.down2 = DownUSSLConv2d(64,dropout=dropout) #out: 128
        self.msconv3 = MSConv2d(128,128, dropout=dropout) # out 128
        self.down3 = DownUSSLConv2d(128,dropout=dropout) #out: 256
        self.msconv4 = MSConv2d(256,128, dropout=dropout) # out 256
        self.pass1 = ConvPass2d2(256, 256)
        
        self.up0 = UpUSSLConv2d(256, 128, dropout=dropout)
        self.p0 = ConvPass2d2(256, 128)
        self.up1 = UpUSSLConv2d(128, 64)
        self.p1 = ConvPass2d2(128, 64)
        self.up2 = UpUSSLConv2d(64, 32)
        self.p2 = ConvPass2d2(64, 32)
        self.up3 = UpConv2d(32, 16)
        self.s0 = ConvPass2d2(16, out_channels)
        self.s1 = UpUSSLConv2d(32, out_channels)
        self.s2 = UpUSSLConv2d2(64, out_channels)
        self.sm0 = nn.Softmax(dim=1)
    def forward(self, x):
        p0 = self.pass0(x)
        d0,s0=self.down0(p0)
        m1 = self.msconv1(d0)
        d1,s1=self.down1(m1)
        m2 = self.msconv2(d1)
        d2,s2=self.down2(m2)
        m3 = self.msconv3(d2)
        d3,s3=self.down3(m3)
        m4 = self.msconv4(d3)
        p1 = self.pass1(m4)

        u0 = self.up0(p1+m4)
        c0 = self.p0(torch.concat((s3,u0), dim=1))
        u1 = self.up1(c0)
        c1 = self.p1(torch.concat((s2,u1), dim=1))
        u2 = self.up2(c1)
        c2 = self.p2(torch.concat((s1,u2), dim=1))
        u3 = self.up3(c2)
        seg0 = self.s0(u3)   
        seg1 = self.s1(u2)
        seg2 = self.s2(u1)
        out = (seg0+seg1+seg2)
        return out

class SUNET2d_Deep(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.35, tr_dropout=0.35):
        super().__init__()
        self.pass0 = SConvPass2d(in_channels, 16, normalize=False, dropout=dropout)
        self.down1 = SConvDown2d(16, 32, dropout=dropout)
        self.down2 = SConvDown2d(32, 64, dropout=dropout)
        self.down3 = SConvDown2d(64, 128, dropout=dropout)
        self.down4 = SConvDown2d(128, 256, dropout=dropout)
        self.down5 = SConvDown2d(256, 320, dropout=dropout)

        self.up1 = SConvUp2d(320, 256, dropout=dropout)
        self.pass1 = SConvPass2d(256*2, 256, dropout=dropout)
        self.up2 = SConvUp2d(256, 128, dropout=dropout)
        self.pass2 = SConvPass2d(128*2, 128, dropout=dropout)
        self.output2 = SConvPass2d(128, out_channels, passthrough=False, normalize=False, dropout=None)
        self.up3 = SConvUp2d(128, 64, dropout=dropout)
        self.pass3 = SConvPass2d(64*2, 64, dropout=dropout)
        self.output3 = SConvPass2d(64, out_channels, passthrough=False, normalize=False, dropout=None)
        self.up4 = SConvUp2d(64, 32, dropout=dropout)
        self.pass4 = SConvPass2d(32*2, 32, dropout=dropout)
        self.output4 = SConvPass2d(32, out_channels, passthrough=False, normalize=False, dropout=None)
        self.up5 = SConvUp2d(32, 16, dropout=dropout)
        self.pass5 = SConvPass2d(16*2, 16, dropout=dropout)
        self.output5 = SConvPass2d(16, out_channels, passthrough=False, normalize=False, dropout=None)
            
    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        a0 = self.pass0(x)
        d1 = self.down1(a0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5)
        p1 = self.pass1(torch.cat((u1, d4), dim=1))
        u2 = self.up2(p1)
        p2 = self.pass2(torch.cat((u2, d3), dim=1))
        o2 = self.output2(p2)
        u3 = self.up3(p2)
        p3 = self.pass3(torch.cat((u3, d2), dim=1))
        o3 = self.output3(p3)
        u4 = self.up4(p3)
        p4 = self.pass4(torch.cat((u4, d1), dim=1))
        o4 = self.output4(p4)
        u5 = self.up5(p4)
        p5 = self.pass5(torch.cat((u5, a0), dim=1))
        o5 = self.output5(p5)
        
        return o5, o4, o3, o2
    
class SUNET2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.35, tr_dropout=0.35):
        super().__init__()
        self.pass0 = SConvPass2d(in_channels, 16, normalize=False, dropout=dropout)
        self.down1 = SConvDown2d(16, 32, dropout=dropout)
        self.down2 = SConvDown2d(32, 64, dropout=dropout)
        self.down3 = SConvDown2d(64, 128, dropout=dropout)
        self.down4 = SConvDown2d(128, 256, dropout=dropout)
        self.down5 = SConvDown2d(256, 320, dropout=dropout)

        self.up1 = SConvUp2d(320, 256, dropout=dropout)
        self.pass1 = SConvPass2d(256*2, 256, dropout=dropout)
        self.up2 = SConvUp2d(256, 128, dropout=dropout)
        self.pass2 = SConvPass2d(128*2, 128, dropout=dropout)
        self.output2 = SConvPass2d(128, 4, passthrough=False, normalize=False, dropout=None)
        self.up3 = SConvUp2d(128, 64, dropout=dropout)
        self.pass3 = SConvPass2d(64*2, 64, dropout=dropout)
        self.output3 = SConvPass2d(64, 4, passthrough=False, normalize=False, dropout=None)
        self.up4 = SConvUp2d(64, 32, dropout=dropout)
        self.pass4 = SConvPass2d(32*2, 32, dropout=dropout)
        self.output4 = SConvPass2d(32, 4, passthrough=False, normalize=False, dropout=None)
        self.up5 = SConvUp2d(32, 16, dropout=dropout)
        self.pass5 = SConvPass2d(16*2, 16, dropout=dropout)
        self.output5 = SConvPass2d(16, out_channels, passthrough=False, normalize=False, dropout=None)
            
    def forward(self, x):

        a0 = self.pass0(x)
        d1 = self.down1(a0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5)
        p1 = self.pass1(torch.cat((u1, d4), dim=1))
        u2 = self.up2(p1)
        p2 = self.pass2(torch.cat((u2, d3), dim=1))
        o2 = self.output2(p2)
        u3 = self.up3(p2)
        p3 = self.pass3(torch.cat((u3, d2), dim=1))
        o3 = self.output3(p3)
        u4 = self.up4(p3)
        p4 = self.pass4(torch.cat((u4, d1), dim=1))
        o4 = self.output4(p4)
        u5 = self.up5(p4)
        p5 = self.pass5(torch.cat((u5, a0), dim=1))
        o5 = self.output5(p5)
        
        return o5

        
class CVITUNETUP2DCHUNK(nn.Module):
    def __init__(self, in_channels=4*3, out_channels=1, dropout=0.35, tr_dropout=0.35):
        super().__init__()
        self.convp0 = ConvPass2d(in_channels, 256, normalize=False, dropout=dropout)
        self.convp1 = ConvPass2d(256, 16, dropout=dropout)
        self.down1 = UD2d(16, 32, dropout=dropout) 
        self.down2 = UD2d(32, 64, dropout=dropout)
        self.down3 = UD2d(64, 128, dropout=dropout)
        self.down4 = UD2d(128, 256, dropout=dropout)
        self.down5 = UD2d(256, 320, dropout=dropout)
        self.up1 = UU2d(320, 256, dropout=dropout)
        self.up2 = UU2d(256, 128, dropout=dropout)
        self.up3 = UU2d(128, 64, dropout=dropout)
        self.up4 = UU2d(64, 32, dropout=dropout)
        self.up5 = UU2d(32, 16, dropout=dropout)
        self.conv1 =ConvUp2d(16, 8, dropout=dropout)
        self.up6 = UU2d(16, 8, dropout=dropout)
        self.final = ConvPass2d(8, out_channels, dropout=dropout)
        self.pool=nn.MaxPool2d(2)
        self.vit = CrossViTCLS2D(
            image_size = 320,
            channels=16,
            num_classes = 256,
            depth = 4,               
            sm_dim = 192,      
            sm_patch_size = 32,   
            sm_enc_depth = 3,    
            sm_enc_heads = 4,      
            sm_enc_mlp_dim = 1024,  
            lg_dim = 384,          
            lg_patch_size = 16,   
            lg_enc_depth = 2,      
            lg_enc_heads = 2,       
            lg_enc_mlp_dim = 1024,  
            cross_attn_depth = 2,  
            cross_attn_heads = 8,    
            dropout = tr_dropout,
            emb_dropout = tr_dropout
        )
        self.attnmapper = nn.Sequential(
            Rearrange('b c x y -> b x y c'),
            nn.Linear(8, 256),
            Rearrange('b x y c -> b c x y'),
            nn.Dropout(tr_dropout)
            )
        self.attnmapper0 = nn.Sequential(
            Rearrange('b c x y -> b x y c'),
            nn.Linear(8, 320),
            Rearrange('b x y c -> b c x y'),
            nn.Dropout(tr_dropout)
            )                
    def forward(self, x):
        a00 = self.convp0(x)
        a0 = self.convp1(a00)
        d1 = self.down1(a0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        sm, lg, v1, v2 = self.vit(a0) 
        u0=torch.mul(d5, self.attnmapper0(v2))
        u1 = self.up1(u0, d4)
        u1b=torch.mul(u1, self.attnmapper(v1))
        u2 = self.up2(u1b, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        u5 = self.up5(u4, a0)
        a1 = self.conv1(a0) 
        u6 = self.up6(u5, a1)
        x=self.final(u6)
        x=self.pool(x)
        return x, sm, lg

class CVITUNETUP2DCHUNKNOVIT(nn.Module):
    def __init__(self, in_channels=4*3, out_channels=1, dropout=0.35, tr_dropout=0.35):
        super().__init__()
        self.convp0 = ConvPass2d(in_channels, 256, normalize=False, dropout=dropout)
        self.convp1 = ConvPass2d(256, 16, dropout=dropout)
        self.down1 = UD2d(16, 32, dropout=dropout) #48, 24,6
        self.down2 = UD2d(32, 64, dropout=dropout)
        self.down3 = UD2d(64, 128, dropout=dropout)
        self.down4 = UD2d(128, 256, dropout=dropout)
        self.down5 = UD2d(256, 320, dropout=dropout)
        self.up1 = UU2d(320, 256, dropout=dropout)
        self.up2 = UU2d(256, 128, dropout=dropout)
        self.up3 = UU2d(128, 64, dropout=dropout)
        self.up4 = UU2d(64, 32, dropout=dropout)
        self.up5 = UU2d(32, 16, dropout=dropout)
        self.conv1 =ConvUp2d(16, 8, dropout=dropout)
        self.up6 = UU2d(16, 8, dropout=dropout)
        self.final = ConvPass2d(8, out_channels, dropout=dropout)
        self.pool=nn.MaxPool2d(2)
  
    def forward(self, x):
        a00 = self.convp0(x)
        a0 = self.convp1(a00)
        d1 = self.down1(a0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        sm, lg = torch.randn(10), torch.randn(10)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        u5 = self.up5(u4, a0)
        a1 = self.conv1(a0) 
        u6 = self.up6(u5, a1)
        x=self.final(u6)
        x=self.pool(x)
        return x, sm, lg
