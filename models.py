import torch
import torch.nn as nn

# The RRDB block was referenced from:
# https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py

import functools

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.Hardswish()



    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out1 = self.RDB1(x)*0.2 + x
        out2 = self.RDB2(out1)*0.2 + out1
        out3 = self.RDB3(out2)*0.2 + out2
        
        return out3*0.2 + x


    

class RRDB_AutoEncoderSpatial(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=5, gc=32):
        super(RRDB_AutoEncoderSpatial, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.activation = nn.Hardswish()
        self.activation_last = nn.Hardswish()


        
        # Encoder
        self.conv_e1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_e2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_e3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        
        # Feature Enhancement
        self.RRDBs = make_layer(RRDB_block_f, nb)
        
        
        # Decoder
        self.conv_d1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_d2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_d3 = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)



    def forward(self, x):
        # Encoder
        e1 = self.conv_e1(x)
        e2 = self.activation(self.conv_e2(e1))
        e3 = self.activation(self.conv_e3(e2))
        
        #Feature Enhancement
        feat_enhancement = self.RRDBs(e3)
        
        # Decoder
        d1 = self.activation(self.conv_d1(feat_enhancement))
        d2 = self.activation(self.conv_d2(d1))
        noise = self.conv_d3(d2)
        
        return x-noise
    
    
class RRDB_AutoEncoderTemporal(nn.Module):
    def __init__(self, in_nc=9, out_nc=3, nf=64, nb=4, gc=32):
        super(RRDB_AutoEncoderTemporal, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.activation = nn.Hardswish()
        self.activation_last = nn.Hardswish()


        
        # Encoder
        self.conv_e1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_e2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_e3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        
        # Feature Enhancement
        self.RRDBs = make_layer(RRDB_block_f, nb)
        
        
        # Decoder
        self.conv_d1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_d2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_d3 = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)



    def forward(self, x_):
        x,centre = x_
        # Encoder
        e1 = self.conv_e1(x)
        e2 = self.activation(self.conv_e2(e1))
        e3 = self.activation(self.conv_e3(e2))
        
        #Feature Enhancement
        feat_enhancement = self.RRDBs(e3)
        
        # Decoder
        d1 = self.activation(self.conv_d1(feat_enhancement))
        d2 = self.activation(self.conv_d2(d1))
        noise = self.conv_d3(d2)
        
        return centre-noise