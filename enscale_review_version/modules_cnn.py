import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_, kaiming_uniform_, orthogonal_
import pdb
from utils import make_dataloader


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        # kaiming_uniform_(m.weight)
        orthogonal_(m.weight)
        m.bias.data.fill_(0.)

def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snconvtrans2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out

class NoiseInjection(nn.Module):
    def __init__(self, channel, size=1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, size, size))

    def forward(self, image):
        one_channel = image[:,0:1,:,:]
        noise = one_channel.new(one_channel.shape).normal_(0, 1)
        # noise = image.normal_(0, 1)
        out = image + self.weight * noise
        return out
    
class NoiseInjectionExternalNoise(nn.Module):
    def __init__(self, channel, size=1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, size, size))

    def forward(self, image, noise):
        one_channel = image[:,0:1,:,:]
        # noise = one_channel.new(one_channel.shape).normal_(0, 1)
        assert noise.shape == one_channel.shape
        # noise = image.normal_(0, 1)
        out = image + self.weight * noise
        return out
    
class NoiseConcatenation(nn.Module):
    def __init__(self, channel, size=1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, size, size))
        self.num_noise_channel = channel

    def forward(self, image):
        noise = torch.randn(image.size(0), self.num_noise_channel, image.size(2), image.size(3), device=image.device)
        # noise = image.normal_(0, 1)
        out = torch.cat([image, self.weight * noise], dim=1)
        return out

class GenIniBlock(nn.Module):
    def __init__(self, z_dim, out_channels, size=1, add_noise=True):
        super().__init__()
        self.out_channels = out_channels
        self.add_noise = add_noise
        self.snlinear0 = snlinear(in_features=z_dim, out_features=out_channels * 4 * 4)
        if add_noise:
            self.noise0 = NoiseInjection(out_channels, size)

    def forward(self, z):
        act0 = self.snlinear0(z)            # n x g_conv_dim*4*4*4
        # n x 256 x 4 x 4 Compare with the original big model, we reduce the width of the first 4 layers.
        act0 = act0.view(-1, self.out_channels, 4, 4)
        if self.add_noise:
            act0 = self.noise0(act0)

        return act0

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=1, add_noise=True):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.add_noise = add_noise
        if add_noise:
            self.noise1 = NoiseInjection(out_channels, size)
            self.noise2 = NoiseInjection(out_channels, size)
        self.conv_0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(self, x,):
        x0 = x
        x = self.relu(self.bn1(x))
        x = self.upsample(x)  # upsample
        x = self.conv_1(x)
        if self.add_noise:
            x = self.noise1(x)
        x = self.relu(self.bn2(x))
        x = self.conv_2(x)
        if self.add_noise:
            x = self.noise2(x)
        x0 = self.upsample(x0)  # upsample
        x0 = self.conv_0(x0)
        out = x + x0
        return out
    
class GenBlockConcat(nn.Module):
    def __init__(self, in_channels, out_channels, size=1, add_noise=True, num_noise_channels=1):
        # output will have out_channels + num_noise_channels channels
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = snconv2d(in_channels=out_channels + num_noise_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.add_noise = add_noise
        if add_noise:
            self.noise1 = NoiseConcatenation(num_noise_channels, size)
            self.noise2 = NoiseConcatenation(num_noise_channels, size)
        self.conv_0 = snconv2d(in_channels=in_channels, out_channels=out_channels + num_noise_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels + num_noise_channels)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(self, x,):
        x0 = x
        x = self.relu(self.bn1(x))
        x = self.upsample(x)  # upsample
        x = self.conv_1(x)
        if self.add_noise:
            x = self.noise1(x)
        x = self.relu(self.bn2(x))
        x = self.conv_2(x)
        if self.add_noise:
            x = self.noise2(x)
        x0 = self.upsample(x0)  # upsample
        x0 = self.conv_0(x0)
        out = x + x0
        return out


class GenBlockExternalNoise(nn.Module):
    # within one GenBlock, size of x is doubled
    def __init__(self, in_channels, out_channels, size=1, add_noise=True):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.add_noise = add_noise
        if add_noise:
            self.noise1 = NoiseInjectionExternalNoise(out_channels, size)
            self.noise2 = NoiseInjectionExternalNoise(out_channels, size)
        self.conv_0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(self, x, noise=None):
        x0 = x
        x = self.relu(self.bn1(x))
        x = self.upsample(x)  # upsample
        x = self.conv_1(x)
        
        if noise is None and self.add_noise:
            noise = torch.randn(x.size(0), 2, x.size(2), x.size(3), device=x.device)
        elif self.add_noise:
            assert noise.shape == (x.size(0), 2, x.size(2), x.size(3))
        
        if self.add_noise:
            # noise needs shape (x.shape(0), 1, x.shape(2), x.shape(3))
            x = self.noise1(x, noise[:, 0:1, :, :])
        x = self.relu(self.bn2(x))
        x = self.conv_2(x)
        if self.add_noise:
            # noise needs shape (x.shape(0), 1, x.shape(2), x.shape(3))
            x = self.noise2(x, noise[:, 1:2, :, :])
        x0 = self.upsample(x0)  # upsample
        x0 = self.conv_0(x0)
        out = x + x0
        return out

class GeneratorBase(nn.Module):
    def __init__(self):
        super().__init__()
    
    def sample_onebatch(self, x, sample_size=100, expand_dim=True):
        """Sampling new response data (for one batch of data).

        Args:
            x (torch.Tensor): new data of predictors of shape [data_size, covariate_dim]
            sample_size (int, optional): new sample size. Defaults to 100.
            expand_dim (bool, optional): whether to expand the sample dimension. Defaults to True.

        Returns:
            torch.Tensor of shape (data_size, response_dim, sample_size) if expand_dim else (data_size*sample_size, response_dim), where response_dim could have multiple channels.
        """
        data_size = x.size(0) ## input data size
        with torch.no_grad():
            ## repeat the data for sample_size times, get a tensor [data, data, ..., data]
            if len(x.shape) == 2:
                x_rep = x.repeat(sample_size, 1)
            elif len(x.shape) == 3:
                x_rep = x.repeat(sample_size, 1, 1)
            ## samples of shape (data_size*sample_size, response_dim) such that samples[data_size*(i-1):data_size*i,:] contains one sample for each data point, for i = 1, ..., sample_size
            samples = self.forward(x=x_rep).detach()
        if not expand_dim:# or sample_size == 1:
            return samples
        else:
            expand_dim = len(samples.shape)
            samples = samples.unsqueeze(expand_dim) ## (data_size*sample_size, response_dim, 1)
            ## a list of length data_size, each element is a tensor of shape (data_size, response_dim, 1)
            samples = list(torch.split(samples, data_size)) 
            samples = torch.cat(samples, dim=expand_dim) ## (data_size, response_dim, sample_size)
            return samples
            # without expanding dimensions:
            # samples.reshape(-1, *samples.shape[1:-1])
    
    def sample_batch(self, x, sample_size=100, expand_dim=True, batch_size=None):
        """Sampling with mini-batches; only used when out-of-memory.

        Args:
            x (torch.Tensor): new data of predictors of shape [data_size, covariate_dim]
            sample_size (int, optional): new sample size. Defaults to 100.
            expand_dim (bool, optional): whether to expand the sample dimension. Defaults to True.
            batch_size (int, optional): batch size. Defaults to None.

        Returns:
            torch.Tensor of shape (data_size, response_dim, sample_size) if expand_dim else (data_size*sample_size, response_dim), where response_dim could have multiple channels.
        """
        if batch_size is not None and batch_size < x.shape[0]:
            test_loader = make_dataloader(x, batch_size=batch_size, shuffle=False)
            samples = []
            for (x_batch,) in test_loader:
                samples.append(self.sample_onebatch(x_batch, sample_size, expand_dim))
            samples = torch.cat(samples, dim=0)
        else:
            samples = self.sample_onebatch(x, sample_size, expand_dim)
        return samples
    
    def sample(self, x, sample_size=100, expand_dim=True, verbose=True):
        """Sampling that adaptively adjusts the batch size according to the GPU memory."""
        batch_size = x.shape[0]
        while True:
            try:
                samples = self.sample_batch(x, sample_size, expand_dim, batch_size)
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size = batch_size // 2
                    if verbose:
                        print("Out of memory; reduce the batch size to {}".format(batch_size))
        return samples
    
class Generator2x(GeneratorBase):
    def __init__(self, n_channels=1, image_size=32, conv_dim=32, act_func=None):
        super().__init__()
        # input size: 1 x 32 x 32
        self.n_channels = n_channels
        self.layers = nn.ModuleList([
            snconv2d(in_channels=n_channels, out_channels=conv_dim * 4, kernel_size=3, stride=1, padding=1), # 4*conv_dim x 32 x 32
            GenBlock(conv_dim * 4, conv_dim * 2, size=image_size * 2), # 2*conv_dim x 64 x 64
            snconv2d(in_channels=conv_dim * 2, out_channels=n_channels, kernel_size=3, stride=1, padding=1) # 1 x 128 x 128
        ])
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()
        else:
            self.act_func = None
        
    def forward(self, x):
        x0 = x
        if len(x.shape) == 2:
            size = int(x.shape[1] ** 0.5)
            x = x.view(-1, 1, size, size)
        if len(x.shape) == 3:
            size = int(x.shape[2] ** 0.5)
            assert x.shape[1] == self.n_channels
            x = x.view(-1, self.n_channels, size, size)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        if self.act_func is not None:
            x = self.act_func(x)
        if len(x0.shape) == 2:
            return x.view(x.shape[0], -1)
        elif len(x0.shape) == 3:
            return x.view(x.shape[0], self.n_channels, -1)


class Generator4x(GeneratorBase):
    def __init__(self, n_channels=1, image_size=32, conv_dim=32, act_func=None, one_hot_channel=False, one_hot_dim=0):
        super().__init__()
        # input size: 1 x 32 x 32
        if one_hot_channel:
            n_channels_in = n_channels + one_hot_dim
        else:
            n_channels_in = n_channels
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels
        self.layers = nn.ModuleList([
            snconv2d(in_channels=n_channels_in, out_channels=conv_dim * 4, kernel_size=3, stride=1, padding=1), # 4*conv_dim x 32 x 32
            GenBlock(conv_dim * 4, conv_dim * 2, size=image_size * 2), # 2*conv_dim x 64 x 64
            GenBlock(conv_dim * 2, conv_dim, size=image_size * 4), # conv_dim x 128 x 128
            snconv2d(in_channels=conv_dim, out_channels=n_channels, kernel_size=3, stride=1, padding=1) # 1 x 128 x 128
        ])
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()
        else:
            self.act_func = None
        
    def forward(self, x):        
        x0 = x
        if len(x.shape) == 2:
            size = int(x.shape[1] ** 0.5)
            x = x.view(-1, 1, size, size)
        if len(x.shape) == 3:
            size = int(x.shape[2] ** 0.5)
            assert x.shape[1] == self.n_channels_in
            x = x.view(-1, self.n_channels_in, size, size)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        if self.act_func is not None:
            x = self.act_func(x)
        if len(x0.shape) == 2:
            return x.view(x.shape[0], -1)
        elif len(x0.shape) == 3:
            return x.view(x.shape[0], self.n_channels_out, -1)
        
class Generator4xConcat(GeneratorBase):
    def __init__(self, n_channels=1, image_size=32, conv_dim=32, act_func=None, num_noise_channels=1,  one_hot_channel=False, one_hot_dim=0):
        super().__init__()
        # input size: 1 x 32 x 32
        if one_hot_channel:
            n_channels_in = n_channels + one_hot_dim
        else:
            n_channels_in = n_channels
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels
        
        self.layers = nn.ModuleList([
            snconv2d(in_channels=n_channels_in, out_channels=conv_dim * 4, kernel_size=3, stride=1, padding=1), # 4*conv_dim x 32 x 32
            GenBlockConcat(conv_dim * 4, conv_dim * 2, size=image_size * 2), # 2*conv_dim x 64 x 64
            GenBlockConcat(conv_dim * 2 + num_noise_channels, conv_dim, size=image_size * 4), # conv_dim x 128 x 128
            snconv2d(in_channels=conv_dim + num_noise_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1) # 1 x 128 x 128
        ])
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()
        else:
            self.act_func = None
        
    def forward(self, x):
        x0 = x
        if len(x.shape) == 2:
            size = int(x.shape[1] ** 0.5)
            x = x.view(-1, 1, size, size)
        if len(x.shape) == 3:
            size = int(x.shape[2] ** 0.5)
            assert x.shape[1] == self.n_channels_in
            x = x.view(-1, self.n_channels_in, size, size)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        if self.act_func is not None:
            x = self.act_func(x)
        if len(x0.shape) == 2:
            return x.view(x.shape[0], -1)
        elif len(x0.shape) == 3:
            return x.view(x.shape[0], self.n_channels_out, -1)
        
class Generator4xExternalNoise(GeneratorBase):
    def __init__(self, n_channels=1, image_size=32, conv_dim=32, act_func=None, one_hot_channel=False, one_hot_dim=0):
        super().__init__()
        # input size: 1 x 32 x 32
        if one_hot_channel:
            n_channels_in = n_channels + one_hot_dim
        else:
            n_channels_in = n_channels
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels
        self.layers = nn.ModuleList([
            snconv2d(in_channels=n_channels_in, out_channels=conv_dim * 4, kernel_size=3, stride=1, padding=1), # 4*conv_dim x 32 x 32
            GenBlockExternalNoise(conv_dim * 4, conv_dim * 2, size=image_size * 2), # 2*conv_dim x 64 x 64
            GenBlockExternalNoise(conv_dim * 2, conv_dim, size=image_size * 4), # conv_dim x 128 x 128
            snconv2d(in_channels=conv_dim, out_channels=n_channels, kernel_size=3, stride=1, padding=1) # 1 x 128 x 128
        ])
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()
        else:
            self.act_func = None
        
    def forward(self, x, eps=None):
        x0 = x
        if len(x.shape) == 2:
            size = int(x.shape[1] ** 0.5)
            x = x.view(-1, 1, size, size)
        if len(x.shape) == 3:
            size = int(x.shape[2] ** 0.5)
            assert x.shape[1] == self.n_channels_in
            x = x.view(-1, self.n_channels_in, size, size)
        
        if eps is None:
            noise1 = torch.randn(x.size(0), 2, x.size(2) * 2, x.size(3) * 2, device=x.device)
            noise2 = torch.randn(x.size(0), 2, x.size(2) * 4, x.size(3) * 4, device=x.device)
        else:
            # assert noise.shape == (x.size(0), 40 * x.size(2) * x.size(3))
            noise1 = eps[:, :(8*x.size(2)*x.size(3))].view(x.size(0), 2, x.size(2) * 2, x.size(3) * 2)
            noise2 = eps[:, (8*x.size(2)*x.size(3)):].view(x.size(0), 2, x.size(2) * 4, x.size(3) * 4)
        
        x = self.layers[0](x)
        x = self.layers[1](x, noise1) # needs noise of shape BS x 2 x 64 x 64
        x = self.layers[2](x, noise2) # needs noise of shape BS x 2 x 128 x 128
        x = self.layers[3](x)
        
        if self.act_func is not None:
            x = self.act_func(x)
        if len(x0.shape) == 2:
            return x.view(x.shape[0], -1)
        elif len(x0.shape) == 3:
            return x.view(x.shape[0], self.n_channels_out, -1)        

        
class Generator16x(GeneratorBase):
    def __init__(self, n_channels=1, input_size=8, conv_dim=32, act_func=None):
        super().__init__()
        # input size: 1 x 8 x 8
        self.layers = nn.ModuleList([
            snconv2d(in_channels=n_channels, out_channels=conv_dim * 16, kernel_size=3, stride=1, padding=1), # 16*conv_dim x 8 x 8
            GenBlock(conv_dim * 16, conv_dim * 8, size=input_size * 2), # 8*conv_dim x 16 x 16
            GenBlock(conv_dim * 8, conv_dim * 4, size=input_size * 4), # 4*conv_dim x 32 x 32
            GenBlock(conv_dim * 4, conv_dim * 2, size=input_size * 8), # 2*conv_dim x 64 x 64
            GenBlock(conv_dim * 2, conv_dim, size=input_size * 16), # conv_dim x 128 x 128
            snconv2d(in_channels=conv_dim, out_channels=n_channels, kernel_size=3, stride=1, padding=1) # 1 x 128 x 128
        ])
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()
        else:
            self.act_func = None
        
    def forward(self, x):
        if len(x.shape) == 2:
            size = int(x.shape[1] ** 0.5)
            x = x.view(-1, 1, size, size)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x.view(x.shape[0], -1)

class Generator(nn.Module):
    r'''SAGAN Generator

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        out_channels: number of output channels
        add_noise: whether to add noises to each conv layer
        attn: whether to add self-attention layer
    '''

    def __init__(self, latent_dim, conv_dim=32, image_size=128, out_channels=3, add_noise=True, attn=True, act_func=None):
        super().__init__()

        self.latent_dim = latent_dim
        self.conv_dim = conv_dim
        self.image_size = image_size
        self.add_noise = add_noise
        self.attn = attn

        self.block0 = GenIniBlock(latent_dim, conv_dim * 16, 4, add_noise=add_noise)
        self.block1 = GenBlock(conv_dim * 16, conv_dim * 16, size=8, add_noise=add_noise)
        self.block2 = GenBlock(conv_dim * 16, conv_dim * 8, size=16, add_noise=add_noise)
        if image_size == 64:
            self.block3 = GenBlock(conv_dim * 8, conv_dim * 4, size=32, add_noise=add_noise)
            if attn:
                self.self_attn1 = Self_Attn(conv_dim * 4)
            self.block4 = GenBlock(conv_dim * 4, conv_dim * 2, size=64, add_noise=add_noise)
            conv_dim = conv_dim * 2
        elif image_size == 128:
            self.block3 = GenBlock(conv_dim * 8, conv_dim * 4, add_noise=add_noise)
            if attn:
                self.self_attn1 = Self_Attn(conv_dim * 4)
            self.block4 = GenBlock(conv_dim * 4, conv_dim * 2, add_noise=add_noise)
            # self.self_attn2 = Self_Attn(conv_dim*2)
            self.block5 = GenBlock(conv_dim * 2, conv_dim, add_noise=add_noise)
        else: # image_size == 256 or 512
            self.block3 = GenBlock(conv_dim * 8, conv_dim * 8, add_noise=add_noise)
            self.block4 = GenBlock(conv_dim * 8, conv_dim * 4, add_noise=add_noise)
            if attn:
                self.self_attn1 = Self_Attn(conv_dim * 4)
            self.block5 = GenBlock(conv_dim * 4, conv_dim * 2, add_noise=add_noise)
            self.block6 = GenBlock(conv_dim * 2, conv_dim, add_noise=add_noise)
            if image_size == 512:
                self.block7 = GenBlock(conv_dim, conv_dim, add_noise=add_noise)

        self.bn = nn.BatchNorm2d(conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.toRGB = snconv2d(in_channels=conv_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        if act_func == 'tanh':
            self.act_func = nn.Tanh()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()
        else:
            self.act_func = None

        # Weight init
        self.apply(init_weights)

    def forward(self, z):
        out = self.block0(z)      # n x g_conv_dim*16 x 4 x 4
        out = self.block1(out)    # n x g_conv_dim*16 x 8 x 8 = n x 1024 x 8 x 8
        out = self.block2(out)    # n x g_conv_dim*8 x 16 x 16 = n x 512 x 16 x 16
        out = self.block3(out)    # n x g_conv_dim*4 x 32 x 32 = n x 256 x 32 x 32
        if self.attn:
            out = self.self_attn1(out)         # n x g_conv_dim*4 x 32 x 32
        out = self.block4(out)    # n x g_conv_dim*2 x 64 x 64 = n x 128 x 64 x 64
        if self.image_size > 64:
            out = self.block5(out)    # n x g_conv_dim  x 128 x 128 = n x 64 x 128 x 128
            if self.image_size == 256 or self.image_size == 512:
                out = self.block6(out)  # 64 x 256 x 256
                if self.image_size == 512:
                    out = self.block7(out) # 64 x 512 x 512
        out = self.bn(out)                # n x g_conv_dim  x 128 x 128
        out = self.relu(out)              # n x g_conv_dim  x 128 x 128
        out = self.toRGB(out)         # n x 3 x 128 x 128
        if self.act_func is not None:
            out = self.act_func(out)              # n x 3 x 128 x 128
        return out


class DiscOptBlock(nn.Module):
    # Compared with block, optimized_block always downsamples the spatial resolution of the input vector by a factor of 4.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        # self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x

        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.conv_0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, add_bn=False):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        # self.lrelu = nn.LeakyReLU(0.1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        
        self.add_bn = add_bn
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, downsample=True):
        x0 = x

        if self.add_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_1(x)
        if self.add_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.conv_0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, conv_dim, image_size=128, in_channels=3, out_channels=1, out_feature=False, add_bn=False):
        super().__init__()
        self.conv_dim = conv_dim
        self.image_size = image_size
        self.out_feature = out_feature

        # self.opt_block1 = DiscOptBlock(in_channels, conv_dim) # for small model
        self.fromRGB = snconv2d(in_channels, conv_dim, 1, bias=True) # 128 x 128 x 64 newly added for BIGGER MODEL

        self.block1 = DiscBlock(conv_dim, conv_dim * 2, add_bn)
        self.self_attn = Self_Attn(conv_dim*2)
        self.block2 = DiscBlock(conv_dim * 2, conv_dim * 4, add_bn)
        self.block3 = DiscBlock(conv_dim * 4, conv_dim * 8, add_bn)
        if image_size == 64:
            self.block4 = DiscBlock(conv_dim * 8, conv_dim * 16, add_bn)
            self.block5 = DiscBlock(conv_dim * 16, conv_dim * 16, add_bn) # newly added for BIGGER MODEL
        elif image_size == 128:
            self.block4 = DiscBlock(conv_dim * 8, conv_dim * 16, add_bn)
            self.block5 = DiscBlock(conv_dim * 16, conv_dim * 16, add_bn)
            # self.block6 = DiscBlock(conv_dim * 16, conv_dim * 16) # newly added for BIGGER MODEL
        else: # image_size == 256
            self.block4 = DiscBlock(conv_dim * 8, conv_dim * 8, add_bn)
            self.block5 = DiscBlock(conv_dim * 8, conv_dim * 16, add_bn)
            self.block6 = DiscBlock(conv_dim * 16, conv_dim * 16, add_bn)
            # self.block7 = DiscBlock(conv_dim * 16, conv_dim * 16) # newly added for BIGGER MODEL
        # self.final = DisFinalBlock(conv_dim * 16) # scalar
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=conv_dim*16, out_features=out_channels)

        # Weight init
        self.apply(init_weights)

    def forward(self, x, z=None):
        # n x 3 x 128 x 128
        # BIGGER MODEL
        if z is not None: # for joint D(x,z)
            b = x.shape[0]
            c = z.shape[1]
            w = x.shape[2]
            h = x.shape[3]
            z = z.view(b, c, 1, 1).expand(b, c, w, h)
            x = torch.cat([x, z], 1)
        h0 = self.fromRGB(x) # 128 x 128 x 64
        h1 = self.block1(h0)    # n x d_conv_dim*2 x 32 x 32 # 64 x 64 x 128
        h1 = self.self_attn(h1) # n x d_conv_dim*2 x 32 x 32
        h2 = self.block2(h1)    # n x d_conv_dim*4 x 16 x 16 # 32 x 32 x 256
        h3 = self.block3(h2)    # n x d_conv_dim*8 x  8 x  8 # 16 x 16 x 512
        h4 = self.block4(h3)    # n x d_conv_dim*16 x 4 x  4 # 8 x 8 x 1024
        if self.image_size == 64:
            h5 = self.block5(h4, downsample=False)
            h6 = h5
        elif self.image_size == 128:
            h5 = self.block5(h4)  # n x d_conv_dim*16 x 4 x 4 # 4 x 4 x 1024
            h6 = h5
            # h6 = self.block6(h5, downsample=False)  # n x d_conv_dim*16 x 4 x 4 # 4 x 4 x 1024
        else: # (self.image_size == 256):
            h5 = self.block5(h4)  # n x d_conv_dim*16 x 4 x 4 # 4 x 4 x 1024
            h6 = self.block6(h5)  # newly added
            h6 = self.block7(h6, downsample=False)
        h6 = self.relu(h6)              # n x d_conv_dim*16 x 4 x 4
        # h7 = self.final(h6)
        # out = h7

        # Global sum pooling
        h7 = torch.sum(h6, dim=[2,3])   # n x d_conv_dim*16
        out = torch.squeeze(self.snlinear1(h7)) # n

        # SMALLER MODEL
        # if z is not None: # for joint D(x,z)
        #     b = x.shape[0]
        #     c = z.shape[1]
        #     w = x.shape[2]
        #     h = x.shape[3]
        #     z = z.view(b, c, 1, 1).expand(b, c, w, h)
        #     x = torch.cat([x, z], 1)
        #
        # h0 = self.opt_block1(x) # n x d_conv_dim   x 64 x 64
        # h1 = self.block1(h0)    # n x d_conv_dim*2 x 32 x 32
        # # h1 = self.self_attn(h1) # n x d_conv_dim*2 x 32 x 32
        # h2 = self.block2(h1)    # n x d_conv_dim*4 x 16 x 16
        # h3 = self.block3(h2)    # n x d_conv_dim*8 x  8 x  8
        # h4 = self.block4(h3)    # n x d_conv_dim*16 x 4 x  4
        # if self.image_size == 64:
        #     h5 = h4
        # elif self.image_size == 128:
        #     h5 = self.block5(h4, downsample=False)  # n x d_conv_dim*16 x 4 x 4
        # else: # (self.image_size == 256):
        #     h5 = self.block5(h4)
        #     h5 = self.block6(h5, downsample=False)
        # h5 = self.relu(h5)              # n x d_conv_dim*16 x 4 x 4
        # # h6 = self.final(h5)
        # # out = h6
        #
        # # Global sum pooling
        # h6 = torch.sum(h5, dim=[2,3])   # n x d_conv_dim*16
        # out = torch.squeeze(self.snlinear1(h6)) # n

        if self.out_feature:
            return out, h7
        else:
            return out


class DisFinalBlock(nn.Module):
    """ Final block for the Discriminator """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv_2 = snconv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4)
        self.conv_3 = snconv2d(in_channels=in_channels, out_channels=1, kernel_size=1)

        # self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm2d(in_channels)
        # self.ln = LayerNorm()

    def forward(self, x):
        y = self.relu(self.conv_1(x))
        y = self.relu(self.conv_2(y))
        # fully connected layer with linear activation
        y = self.conv_3(y)

        return y


class SNResMLPBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Sequential(
            snlinear(channels, channels),
            nn.ReLU(inplace=True)
        )
        self.fc2 = snlinear(channels, channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.fc2(self.fc1(x))
        out += x
        return self.relu(out)


class Discriminator_MLP(nn.Module):

    def __init__(self, in_channels, out_channels, out_feature=True, num_block=3):
        super().__init__()
        self.out_feature = out_feature
        self.num_block = num_block

        self.fc1 = nn.Sequential(
            snlinear(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )
        self.block1 = SNResMLPBlock(out_channels)
        if num_block > 1:
            self.block2 = SNResMLPBlock(out_channels)
        if num_block > 2:
            self.block3 = SNResMLPBlock(out_channels)
        self.fc4 = snlinear(out_channels, 1)


        self.apply(init_weights)

    def forward(self, z):

        out = self.fc1(z)
        f = self.block1(out)
        if self.num_block > 1:
            f = self.block2(f)
        if self.num_block > 2:
            f = self.block3(f)
        out = self.fc4(f)

        if self.out_feature:
            return out, f
        else:
            return out


class DCDiscriminator(nn.Module):
    def __init__(self, conv_dim=64, image_size=64, image_channel=3):
        super().__init__()
        # Input x: 28*28*3, z: 100
        self.conv = nn.Sequential(
            nn.Conv2d(image_channel, conv_dim, 5, 2, 2),  # 14*14*64
            # nn.BatchNorm2d(conv_dim * 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(conv_dim, conv_dim * 2, 5, 2, 2),  # 7*7*128
            # nn.BatchNorm2d(conv_dim * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 5, 2, 2),  # 4*4*256
            # nn.BatchNorm2d(conv_dim * 4),
            nn.LeakyReLU(inplace=True),
            # nn.Conv2d(conv_dim * 4, conv_dim * 8, 5, 2, 2),  # 2*2*512
            # # nn.BatchNorm2d(conv_dim * 8),
            # nn.LeakyReLU(inplace=True),
        )
        # self.fc = nn.Linear(conv_dim * 8 * 2 * 2, 1)
        self.fc = nn.Linear(conv_dim * 4 * 4 * 4, 1)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)


class MNISTGenerator(nn.Module):
    def __init__(self, image_size=64, latent_dim=64, image_channel=3):
        super().__init__()

        self.init_size = image_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, image_channel, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class MNISTDiscriminator(nn.Module):
    def __init__(self, image_size=64, image_channel=3):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(image_channel, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = image_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
