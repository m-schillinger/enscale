import torch
import torch.nn as nn
import pdb
from modules_cnn import Generator4x, Generator4xExternalNoise
from utils import make_dataloader, add_one_hot
import numpy as np

class StoLayer(nn.Module):    
    """A stochastic layer.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
    """
    def __init__(self, in_dim, out_dim, noise_dim=100, add_bn=True, out_act=None, noise_std=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.noise_std = noise_std
        
        layer = [nn.Linear(in_dim + noise_dim, out_dim)]
        if add_bn:
            layer += [nn.BatchNorm1d(out_dim)]
        self.layer = nn.Sequential(*layer)
        if out_act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
        else:
            self.out_act = None
    
    def forward(self, x):
        eps = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
        out = torch.cat([x, eps], dim=1)
        out = self.layer(out)
        if self.out_act is not None:
            out = self.out_act(out)
        return out


class StoResBlock(nn.Module):
    """A stochastic residual net block.

    Args:
        dim (int, optional): input dimension. Defaults to 100.
        hidden_dim (int, optional): hidden dimension (default to dim). Defaults to None.
        out_dim (int, optional): output dimension (default to dim). Defaults to None.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add batch normalization. Defaults to True.
        out_act (str, optional): output activation function. Defaults to None.
    """
    def __init__(self, dim=100, hidden_dim=None, out_dim=None, noise_dim=100, add_bn=True, out_act=None, noise_std=1):
        super().__init__()
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        if hidden_dim is None:
            hidden_dim = dim
        if out_dim is None:
            out_dim = dim
        self.layer1 = [nn.Linear(dim + noise_dim, hidden_dim)]
        self.add_bn = add_bn
        if add_bn:
            self.layer1.append(nn.BatchNorm1d(hidden_dim))
        self.layer1.append(nn.ReLU())
        self.layer1 = nn.Sequential(*self.layer1)
        self.layer2 = nn.Linear(hidden_dim + noise_dim, out_dim)
        if add_bn and out_act == "relu": # for intermediate blocks
            self.layer2 = nn.Sequential(*[self.layer2, nn.BatchNorm1d(out_dim)])
        if out_dim != dim:
            self.layer3 = nn.Linear(dim, out_dim)
        self.dim = dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        if out_act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid() 
        elif out_act == "tanh":
            self.out_act = nn.Tanh() 
        elif out_act == "softmax":
            self.out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
        else:
            self.out_act = None

    def forward(self, x):
        if self.noise_dim > 0:
            eps = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
            out = self.layer1(torch.cat([x, eps], dim=1))
            eps = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
            out = self.layer2(torch.cat([out, eps], dim=1))
        else:
            out = self.layer2(self.layer1(x))
        if self.out_dim != self.dim:
            out2 = self.layer3(x)
            out = out + out2
        else:
            out += x
        if self.out_act is not None:
            out = self.out_act(out)
        return out
    
class StoResBlock_ExternalNoise(nn.Module):
    """A stochastic residual net block.

    Args:
        dim (int, optional): input dimension. Defaults to 100.
        hidden_dim (int, optional): hidden dimension (default to dim). Defaults to None.
        out_dim (int, optional): output dimension (default to dim). Defaults to None.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add batch normalization. Defaults to True.
        out_act (str, optional): output activation function. Defaults to None.
    """
    def __init__(self, dim=100, hidden_dim=None, out_dim=None, noise_dim=100, add_bn=True, out_act=None, noise_std=1,
                 dropout=False, dropout_final=True):
        super().__init__()
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        self.dropout = dropout
        self.dropout_final = dropout_final
        if hidden_dim is None:
            hidden_dim = dim
        if out_dim is None:
            out_dim = dim
        if dropout:
            self.layer1 = [nn.Linear(dim, hidden_dim)]
        else:
            self.layer1 = [nn.Linear(dim + noise_dim, hidden_dim)]
        self.add_bn = add_bn
        if add_bn:
            self.layer1.append(nn.BatchNorm1d(hidden_dim))
        self.layer1.append(nn.ReLU())
        self.layer1 = nn.Sequential(*self.layer1)
        if dropout:
            self.layer2 = nn.Linear(hidden_dim, out_dim)
        else:
            self.layer2 = nn.Linear(hidden_dim + noise_dim, out_dim)
        if add_bn and out_act == "relu": # for intermediate blocks
            self.layer2 = nn.Sequential(*[self.layer2, nn.BatchNorm1d(out_dim)])
        if out_dim != dim:
            self.layer3 = nn.Linear(dim, out_dim)
        self.dim = dim
        self.out_dim = out_dim
        self.noise_dim = noise_dim
        if out_act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid() 
        elif out_act == "tanh":
            self.out_act = nn.Tanh() 
        elif out_act == "softmax":
            self.out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
        else:
            self.out_act = None

    def forward(self, x, eps = None):
        if self.noise_dim > 0:
            if not self.dropout:
                if eps is None:
                    eps1 = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
                    eps2 = torch.randn(x.size(0), self.noise_dim, device=x.device) * self.noise_std
                else:
                    assert eps.size(0) == x.size(0)
                    assert eps.size(1) == 2*self.noise_dim
                    eps1 = eps[:,0:self.noise_dim] * self.noise_std
                    eps2 = eps[:,self.noise_dim:] * self.noise_std
                
                out = self.layer1(torch.cat([x, eps1], dim=1))
                out = self.layer2(torch.cat([out, eps2], dim=1))
            
            else:
                out = self.layer1(x)
                # dropout manually
                if eps is None:
                    eps1 = torch.bernoulli(torch.full(out.size(), 0.9, device=x.device))
                    out = self.layer2(out * eps1)
                else:
                    raise NotImplementedError("Dropout with external noise not implemented yet.")
                    # eps1 = ?
                    #assert eps.size(0) == x.size(0)
                    #assert eps.size(1) == self.noise_dim
                    out = self.layer2(out * eps1)
                # Q: dropout again?
                if self.dropout_final:
                    if eps is None:
                        eps2 = torch.bernoulli(torch.full(out.size(), 0.9, device=x.device))
                        out = out * eps2
                    else:
                       pass
        else:
            out = self.layer2(self.layer1(x))
        if self.out_dim != self.dim:
            out2 = self.layer3(x)
            out = out + out2
        else:
            out += x
        if self.out_act is not None:
            out = self.out_act(out)
        return out

class RankValueLayer(nn.Module):
    def __init__(self, rank_dim=720, preproc_rk_dim=20, val_dim=5, preproc_val_dim=10):
        super().__init__()
       
        self.rank_layer = nn.Sequential(nn.Linear(rank_dim,preproc_rk_dim), nn.Sigmoid())
        self.value_layer = nn.Sequential(nn.Linear(val_dim,preproc_val_dim), nn.ReLU(inplace=True))
        self.rank_dim = rank_dim
        self.val_dim = val_dim
    
    def forward(self, x):
        x1 = self.rank_layer(x[:,0:self.rank_dim])
        x2 = self.value_layer(x[:,self.rank_dim:(self.rank_dim+self.val_dim)])
        return torch.cat([x1,x2], dim=1)

# ------ STONET MODELS FROM XINWEI -------------------------

class StoNetBase(nn.Module):
    def __init__(self, noise_dim):
        super().__init__()
        self.noise_dim = noise_dim
    
    @torch.no_grad()
    def predict(self, x, target=["mean"], sample_size=100):
        """Point prediction.

        Args:
            x (torch.Tensor): input data
            target (str or float or list, optional): quantities to predict. float refers to the quantiles. Defaults to ["mean"].
            sample_size (int, optional): sample sizes for each x. Defaults to 100.

        Returns:
            torch.Tensor or list of torch.Tensor: point predictions
                - [:,:,i] gives the i-th sample of all x.
                - [i,:,:] gives all samples of x_i.
            
        Here we do not call `sample` but directly call `forward`.
        """
        if self.noise_dim == 0:
            sample_size = 1
        samples = self.sample(x=x, sample_size=sample_size, expand_dim=True)
        if not isinstance(target, list):
            target = [target]
        results = []
        extremes = []
        for t in target:
            if t == "mean":
                results.append(samples.mean(dim=len(samples.shape) - 1))
            else:
                if t == "median":
                    t = 0.5
                assert isinstance(t, float)
                results.append(samples.quantile(t, dim=len(samples.shape) - 1))
                if min(t, 1 - t) * sample_size < 10:
                    extremes.append(t)
        
        if len(extremes) > 0:
            print("Warning: the estimate for quantiles at {} with a sample size of {} could be inaccurate. Please increase the `sample_size`.".format(extremes, sample_size))

        if len(results) == 1:
            return results[0]
        else:
            return results

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
            x_rep = x.repeat(sample_size, 1)
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
    

class StoUNet(StoNetBase):
    """Stochastic neural network. UNet shape.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        out_act (str, optional): output activation function. Defaults to None.
        resblock (bool, optional): whether to use residual blocks. Defaults to False.
    """
    def __init__(self, in_dim, out_dim, num_layer=2, hidden_dim=100, 
                 noise_dim=100, add_bn=True, out_act=None, resblock=False, noise_std=1,
                 preproc_layer=False, n_vars=5, time_dim=6, val_dim=None, rank_dim=720, preproc_dim=20,
                 layer_shrinkage=16, extra_input_dim=0, dropout=False, input_dims_for_preproc=None):
        super().__init__(noise_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.noise_std = noise_std
        if out_act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_act == "softmax":
            self.out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
        elif out_act == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = None
        
        self.num_blocks = None
        if resblock:
            if num_layer % 2 != 0:
                num_layer += 1
                print("The number of layers must be an even number for residual blocks. Changed to {}".format(str(num_layer)))
            num_blocks = num_layer // 2
            self.num_blocks = num_blocks
        self.resblock = resblock
        self.num_layer = num_layer
        
        self.preproc = preproc_layer
        self.input_dims_for_preproc = input_dims_for_preproc
        if self.preproc:
            if self.input_dims_for_preproc is not None:
                self.preproc_layers = nn.ModuleList([
                    nn.Sequential(nn.Linear(input_dims_for_preproc[i], preproc_dim), nn.ReLU(inplace=True)) for i in range(len(input_dims_for_preproc))])
                in_dim = preproc_dim * len(input_dims_for_preproc)
                self.in_dim = in_dim
            else:
                pass
                """ # remove if not needed
                self.val_dim = val_dim
                self.time_dim = time_dim
                self.rank_dim = rank_dim
                self.n_vars = n_vars
                if val_dim is None:
                    one_hot_dim = in_dim - n_vars*720 - time_dim - extra_input_dim
                    self.one_hot_dim = one_hot_dim
                    self.extra_input_dim = extra_input_dim
                    if one_hot_dim > 0 and extra_input_dim > 0:
                        self.preproc_layers =  nn.ModuleList([ 
                            nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                            nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                            nn.Sequential(nn.Linear(one_hot_dim,preproc_dim), nn.ReLU(inplace=True)),
                            nn.Sequential(nn.Linear(extra_input_dim,preproc_dim), nn.ReLU(inplace=True))])
                        in_dim = preproc_dim * (n_vars + 3)
                    elif one_hot_dim > 0:
                        self.preproc_layers =  nn.ModuleList([ 
                            nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                            nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                            nn.Sequential(nn.Linear(one_hot_dim,preproc_dim), nn.ReLU(inplace=True))])
                        in_dim = preproc_dim * (n_vars + 2)
                    elif extra_input_dim > 0:
                        self.preproc_layers =  nn.ModuleList([ 
                            nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                            nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                            nn.Sequential(nn.Linear(extra_input_dim,preproc_dim), nn.ReLU(inplace=True))])
                        in_dim = preproc_dim * (n_vars + 2)
                    else:
                        self.preproc_layers =  nn.ModuleList([ 
                            nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                            nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True))])
                        in_dim = preproc_dim * (n_vars + 1)
                    self.in_dim = in_dim
                else:
                    if extra_input_dim > 0:
                        raise NotImplementedError("Extra input dimension not implemented yet.")
                    one_hot_dim = in_dim - n_vars*720 - time_dim - 5*val_dim
                    self.one_hot_dim = one_hot_dim
                    # pdb.set_trace()
                    if one_hot_dim > 0:
                        self.preproc_layers =  nn.ModuleList([
                            RankValueLayer(rank_dim=rank_dim, val_dim=val_dim, preproc_rk_dim=preproc_dim, preproc_val_dim=preproc_dim//2) for i in range(n_vars)] + [
                            nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                            nn.Sequential(nn.Linear(one_hot_dim,preproc_dim), nn.ReLU(inplace=True)),
                        ])
                    else:
                        self.preproc_layers =  nn.ModuleList([
                            RankValueLayer(rank_dim=rank_dim, val_dim=val_dim, preproc_rk_dim=preproc_dim, preproc_val_dim=preproc_dim//2) for i in range(n_vars)] + [
                            nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                        ])
                    in_dim = preproc_dim * (n_vars + 2) + preproc_dim//2 * n_vars
                    self.in_dim = in_dim
                """
            
        if self.resblock: 
            if self.num_blocks == 1:
                if dropout:
                    raise NotImplementedError("Dropout not implemented yet.")
                self.net = StoResBlock_ExternalNoise(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                       noise_dim=noise_dim, add_bn=add_bn, out_act=out_act, noise_std=noise_std, dropout=dropout)
            else:
                if self.num_blocks > 2:
                    self.input_layer = StoResBlock_ExternalNoise(dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, 
                                               noise_dim=noise_dim, add_bn=add_bn, out_act="relu", noise_std=noise_std, dropout=dropout)
                else:
                    self.input_layer = StoResBlock_ExternalNoise(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim // layer_shrinkage, 
                                               noise_dim=noise_dim, add_bn=add_bn, out_act="relu", noise_std=noise_std, dropout=dropout)
                if self.num_blocks > 2:
                    self.small_inter_layer = nn.Sequential(*[StoResBlock_ExternalNoise(dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, 
                                                                                       out_act="relu", dropout=dropout)]*(self.num_blocks - 3))
                    self.second_to_last = StoResBlock_ExternalNoise(dim=hidden_dim, hidden_dim=hidden_dim, out_dim=out_dim // layer_shrinkage,
                                                    noise_dim=noise_dim, add_bn=add_bn, out_act="relu", noise_std=noise_std, dropout=dropout)
                    
                    layers = list(self.small_inter_layer.children()) + [self.second_to_last]
                    self.inter_layer = nn.Sequential(*layers)
                
                else:
                    self.inter_layer = nn.Sequential(nn.Identity())
                
                if layer_shrinkage > 3:
                    hd = out_dim // (layer_shrinkage // 4)
                elif layer_shrinkage > 1:
                    hd = out_dim // (layer_shrinkage // 2)
                else:
                    hd = out_dim
                self.out_layer = StoResBlock_ExternalNoise(dim=out_dim // layer_shrinkage, hidden_dim = hd, out_dim=out_dim, 
                                             noise_dim=noise_dim, add_bn=add_bn, out_act=out_act, noise_std=noise_std, dropout=dropout, dropout_final=False) # output layer with concatenated noise
        else:
            raise ValueError("Only resblock version implemented yet.")
        
    
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
            x_rep = x.repeat(sample_size, 1)
            ## samples of shape (data_size*sample_size, response_dim) such that samples[data_size*(i-1):data_size*i,:] contains one sample for each data point, for i = 1, ..., sample_size
            samples = self.forward(x=x_rep).detach()
        if not expand_dim: # or sample_size == 1:
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
            raise ValueError("Shape of x and batch size need to agree in sample_batch.")
        else:
            samples = self.sample_onebatch(x, sample_size, expand_dim)
        return samples
    
    def sample(self, x, sample_size=100, expand_dim=True):
        batch_size = x.shape[0]
        samples = self.sample_batch(x, sample_size, expand_dim, batch_size)
        return samples
            
    def forward(self, x, eps=None):
        if self.preproc:
            if self.input_dims_for_preproc is not None:
                cum_input_dims = np.concatenate([np.array([0]), np.cumsum(self.input_dims_for_preproc)])
                x = torch.cat([
                    layer(x[:, cum_input_dims[i]:cum_input_dims[i+1]]) for i, layer in enumerate(self.preproc_layers)], dim=1)
            else:
                
                pass 
                """
                # remove if not needed
                if self.val_dim is None:
                    variable_dim = self.rank_dim
                else:
                    variable_dim = self.rank_dim + self.val_dim
                # preproc layer is deterministic, so no noise
                assert x.shape[-1] == variable_dim * self.n_vars + self.time_dim + self.one_hot_dim + self.extra_input_dim
                if self.one_hot_dim > 0 and self.extra_input_dim > 0:
                    x = torch.cat([layer(x[:,i*variable_dim:(i+1)*variable_dim]) for i, layer in enumerate(self.preproc_layers[0:self.n_vars])] +
                                [self.preproc_layers[self.n_vars](x[:, (variable_dim*self.n_vars):(variable_dim*self.n_vars+self.time_dim)])] + 
                                [self.preproc_layers[self.n_vars+1](x[:, (variable_dim*self.n_vars+self.time_dim):(variable_dim*self.n_vars+self.time_dim+self.one_hot_dim)])] +
                                [self.preproc_layers[self.n_vars+2](x[:, -self.extra_input_dim:])], dim=1)
                elif self.one_hot_dim > 0:
                    x = torch.cat([layer(x[:,i*variable_dim:(i+1)*variable_dim]) for i, layer in enumerate(self.preproc_layers[0:self.n_vars])] +
                                [self.preproc_layers[self.n_vars](x[:, (variable_dim*self.n_vars):(variable_dim*self.n_vars+self.time_dim)])] + 
                                [self.preproc_layers[self.n_vars+1](x[:, (variable_dim*self.n_vars+self.time_dim):])], dim=1)
                elif self.extra_input_dim > 0:
                    x = torch.cat([layer(x[:,i*variable_dim:(i+1)*variable_dim]) for i, layer in enumerate(self.preproc_layers[0:self.n_vars])] +
                                [self.preproc_layers[self.n_vars](x[:, (variable_dim*self.n_vars):(variable_dim*self.n_vars+self.time_dim)])] + 
                                [self.preproc_layers[self.n_vars+1](x[:, self.extra_input_dim:])], dim=1)            
                else:
                    pdb.set_trace()
                    x = torch.cat([layer(x[:,i*variable_dim:(i+1)*variable_dim]) for i, layer in enumerate(self.preproc_layers[0:self.n_vars])] +
                                [self.preproc_layers[self.n_vars](x[:, (variable_dim*self.n_vars):(variable_dim*self.n_vars+self.time_dim)])], dim=1)
                """
        if eps is None: # no noise supplied, just sample noise internally
            if self.num_blocks == 1:
                return self.net(x)
            else:
                x = self.input_layer(x)
                x = self.inter_layer(x)
                return self.out_layer(x)
        else: # need to pass noise externally
            assert eps.size(0) == x.size(0)
            if self.num_blocks == 1:
                return self.net(x, eps = eps)
            else:
                assert eps.size(1) == self.num_blocks*2*self.noise_dim
                eps1 = eps[:,0:(2*self.noise_dim)]
                x = self.input_layer(x, eps1)
                n_inter_layers = len(self.inter_layer)               
                x = torch.cat([self.inter_layer[i - 2](x, eps[:,i*self.noise_dim:(i+2)*self.noise_dim]) for i in range(2, n_inter_layers * 2 + 1, 2)], dim=1)
                return self.out_layer(x, eps[:,(n_inter_layers+1)*2*self.noise_dim:])
        
         
class StoUNetNoiseEnd(StoUNet):
    """Stochastic neural network. UNet shape. Noise only at the end.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        out_act (str, optional): output activation function. Defaults to None.
        resblock (bool, optional): whether to use residual blocks. Defaults to False.
    """
    def __init__(self, in_dim, out_dim, num_layer=2, hidden_dim=100, 
                 noise_dim=100, add_bn=True, out_act=None, resblock=False, noise_std=1,
                 preproc_layer=False, n_vars=5, time_dim=6, val_dim=None, rank_dim=720, preproc_dim=20,
                 layer_shrinkage=16, extra_input_dim=0):
        super().__init__(in_dim, out_dim, num_layer=num_layer, hidden_dim=hidden_dim, 
                 noise_dim=noise_dim, add_bn=add_bn, out_act=out_act, resblock=resblock, noise_std=noise_std,
                 preproc_layer=preproc_layer, n_vars=n_vars, time_dim=time_dim, val_dim=val_dim, rank_dim=rank_dim, preproc_dim=preproc_dim,
                 layer_shrinkage=layer_shrinkage, extra_input_dim=extra_input_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.noise_std = noise_std
        if out_act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_act == "softmax":
            self.out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
        elif out_act == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = None
        
        self.num_blocks = None
        if resblock:
            if num_layer % 2 != 0:
                num_layer += 1
                print("The number of layers must be an even number for residual blocks. Changed to {}".format(str(num_layer)))
            num_blocks = num_layer // 2
            self.num_blocks = num_blocks
        self.resblock = resblock
        self.num_layer = num_layer
        
        self.preproc = preproc_layer
        if self.preproc:
            self.val_dim = val_dim
            self.time_dim = time_dim
            self.rank_dim = rank_dim
            self.n_vars = n_vars
            if val_dim is None:
                one_hot_dim = in_dim - n_vars*720 - time_dim - extra_input_dim
                self.one_hot_dim = one_hot_dim
                self.extra_input_dim = extra_input_dim
                if one_hot_dim > 0 and extra_input_dim > 0:
                    self.preproc_layers =  nn.ModuleList([ 
                        nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Linear(one_hot_dim,preproc_dim), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Linear(extra_input_dim,preproc_dim), nn.ReLU(inplace=True))])
                    in_dim = preproc_dim * (n_vars + 3)
                elif one_hot_dim > 0:
                    self.preproc_layers =  nn.ModuleList([ 
                        nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Linear(one_hot_dim,preproc_dim), nn.ReLU(inplace=True))])
                    in_dim = preproc_dim * (n_vars + 2)
                elif extra_input_dim > 0:
                    self.preproc_layers =  nn.ModuleList([ 
                        nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Linear(extra_input_dim,preproc_dim), nn.ReLU(inplace=True))])
                    in_dim = preproc_dim * (n_vars + 2)
                else:
                    self.preproc_layers =  nn.ModuleList([ 
                        nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True))])
                    in_dim = preproc_dim * (n_vars + 1)
                self.in_dim = in_dim
            else:
                if extra_input_dim > 0:
                    raise NotImplementedError("Extra input dimension not implemented yet.")
                one_hot_dim = in_dim - n_vars*720 - time_dim - 5*val_dim
                self.one_hot_dim = one_hot_dim
                # pdb.set_trace()
                if one_hot_dim > 0:
                    self.preproc_layers =  nn.ModuleList([
                        RankValueLayer(rank_dim=rank_dim, val_dim=val_dim, preproc_rk_dim=preproc_dim, preproc_val_dim=preproc_dim//2) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Linear(one_hot_dim,preproc_dim), nn.ReLU(inplace=True)),
                    ])
                else:
                    self.preproc_layers =  nn.ModuleList([
                        RankValueLayer(rank_dim=rank_dim, val_dim=val_dim, preproc_rk_dim=preproc_dim, preproc_val_dim=preproc_dim//2) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                    ])
                in_dim = preproc_dim * (n_vars + 2) + preproc_dim//2 * n_vars
                self.in_dim = in_dim
        
        if self.resblock: 
            if self.num_blocks == 1:
                self.net = StoResBlock_ExternalNoise(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                       noise_dim=0, add_bn=add_bn, out_act=out_act, noise_std=noise_std)
            else:
                if self.num_blocks > 2:
                    self.input_layer = StoResBlock_ExternalNoise(dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, 
                                               noise_dim=0, add_bn=add_bn, out_act="relu", noise_std=noise_std)
                else:
                    self.input_layer = StoResBlock_ExternalNoise(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim // layer_shrinkage, 
                                               noise_dim=0, add_bn=add_bn, out_act="relu", noise_std=noise_std)
                if self.num_blocks > 2:
                    self.small_inter_layer = nn.Sequential(*[StoResBlock_ExternalNoise(dim=hidden_dim, noise_dim=0, add_bn=add_bn, out_act="relu")]*(self.num_blocks - 3))
                    self.second_to_last = StoResBlock_ExternalNoise(dim=hidden_dim, hidden_dim=hidden_dim, out_dim=out_dim // layer_shrinkage,
                                                    noise_dim=noise_dim, add_bn=add_bn, out_act="relu", noise_std=noise_std)
                    
                    layers = list(self.small_inter_layer.children()) + [self.second_to_last]
                    self.inter_layer = nn.Sequential(*layers)
                
                else:
                    self.inter_layer = nn.Sequential(nn.Identity())
                
                if layer_shrinkage > 3:
                    hd = out_dim // (layer_shrinkage // 4)
                elif layer_shrinkage > 1:
                    hd = out_dim // (layer_shrinkage // 2)
                else:
                    hd = out_dim
                self.out_layer = StoResBlock_ExternalNoise(dim=out_dim // layer_shrinkage, hidden_dim = hd, out_dim=out_dim, 
                                             noise_dim=noise_dim, add_bn=add_bn, out_act=out_act, noise_std=noise_std) # output layer with concatenated noise
        else:
            raise ValueError("Only resblock version implemented yet.")
   
            

class StoEncNet(StoNetBase):
    """Stochastic neural network. Encoder shape (first larger layers, then decreasing size)

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        out_act (str, optional): output activation function. Defaults to None.
        resblock (bool, optional): whether to use residual blocks. Defaults to False.
    """
    def __init__(self, in_dim, out_dim, num_layer=2, hidden_dim=100, 
                 noise_dim=100, add_bn=True, out_act=None, resblock=False, noise_std=1,
                 preproc_layer=False, n_vars=5, time_dim=6, val_dim=None, rank_dim=720, preproc_dim=20,
                 layer_growth = 16, extra_input_dim=0):
        super().__init__(noise_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.noise_std = noise_std
        if out_act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_act == "softmax":
            self.out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
        elif out_act == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = None
        
        self.num_blocks = None
        if resblock:
            if num_layer % 2 != 0:
                num_layer += 1
                print("The number of layers must be an even number for residual blocks. Changed to {}".format(str(num_layer)))
            num_blocks = num_layer // 2
            self.num_blocks = num_blocks
        self.resblock = resblock
        self.num_layer = num_layer
        
        self.preproc = preproc_layer
        if self.preproc:
            self.val_dim = val_dim
            self.time_dim = time_dim
            self.rank_dim = rank_dim
            self.n_vars = n_vars
            if val_dim is None:
                one_hot_dim = in_dim - n_vars*720 - time_dim - extra_input_dim
                self.one_hot_dim = one_hot_dim
                self.extra_input_dim = extra_input_dim
                if one_hot_dim > 0 and extra_input_dim > 0:
                    self.preproc_layers =  nn.ModuleList([ 
                        nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Linear(one_hot_dim,preproc_dim), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Linear(extra_input_dim,preproc_dim), nn.ReLU(inplace=True))])
                    in_dim = preproc_dim * (n_vars + 3)
                elif one_hot_dim > 0:
                    self.preproc_layers =  nn.ModuleList([ 
                        nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Linear(one_hot_dim,preproc_dim), nn.ReLU(inplace=True))])
                    in_dim = preproc_dim * (n_vars + 2)
                elif extra_input_dim > 0:
                    self.preproc_layers =  nn.ModuleList([ 
                        nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Linear(extra_input_dim,preproc_dim), nn.ReLU(inplace=True))])
                    in_dim = preproc_dim * (n_vars + 2)
                else:
                    self.preproc_layers =  nn.ModuleList([ 
                        nn.Sequential(nn.Linear(rank_dim,preproc_dim), nn.ReLU(inplace=True)) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True))])
                    in_dim = preproc_dim * (n_vars + 1)
                self.in_dim = in_dim
            else:
                if extra_input_dim > 0:
                    raise NotImplementedError("Extra input dimension not implemented yet.")
                one_hot_dim = in_dim - n_vars*720 - time_dim - 5*val_dim
                self.one_hot_dim = one_hot_dim
                # pdb.set_trace()
                if one_hot_dim > 0:
                    self.preproc_layers =  nn.ModuleList([
                        RankValueLayer(rank_dim=rank_dim, val_dim=val_dim, preproc_rk_dim=preproc_dim, preproc_val_dim=preproc_dim//2) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                        nn.Sequential(nn.Linear(one_hot_dim,preproc_dim), nn.ReLU(inplace=True)),
                    ])
                else:
                    self.preproc_layers =  nn.ModuleList([
                        RankValueLayer(rank_dim=rank_dim, val_dim=val_dim, preproc_rk_dim=preproc_dim, preproc_val_dim=preproc_dim//2) for i in range(n_vars)] + [
                        nn.Sequential(nn.Linear(time_dim,preproc_dim), nn.ReLU(inplace=True)),
                    ])
                in_dim = preproc_dim * (n_vars + 2) + preproc_dim//2 * n_vars
                self.in_dim = in_dim
        
        if self.resblock: 
            if self.num_blocks == 1:
                self.net = StoResBlock_ExternalNoise(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                       noise_dim=noise_dim, add_bn=add_bn, out_act=out_act, noise_std=noise_std)
            else:
                if self.num_blocks > 2:
                    self.input_layer = StoResBlock_ExternalNoise(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim * layer_growth, 
                                               noise_dim=noise_dim, add_bn=add_bn, out_act="relu", noise_std=noise_std)
                else:
                    self.input_layer = StoResBlock_ExternalNoise(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim * layer_growth, 
                                               noise_dim=noise_dim, add_bn=add_bn, out_act="relu", noise_std=noise_std)
                if self.num_blocks > 2:
                    self.small_inter_layer = nn.Sequential(*[StoResBlock_ExternalNoise(dim=out_dim * layer_growth, noise_dim=noise_dim, add_bn=add_bn, out_act="relu")]*(self.num_blocks - 3))
                    self.second_to_last = StoResBlock_ExternalNoise(dim=out_dim * layer_growth, hidden_dim=out_dim * layer_growth, out_dim=out_dim * layer_growth,
                                                    noise_dim=noise_dim, add_bn=add_bn, out_act="relu", noise_std=noise_std)
                    
                    layers = list(self.small_inter_layer.children()) + [self.second_to_last]
                    self.inter_layer = nn.Sequential(*layers)
                
                else:
                    self.inter_layer = nn.Sequential(nn.Identity())
                
                if layer_growth > 3:
                    hd = out_dim * (layer_growth // 4)
                elif layer_growth > 1:
                    hd = out_dim * (layer_growth // 2)
                else:
                    hd = out_dim
                self.out_layer = StoResBlock_ExternalNoise(dim=out_dim * layer_growth, hidden_dim = hd, out_dim=out_dim, 
                                             noise_dim=noise_dim, add_bn=add_bn, out_act=out_act, noise_std=noise_std) # output layer with concatinated noise
        else:
            raise ValueError("Only resblock version implemented yet.")
        
    
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
            x_rep = x.repeat(sample_size, 1)
            ## samples of shape (data_size*sample_size, response_dim) such that samples[data_size*(i-1):data_size*i,:] contains one sample for each data point, for i = 1, ..., sample_size
            samples = self.forward(x=x_rep).detach()
        if not expand_dim: # or sample_size == 1:
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
            raise ValueError("Shape of x and batch size need to agree in sample_batch.")
        else:
            samples = self.sample_onebatch(x, sample_size, expand_dim)
        return samples
    
    def sample(self, x, sample_size=100, expand_dim=True):
        batch_size = x.shape[0]
        samples = self.sample_batch(x, sample_size, expand_dim, batch_size)
        return samples
        
    def forward(self, x, eps=None):
        if self.preproc:
            if self.val_dim is None:
                variable_dim = self.rank_dim
            else:
                variable_dim = self.rank_dim + self.val_dim
            # preproc layer is deterministic, so no noise
            assert x.shape[-1] == variable_dim * self.n_vars + self.time_dim + self.one_hot_dim + self.extra_input_dim
            if self.one_hot_dim > 0 and self.extra_input_dim > 0:
                x = torch.cat([layer(x[:,i*variable_dim:(i+1)*variable_dim]) for i, layer in enumerate(self.preproc_layers[0:self.n_vars])] +
                            [self.preproc_layers[self.n_vars](x[:, (variable_dim*self.n_vars):(variable_dim*self.n_vars+self.time_dim)])] + 
                            [self.preproc_layers[self.n_vars+1](x[:, (variable_dim*self.n_vars+self.time_dim):(variable_dim*self.n_vars+self.time_dim+self.one_hot_dim)])] +
                            [self.preproc_layers[self.n_vars+2](x[:, -self.extra_input_dim:])], dim=1)
            elif self.one_hot_dim > 0:
                x = torch.cat([layer(x[:,i*variable_dim:(i+1)*variable_dim]) for i, layer in enumerate(self.preproc_layers[0:self.n_vars])] +
                            [self.preproc_layers[self.n_vars](x[:, (variable_dim*self.n_vars):(variable_dim*self.n_vars+self.time_dim)])] + 
                            [self.preproc_layers[self.n_vars+1](x[:, (variable_dim*self.n_vars+self.time_dim):])], dim=1)
            elif self.extra_input_dim > 0:
                x = torch.cat([layer(x[:,i*variable_dim:(i+1)*variable_dim]) for i, layer in enumerate(self.preproc_layers[0:self.n_vars])] +
                            [self.preproc_layers[self.n_vars](x[:, (variable_dim*self.n_vars):(variable_dim*self.n_vars+self.time_dim)])] + 
                            [self.preproc_layers[self.n_vars+1](x[:, self.extra_input_dim:])], dim=1)            
            else:
                x = torch.cat([layer(x[:,i*variable_dim:(i+1)*variable_dim]) for i, layer in enumerate(self.preproc_layers[0:self.n_vars])] +
                            [self.preproc_layers[self.n_vars](x[:, (variable_dim*self.n_vars):(variable_dim*self.n_vars+self.time_dim)])], dim=1)
        if eps is None: # no noise supplied, just sample noise internally
            if self.num_blocks == 1:
                return self.net(x)
            else:
                x = self.input_layer(x)
                x = self.inter_layer(x)
                return self.out_layer(x)
        else: # need to pass noise externally
            assert eps.size(0) == x.size(0)
            if self.num_blocks == 1:
                return self.net(x, eps = eps)
            else:
                assert eps.size(1) == self.num_blocks*2*self.noise_dim
                eps1 = eps[:,0:(2*self.noise_dim)]
                x = self.input_layer(x, eps1)
                n_inter_layers = len(self.inter_layer)               
                x = torch.cat([self.inter_layer[i - 2](x, eps[:,i*self.noise_dim:(i+2)*self.noise_dim]) for i in range(2, n_inter_layers * 2 + 1, 2)], dim=1)
                return self.out_layer(x, eps[:,(n_inter_layers+1)*2*self.noise_dim:])
    
    ####
    
    def sample_temporal(self, x, sample_size=10, expand_dim=True, start_xc = None):
        if expand_dim:
            return torch.stack([self.forward_temporal(x, eps=None, start_xc = start_xc) for k in range(sample_size)], dim = -1)
        else:
            return torch.cat([self.forward_temporal(x, eps=None, start_xc = start_xc) for k in range(sample_size)], dim = -1)
    
    def forward_temporal(self, x, eps=None, start_xc = None):
        if self.preproc:
            raise NotImplementedError("Preprocessing not implemented for forward_temporal.")
        if eps is None: # no noise supplied, just sample noise internally
            time_series = torch.empty((x.size(0), start_xc.size(0)), device=x.device)
            for i in range(len(x)):
                if i == 0:
                    time_series[i] = self.forward(torch.cat([x[i].unsqueeze(0), start_xc.unsqueeze(0)], dim=1))[0] # forward with BS = 1, dim (1, spatial_dim)
                else:
                    time_series[i] = self.forward(torch.cat([x[i].unsqueeze(0), time_series[i-1].unsqueeze(0)], dim=1))[0]
            return time_series
        else:
            raise NotImplementedError("External noise not implemented for forward_temporal.")
    
class StoNet(StoNetBase):
    """Stochastic neural network.

    Args:
        in_dim (int): input dimension 
        out_dim (int): output dimension
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        noise_dim (int, optional): noise dimension. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        out_act (str, optional): output activation function. Defaults to None.
        resblock (bool, optional): whether to use residual blocks. Defaults to False.
    """
    def __init__(self, in_dim, out_dim, num_layer=2, hidden_dim=100, 
                 noise_dim=100, add_bn=True, out_act=None, resblock=False, 
                 noise_all_layer=True, out_bias=True):
        super().__init__(noise_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.add_bn = add_bn
        self.noise_all_layer = noise_all_layer
        self.out_bias = out_bias
        if out_act == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif out_act == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_act == "softmax":
            self.out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
        elif out_act == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = None
        
        self.num_blocks = None
        if resblock:
            if num_layer % 2 != 0:
                num_layer += 1
                print("The number of layers must be an even number for residual blocks. Changed to {}".format(str(num_layer)))
            num_blocks = num_layer // 2
            self.num_blocks = num_blocks
        self.resblock = resblock
        self.num_layer = num_layer
        
        if self.resblock: 
            if self.num_blocks == 1:
                self.net = StoResBlock(dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                       noise_dim=noise_dim, add_bn=add_bn, out_act=out_act)
            else:
                self.input_layer = StoResBlock(dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim, 
                                               noise_dim=noise_dim, add_bn=add_bn, out_act="relu")
                if not noise_all_layer:
                    noise_dim = 0
                self.inter_layer = nn.Sequential(*[StoResBlock(dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, out_act="relu")]*(self.num_blocks - 2))
                self.out_layer = StoResBlock(dim=hidden_dim, hidden_dim=hidden_dim, out_dim=out_dim, 
                                             noise_dim=noise_dim, add_bn=add_bn, out_act=out_act) # output layer with concatinated noise
        else:
            self.input_layer = StoLayer(in_dim=in_dim, out_dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, out_act="relu")
            if not noise_all_layer:
                noise_dim = 0
            self.inter_layer = nn.Sequential(*[StoLayer(in_dim=hidden_dim, out_dim=hidden_dim, noise_dim=noise_dim, add_bn=add_bn, out_act="relu")]*(num_layer - 2))
            # self.out_layer = StoLayer(in_dim=hidden_dim, out_dim=out_dim, noise_dim=noise_dim, add_bn=False, out_act=out_act) # output layer with concatinated noise
            self.out_layer = nn.Linear(hidden_dim, out_dim, bias=out_bias)
            if self.out_act is not None:
                self.out_layer = nn.Sequential(*[self.out_layer, self.out_act])
            
    def forward(self, x):
        if self.num_blocks == 1:
            return self.net(x)
        else:
            return self.out_layer(self.inter_layer(self.input_layer(x)))

class DPAmodel(nn.Module):
    def __init__(self, data_dim=2, latent_dim=10, out_dim=None, condition_dim=None,
                 num_layer=3, num_layer_enc=None, hidden_dim=500, noise_dim=None, 
                 dist_enc="deterministic", dist_dec="deterministic", resblock=True,
                 encoder_k=False, bn_enc=False, bn_dec=False, out_act=None, 
                 linear=False, lin_dec=True, lin_bias=True):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        if out_dim is None:
            out_dim = data_dim
        self.out_dim = out_dim
        self.condition_dim = condition_dim
        self.num_layer = num_layer
        if num_layer_enc is None:
            num_layer_enc = num_layer
        self.num_layer_enc = num_layer_enc
        self.hidden_dim = hidden_dim
        noise_dim_enc = 0 if dist_enc == "deterministic" else noise_dim
        noise_dim_dec = 0 if dist_dec == "deterministic" else noise_dim
        self.noise_dim = noise_dim
        self.noise_dim_enc = noise_dim_enc
        self.noise_dim_dec = noise_dim_dec
        self.dist_enc = dist_enc
        self.dist_dec = dist_dec
        self.out_act = out_act
        self.linear = linear
        self.lin_dec = lin_dec
        self.encoder_k = encoder_k
        
        if not linear:
            self.encoder = StoNet(data_dim, latent_dim, num_layer_enc, hidden_dim, noise_dim_enc, bn_enc, resblock=resblock)
            if condition_dim is not None: # conditional decoder
                latent_dim = latent_dim + condition_dim
            self.decoder = StoNet(latent_dim, out_dim, num_layer, hidden_dim, noise_dim_dec, bn_dec, out_act, resblock)     
        else:
            self.encoder = nn.Linear(data_dim, latent_dim, bias=lin_bias)
            if lin_dec:
                if noise_dim_dec == 0:
                    self.decoder = nn.Linear(latent_dim, out_dim, bias=lin_bias)
                    if out_act == "relu":
                        self.decoder = nn.Sequential(*[self.decoder, nn.ReLU(inplace=True)])
                    elif out_act == "sigmoid":
                        self.decoder = nn.Sequential(*[self.decoder, nn.Sigmoid()])
                else:
                    self.decoder = StoLayer(latent_dim, out_dim, noise_dim_dec, out_act=out_act)
            else:
                self.decoder = StoNet(latent_dim, out_dim, num_layer, hidden_dim, noise_dim_dec, bn_dec, out_act, resblock)
        
        if self.encoder_k:
            self.k_embed_layer = nn.Linear(self.latent_dim, self.data_dim*2)
    
    def get_k_embedding(self, k, x=None):
        k_emb = torch.ones(1, self.latent_dim)
        k_emb[:, k:].zero_()
        if x is not None:
            k_emb = k_emb.to(x.device)
            gamma, beta = self.k_embed_layer(k_emb).chunk(2, dim=1)
            k_emb = gamma * x + beta
        return k_emb
    
    def encode(self, x, k=None, mean=True, gen_sample_size=100, in_training=False):
        if k is None:
            k = self.latent_dim
        if self.encoder_k:
            x = self.get_k_embedding(k, x)
        if in_training:
            return self.encoder(x)
        if self.dist_enc == "deterministic":
            gen_sample_size = 1
        if mean:
            z = self.encoder.predict(x, sample_size=gen_sample_size)
        else:
            z = self.encoder.sample(x, sample_size=gen_sample_size)
            if gen_sample_size == 1:
                z = z.squeeze(len(z.shape) - 1)
        return z[:, :k]
        
    def decode(self, z, c=None, mean=True, gen_sample_size=100):
        if z.size(1) != self.latent_dim:
            z_ = torch.randn((z.size(0), self.latent_dim - z.size(1)), device=z.device)
            z = torch.cat([z, z_], dim=1)
        if c is not None:
            z = torch.cat([z, c], dim=1)
        if self.dist_enc == "deterministic":
            gen_sample_size = 1
        if mean:
            x = self.decoder.predict(z, sample_size=gen_sample_size)
        else:
            x = self.decoder.sample(z, sample_size=gen_sample_size)
        return x
    
    @torch.no_grad()
    def reconstruct_onebatch(self, x, c=None, k=None, mean=False, gen_sample_size=100):
        if gen_sample_size > 1 and self.dist_enc == "deterministic" and self.dist_dec == "deterministic":
            print("The model is deterministic. Consider setting `gen_sample_size` to 1.")
        if k is None:
            k = self.latent_dim
        if gen_sample_size == 1:#self.dist_enc == "deterministic" and self.dist_dec == "deterministic" or 
            return self.forward(x, c, k).detach()
        x_rep = x.repeat(gen_sample_size, *[1]*(len(x.shape) - 1))
        samples = self.forward(x_rep, c, k).detach()
        del x_rep
        expand_dim = len(samples.shape)
        samples = samples.unsqueeze(expand_dim)
        samples = list(torch.split(samples, x.size(0)))
        samples = torch.cat(samples, dim=expand_dim)
        if mean:
            mean_recon = samples.mean(dim=len(samples.shape) - 1)
            return mean_recon
        else:
            return samples
    
    def reconstruct_batch(self, x, c=None, k=None, mean=False, gen_sample_size=100, batch_size=None):
        if batch_size is not None and batch_size < x.shape[0]:
            test_loader = make_dataloader(x, c, batch_size=batch_size, shuffle=False)
            results = []
            for x_batch in test_loader:
                if c is None:
                    x_batch = x_batch[0]
                    c_batch = None
                else:
                    x_batch, c_batch = x_batch
                results.append(self.reconstruct_onebatch(x_batch, c_batch, k, mean, gen_sample_size))
            results = torch.cat(results, dim=0)
        else:
            results = self.reconstruct_onebatch(x, c, k, mean, gen_sample_size)
        return results
        
    def reconstruct(self, x, c=None, k=None, mean=False, gen_sample_size=100, verbose=True):
        batch_size = x.shape[0]
        while True:
            try:
                results = self.reconstruct_batch(x, c, k, mean, gen_sample_size, batch_size)
                break
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size = batch_size // 2
                    if verbose:
                        print("Out of memory; reduce the batch size to {}".format(batch_size))
        return results
        
    def forward(self, x, c=None, k=None, gen_sample_size=None, return_latent=False, device=None, double=False):
        if k is None:
            k = self.latent_dim
        if self.encoder_k:
            x = self.get_k_embedding(k, x)
        if double:
            z = self.encode(x, in_training=True)
            z1 = z.clone()
            if return_latent:
                z_ = z.clone()
            z[:, k:].normal_(0, 1)
            x1 = self.decoder(z)
            z1[:, k:].normal_(0, 1)
            x2 = self.decoder(z1)
            if return_latent:
                return x1, x2, z_
            else:
                return x1, x2
        else:
            if x is not None and k > 0:
                z = self.encode(x, in_training=True)
                if return_latent:
                    z_ = z.clone()
                z[:, k:].normal_(0, 1)
            else:
                if return_latent:
                    z_ = self.encode(x, in_training=True)
                if gen_sample_size is None:
                    gen_sample_size = x.size(0)
                if device is None:
                    device = x.device if x is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                z = torch.randn((gen_sample_size, self.latent_dim), device=device)
            if self.condition_dim is not None and c is not None:
                z = torch.cat([z, c], dim=1)
            x = self.decoder(z)
            if return_latent:
                return x, z_
            else:
                return x
            
# ----- TESTING MULTIPLE COARSE MODELS FOR EACH ONE HOT VALUE ------------

"""
class MultipleStoUNetWrapper(StoNetBase):
    def __init__(self, num_models, sto_unet_params, one_hot_dim=7):
        super(MultipleStoUNetWrapper, self).__init__(noise_dim=sto_unet_params["noise_dim"])
        self.one_hot_dim = one_hot_dim
        self.models = nn.ModuleList([StoUNet(**sto_unet_params) for _ in range(num_models)])

    def forward(self, x):
        one_hot = x[:, -self.one_hot_dim:]
        x_vals = x[:, :-self.one_hot_dim]
        splits = one_hot.nonzero(as_tuple=True)[1]  # get the indices of the non-zero elements in one_hot

        # split the batch according to the values in one_hot
        batches = [x_vals[splits == i] for i in range(one_hot.size(1))]

        # pass each split to the corresponding model and keep track of the original indices
        outputs = [(splits == i, self.models[i](batches[i])) for i in range(one_hot.size(1)) if batches[i].size(0) > 0]

        # sort the outputs by the original indices
        outputs.sort(key=lambda x: x[0][0])

        # concatenate the outputs along the batch dimension
        return torch.cat([output for _, output in outputs], dim=0)
"""
class MultipleStoUNetWrapper(StoNetBase):
    def __init__(self, num_models, sto_unet_params, one_hot_dim=7):
        super(MultipleStoUNetWrapper, self).__init__(noise_dim=sto_unet_params["noise_dim"])
        self.models = nn.ModuleList([StoUNet(**sto_unet_params) for _ in range(num_models)])
        self.one_hot_dim = one_hot_dim

    def forward(self, x, model_index):
        x_vals = x[:, :-self.one_hot_dim]
        return self.models[model_index](x_vals)
    
    def sample(self, x, model_index, sample_size=5, expand_dim=True):
        x_vals = x[:, :-self.one_hot_dim]
        return self.models[model_index].sample(x_vals, sample_size=sample_size, expand_dim=expand_dim)
    
    def predict(self, x, model_index, sample_size=5):
        x_vals = x[:, :-self.one_hot_dim]
        return self.models[model_index].predict(x_vals, sample_size=sample_size)
    
    
class MultivariateStoUNetWrapper(StoNetBase):
    def __init__(self, num_models, sto_unet_params, expand_variables=True, split_input=True):
        super(MultivariateStoUNetWrapper, self).__init__(noise_dim=sto_unet_params["noise_dim"])
        self.models = nn.ModuleList([StoUNet(**sto_unet_params) for i in range(num_models)])
        self.num_models = num_models
        self.expand_variables = expand_variables
        self.split_input = split_input
        
    def forward(self, x, eps=None):
        if self.split_input:
            if len(x.shape) == 2:
                dim_per_model = x.size(1) // self.num_models
                res_per_model = [self.models[i](x[:, i*dim_per_model:(i+1)*dim_per_model], eps=eps) for i in range(self.num_models)]
            elif len(x.shape) == 3:
                res_per_model = [self.models[i](x[:, i, :], eps=eps) for i in range(self.num_models)]
        else:
            res_per_model = [self.models[i](x, eps=eps) for i in range(self.num_models)]
        
        if self.expand_variables:
            return torch.stack(res_per_model, dim=1)
        else:
            return torch.cat(res_per_model, dim=1)
    
    def sample(self, x, sample_size=5, expand_dim=True):
        if self.split_input:
            if len(x.shape) == 2:
                dim_per_model = x.size(1) // self.num_models
                res_per_model = [self.models[i].sample(x[:, i*dim_per_model:(i+1)*dim_per_model], sample_size=sample_size, expand_dim=expand_dim) for i in range(self.num_models)]
            elif len(x.shape) == 3:
                res_per_model = [self.models[i].sample(x[:, i, :], sample_size=sample_size, expand_dim=expand_dim) for i in range(self.num_models)]
        else:
            res_per_model = [self.models[i].sample(x, sample_size=sample_size, expand_dim=expand_dim) for i in range(self.num_models)]
        if self.expand_variables:
            return torch.stack(res_per_model, dim=1)
        else:
            return torch.cat(res_per_model, dim=1)
    

class MeanResidualWrapper(StoNetBase):
    def __init__(self, mean_model, residual_model):
        super(MeanResidualWrapper, self).__init__(noise_dim=residual_model.noise_dim)
        self.residual_model = residual_model
        self.mean_model = mean_model
        
    def forward(self, x, eps=None):
        mean = self.mean_model(x)
        residual = self.residual_model(x, eps=eps)
        return mean + residual
    
    def sample(self, x, sample_size=5, expand_dim=True):
        mean = self.mean_model(x).unsqueeze(-1).tile((1,1,sample_size))
        residual = self.residual_model.sample(x, sample_size=sample_size, expand_dim=expand_dim)
        return mean + residual
    
# ---- JOINT MODEL FOR GCM TO COARSE AND SUPER-RESOLUTION ---------------

class GCMCoarseRCMModel(nn.Module):
    def __init__(self, in_dim, interm_dim, out_dim, num_layer, hidden_dim, noise_dim,
                 add_bn=False, out_act=None, resblock=True, noise_std=1.0, preproc_layer=False, n_vars=5, time_dim=6, val_dim=None, rank_dim=720, preproc_dim=20):
        super(GCMCoarseRCMModel, self).__init__()
        self.coarse_model = StoUNet(in_dim, interm_dim, num_layer - 2, hidden_dim // 10, noise_dim,
                            add_bn=add_bn, out_act=out_act, resblock=resblock, noise_std=noise_std,
                            preproc_layer=preproc_layer, n_vars=n_vars, time_dim=time_dim, val_dim=val_dim, 
                            rank_dim=rank_dim, preproc_dim=preproc_dim)
        self.super_model = StoUNet(interm_dim, out_dim, num_layer, hidden_dim, noise_dim,
                            add_bn=add_bn, out_act=None, resblock=resblock, noise_std=noise_std,
                            preproc_layer=False, n_vars=1, time_dim=6, 
                            val_dim=None, rank_dim=720, preproc_dim=20)
        
    def forward(self, x, return_interm = False, eps=None):
        x_rcmc = self.coarse_model(x)
        y = self.super_model(x_rcmc, eps=eps)
        if return_interm:
            return y, x_rcmc
        else:
            return y
    
    def sample(self, x, sample_size=5):
        x_rcmc = self.coarse_model.sample(x, sample_size = sample_size)
        y = torch.stack([self.super_model(x_rcmc[:, :, i]) for i in range(sample_size)], dim = -1)
        return y
    
# ---- BENCHMARK: LINEAR MODEL --------------------------------

class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        
    def forward(self, x):
        return self.linear(x)
    
    def sample(self, x, sample_size=5):
        return self.forward(x).unsqueeze(-1).expand(-1, -1, sample_size)
    
# --- HELPER FOR RANK VALUE ------------------------------------
        
class RankValModel(nn.Module):
    def __init__(self, rank_model, val_model):
        super(RankValModel, self).__init__()
        self.rank_model = rank_model
        self.val_model = val_model
            
    def forward(self, x):
        rank = self.rank_model(x)
        val = self.val_model(x)
        return torch.cat([rank, val], dim = -1)
    
    def sample(self, x, sample_size = 5):
        rank = self.rank_model.sample(x, sample_size = sample_size)
        val = self.val_model.sample(x, sample_size = sample_size)
        return torch.cat([rank, val], dim = -2)
    
# ------ HELPER FOR MULTIVARIATE SEQ. --------------------

class MultipleStoUNetWrapper_v2(StoNetBase):
    def __init__(self, num_models, sto_unet_params, one_hot_dim=7):
        super(MultipleStoUNetWrapper, self).__init__(noise_dim=sto_unet_params["noise_dim"])
        self.models = nn.ModuleList([StoUNet(**sto_unet_params) for _ in range(num_models)])

    def forward(self, x, model_index):
        return self.models[model_index](x)
    
    def sample(self, x, model_index, sample_size=5, expand_dim=True):
        return self.models[model_index].sample(x, sample_size=sample_size, expand_dim=expand_dim)
    
    def predict(self, x, model_index, sample_size=5):
        return self.models[model_index].predict(x, sample_size=sample_size)


class CoarseSuperWrapper(nn.Module):
    def __init__(self, coarse_model, super_model,
                 noise_dim_c = 0, noise_dim_f = -1, vars_as_channels=False, n_vars=1,
                 one_hot_in_super=False, one_hot_dim=7):
        super().__init__()
        self.coarse_model = coarse_model
        self.super_model = super_model
        self.noise_dim_c = noise_dim_c
        self.noise_dim_f = noise_dim_f # noise dim f = -1 means we just pass the entire noise to the super model
        self.vars_as_channels = vars_as_channels
        self.n_vars = n_vars
        self.one_hot_in_super = one_hot_in_super
        self.one_hot_dim = one_hot_dim
        
    def forward(self, x, return_interm = False, eps=None, x_onehot=None):
        # def add_one_hot(xc, one_hot_in_super=False, conv=False, x=None, one_hot_dim=0):

        if self.noise_dim_c == 0 or eps is None:
            # either no noise needed or no noise provided for coarse model
            x_rcmc = self.coarse_model(x)
        else:
            eps1 = eps[:, 0:self.noise_dim_c]
            x_rcmc = self.coarse_model(x, eps=eps1)
        
        if self.noise_dim_f == -1 or eps is None:
            # case eps is None: no noise provided, also won't pass anything on
            # case noise_dim_f is None: just pass the entire noise to the super model
            eps2 = eps
        else:
            assert eps.size(1) == self.noise_dim_c + self.noise_dim_f
            eps2 = eps[:, self.noise_dim_c:]
            
        if self.vars_as_channels:
            x_rcmc = x_rcmc.view(x_rcmc.size(0), self.n_vars, -1)
        
        if self.one_hot_in_super and x_onehot is not None:
            x_rcmc = add_one_hot(x_rcmc, one_hot_in_super=True, conv=self.vars_as_channels, x=x_onehot, one_hot_dim=self.one_hot_dim)
        
        y = self.super_model(x_rcmc, eps=eps2)
        
        if return_interm:
            return y, x_rcmc
        else:
            return y
    
    def sample(self, x, sample_size=5, x_onehot=None):
        if self.vars_as_channels:
            x_rcmc = self.coarse_model.sample(x, sample_size = sample_size).view(x.size(0), self.n_vars, -1, sample_size)
            if self.one_hot_in_super and x_onehot is not None:
                x_rcmc = torch.stack([
                    add_one_hot(x_rcmc[:, :, :, i], one_hot_in_super=True, conv=True, x=x_onehot, one_hot_dim=self.one_hot_dim) for i in range(sample_size)
                ], dim = -1)
                
            # if super model is just a single StoUNet or Generator, directly apply it
            if isinstance(self.super_model, (StoNet, StoUNet, Generator4x, Generator4xExternalNoise)):
                y = torch.stack([self.super_model(x_rcmc[:, :, :, i]) for i in range(sample_size)], dim = -1)
            # if super model is again a wrapper, need to pass one-hot again to add one-hot to super_model.super_model
            elif isinstance(self.super_model, CoarseSuperWrapper):
                y = torch.stack([self.super_model(x_rcmc[:, :, :, i], x_onehot=x_onehot
                                              ) for i in range(sample_size)], dim = -1) 
            else:
                raise ValueError("Unknown super model type: {}".format(type(self.super_model)))
        else:
            x_rcmc = self.coarse_model.sample(x, sample_size = sample_size)
            if self.one_hot_in_super and x_onehot is not None:
                x_rcmc = torch.stack([
                    add_one_hot(x_rcmc[:, :, i], one_hot_in_super=True, conv=False, x=x_onehot, one_hot_dim=self.one_hot_dim) for i in range(sample_size)
                ], dim = -1)
                
            if isinstance(self.super_model, (StoNet, StoUNet, Generator4x, Generator4xExternalNoise)):
                y = torch.stack([self.super_model(x_rcmc[:, :, i]) for i in range(sample_size)], dim = -1)
            elif isinstance(self.super_model, CoarseSuperWrapper):
                y = torch.stack([self.super_model(x_rcmc[:, :, i], x_onehot=x_onehot
                                              ) for i in range(sample_size)], dim = -1)
        return y
    
    def sample_temporal(self, x, sample_size=10, expand_dim=True, start_xc = None, x_onehot=None):
        if expand_dim:
            return torch.stack([self.forward_temporal(x, eps=None, start_xc = start_xc, x_onehot=x_onehot) for k in range(sample_size)], dim = -1)
        else:
            return torch.cat([self.forward_temporal(x, eps=None, start_xc = start_xc, x_onehot=x_onehot) for k in range(sample_size)], dim = -1)
    
    def forward_temporal(self, x, eps=None, start_xc=None, x_onehot=None):
        if eps is None: # no noise supplied, just sample noise internally
            time_series = torch.empty((x.size(0), start_xc.size(0)), device=x.device)
            for i in range(len(x)):
                if i == 0:
                    time_series[i] = self.coarse_model(torch.cat([x[i].unsqueeze(0), start_xc.unsqueeze(0)], dim=1))[0] # forward with BS = 1, dim (1, spatial_dim)
                else:
                    time_series[i] = self.coarse_model(torch.cat([x[i].unsqueeze(0), time_series[i-1].unsqueeze(0)], dim=1))[0]
                    
            if self.one_hot_in_super and x_onehot is not None:
                time_series = add_one_hot(time_series, one_hot_in_super=True, conv=self.vars_as_channels, x=x_onehot, one_hot_dim=self.one_hot_dim)
            return self.super_model(time_series, x_onehot=x_onehot)
        else:
            raise NotImplementedError("External noise not implemented for forward_temporal.")



class ModelSpec:
    def __init__(self, model, vars_as_channels=False, use_one_hot=False, noise_dim=None, one_hot_option="concat"):
        self.model = model
        self.vars_as_channels = vars_as_channels
        self.use_one_hot = use_one_hot
        self.noise_dim = noise_dim  # None = no noise, -1 = pass all remaining noise
        self.one_hot_option = one_hot_option

class HierarchicalWrapper(nn.Module):
    def __init__(self, model_specs, n_vars=1, one_hot_dim=0):
        super().__init__()
        self.model_specs = model_specs  # list of ModelSpec
        self.n_vars = n_vars
        self.one_hot_dim = one_hot_dim
        
    def _apply_remaining_models(self, out, specs, eps=None, x_onehot=None, cls_ids=None, start_idx=1, return_intermediates=False):
        """Pass the output through models[start_idx:], shared by forward and forward_temporal."""
        noise_cursor = 0
        intermediates = []
        for spec in specs[start_idx:]:
            model = spec.model

            # Noise extraction
            if spec.noise_dim is None or eps is None:
                eps_input = None
            elif spec.noise_dim == -1:
                eps_input = eps[:, noise_cursor:] if eps is not None else None
                noise_cursor = eps.shape[1]
            else:
                eps_input = eps[:, noise_cursor:noise_cursor + spec.noise_dim]
                noise_cursor += spec.noise_dim

            # Input reshaping
            if spec.vars_as_channels:
                out = out.view(out.size(0), self.n_vars, -1)

            # One-hot
            if spec.use_one_hot and x_onehot is not None and spec.one_hot_option == "concat":
                out = add_one_hot(out, one_hot_in_super=True, conv=spec.vars_as_channels,
                                  x=x_onehot, one_hot_dim=self.one_hot_dim)

            # Forward pass
            if spec.one_hot_option == "concat" or not spec.use_one_hot:
                out = model(out, eps=eps_input)
            elif spec.one_hot_option == "argument" and spec.use_one_hot and cls_ids is not None:
                out = model(out, cls_ids=cls_ids, eps=eps_input)

        if return_intermediates:
            intermediates.append(out)

        if return_intermediates:
            return out, intermediates
        return out

    # def forward(self, x, eps=None, x_onehot=None, cls_ids=None, return_intermediates=False):
    #     out = x
    #     intermediates = []
    #     noise_cursor = 0

    #     for spec in self.model_specs:
    #         model = spec.model
    #         # Noise extraction
    #         if spec.noise_dim is None or eps is None:
    #             eps_input = None
    #         elif spec.noise_dim == -1:
    #             eps_input = eps[:, noise_cursor:] if eps is not None else None
    #             noise_cursor = eps.shape[1]
    #         else:
    #             eps_input = eps[:, noise_cursor:noise_cursor + spec.noise_dim]
    #             noise_cursor += spec.noise_dim

    #         # Input reshaping
    #         if spec.vars_as_channels:
    #             out = out.view(out.size(0), self.n_vars, -1)

    #         # One-hot concatenation
    #         if spec.use_one_hot and x_onehot is not None and spec.one_hot_option == "concat":
    #             out = add_one_hot(out, one_hot_in_super=True, conv=spec.vars_as_channels,
    #                               x=x_onehot, one_hot_dim=self.one_hot_dim)

    #         # Forward pass
    #         if spec.one_hot_option == "concat" or not spec.use_one_hot:
    #             if eps_input is not None:
    #                 out = model(out, eps=eps_input)
    #             else:
    #                 out = model(out)
    #         elif spec.one_hot_option == "argument" and spec.use_one_hot and cls_ids is not None:
    #             out = model(out, cls_ids=cls_ids)

    #         if return_intermediates:
    #             intermediates.append(out)

    #     if return_intermediates:
    #         return out, intermediates
    #     return out


    # def sample(self, x, sample_size=5, x_onehot=None):
    #     samples = []
    #     for i in range(sample_size):
    #         out = x
    #         for spec in self.model_specs:
    #             model = spec.model
                
    #             if spec.vars_as_channels:
    #                 out = out.view(out.size(0), self.n_vars, -1)
    #             if spec.use_one_hot and x_onehot is not None:
    #                 out = add_one_hot(out, one_hot_in_super=True, conv=spec.vars_as_channels,
    #                                   x=x_onehot, one_hot_dim=self.one_hot_dim)
    #             out = model.sample(out, sample_size=1)                
    #             out = out.squeeze(-1) # if out.dim() > 3 else out
                
    #         samples.append(out)

    #     return torch.stack(samples, dim=-1)

    def forward(self, x, eps=None, x_onehot=None, cls_ids=None, return_intermediates=False):
        # Just reuse the helper for all models
        return self._apply_remaining_models(x, self.model_specs, eps=eps, x_onehot=x_onehot,
                                            cls_ids=cls_ids, start_idx=0, return_intermediates=return_intermediates)


    def forward_temporal(self, x, start_xc, x_onehot=None, cls_ids=None):
        """
        Temporal generation for the first model in model_specs.
        After that, apply the remaining models as in forward().
        """
        # First model is the temporal one
        first_spec = self.model_specs[0]
        first_model = first_spec.model

        T = x.size(0)
        batch_size = x.size(1)
        time_series = torch.empty((T, batch_size, start_xc.size(-1)), device=x.device)

        for t in range(T):
            if t == 0:
                inp = torch.cat([x[t].unsqueeze(0), start_xc.unsqueeze(0)], dim=1)
            else:
                inp = torch.cat([x[t].unsqueeze(0), time_series[t-1].unsqueeze(0)], dim=1)
            time_series[t] = first_model(inp)[0]

        # Now apply the remaining models
        return self._apply_remaining_models(x, self.model_specs, eps=None, x_onehot=x_onehot,
                                            cls_ids=cls_ids, start_idx=1, return_intermediates=False)


    def sample(self, x, sample_size=10, expand_dim=True, **kwargs):
        """Naive sampling: just stack forward passes."""
        samples = [self.forward(x, **kwargs) for _ in range(sample_size)]
        if expand_dim:
            return torch.stack(samples, dim=-1)
        else:
            return torch.cat(samples, dim=-1)

    def sample_temporal(self, x, sample_size=10, expand_dim=True, start_xc=None, **kwargs):
        """Naive temporal sampling: just stack forward_temporal passes."""
        outputs = [self.forward_temporal(x, start_xc=start_xc, **kwargs) for _ in range(sample_size)]
        if expand_dim:
            return torch.stack(outputs, dim=-1)
        else:
            return torch.cat(outputs, dim=-1)

class SequentialWrapper(nn.Module):
    def __init__(self, models_per_var):
        super().__init__()
        self.models_per_var = models_per_var
        #self.vars_as_channels = vars_as_channels
        self.n_vars = len(models_per_var)
        self.variables = models_per_var.keys()
        
    def forward(self, x):
        y_per_var = {}
        for i, var in enumerate(self.variables):
            if i > 0:
                y_prev_vars = torch.cat([y_per_var[v] for j, v in enumerate(self.variables) if j < i], dim=1)
                x_var = torch.cat([x, y_prev_vars], dim=1)
            else:
                x_var = x
            y_per_var[var] = self.models_per_var[var](x_var)
        y = torch.cat([y_per_var[v] for v in self.variables], dim=1)
        return y
    
    def sample(self, x, sample_size=5):
        y_per_var = {}
        for i, var in enumerate(self.variables):
            if i > 0:
                y_prev_vars = torch.cat([y_per_var[v] for j, v in enumerate(self.variables) if j < i], dim=1) # shape [BS, i*spatial_dim_per_var, sample_size]
                y_per_var[var] = torch.stack([
                    self.models_per_var[var](torch.cat([x, y_prev_vars[:, :, k]], dim=1)) for k in range(sample_size)
                ], dim=2) # shape [BS, spatial_dim_per_var, sample_size]      
            else:
                y_per_var[var] = self.models_per_var[var].sample(x, sample_size=sample_size)
                
        y = torch.cat([y_per_var[v] for v in self.variables], dim=1)
        return y    
    
    def sample_temporal(self, x, sample_size=10, expand_dim=True, start_xc = None):
        pass    
    
    def forward_temporal(self, x, eps=None, start_xc=None):
        pass

class MLPConvWrapper(nn.Module):
    def __init__(self, mlp_model, conv_model):
        super().__init__()
        self.mlp_model = mlp_model
        self.conv_model = conv_model
    
    def forward(self, x):
        x_mlp = self.mlp_model(x)
        x_conv = self.conv_model(x_mlp)
        return x_conv
    
    def sample(self, x, sample_size=5):
        x_mlp = self.mlp_model.sample(x, sample_size=sample_size)
        x_conv = torch.stack([self.conv_model(x_mlp[:, :, i]) for i in range(sample_size)], dim = -1)
        return x_conv
