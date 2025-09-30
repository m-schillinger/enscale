
import torch
import numpy as np
import pysteps
import torch.linalg as LA
from torch.linalg import vector_norm


def dist(x, y, beta = 1.0):
    # take matrix norm and sum across channel dimension
    # output will be a tensor of length x.shape[0] (batch size)
    assert x.shape == y.shape
    if len(x.shape) == 4: # got a proper tensor
        if beta == 1.0:
            return torch.sum(LA.matrix_norm(x - y), dim = 1)
        else:
            return torch.sum(torch.pow(LA.matrix_norm(x - y), beta), dim = 1)
    elif len(x.shape) == 3: # got a degenerated tensor (after applying a boolean mask)
        if beta == 1.0:
            return torch.sum(LA.norm(x - y, dim = -1), dim = 1)
        else:
            return torch.sum(torch.pow(LA.norm(x - y, dim = -1), beta), dim = 1)
    elif len(x.shape) == 2: # got a degenerated tensor (after applying a boolean mask)
        if beta == 1.0:
            return LA.norm(x - y, dim = -1)
        else:
            return torch.pow(LA.norm(x - y, dim = -1), beta)
    
def dist_batch(x, y, beta = 1.0):
    # take matrix norm and sum across channel dimension and mean over batch dimension
    # output will be a scalar
    return torch.mean(dist(x, y, beta))
    
# unconditional
def uncond_energy_score(sample1, sample2, truth1, truth2, full = False, beta = 1.0):
    # output will be a tensor of length sample1.shape[0] (batch size)
    score1 = 0.5 * (dist(sample1, truth2, beta = beta) + dist(sample2, truth1, beta = beta)) 
    score2 = 0.5 * dist(sample1, sample2, beta = beta) 
    score3 = 0.5 * dist(truth1, truth2, beta = beta)
    es = score1 - score2 - score3
    if not full:
        return es
    else:
        return es, score1, score2, score3

def uncond_energy_score_batch(sample1, sample2, truth1, truth2, full = False, beta = 1.0):
    # output will be a single scalar
    if not full:
        es = uncond_energy_score(sample1, sample2, truth1, truth2, full = False, beta = beta)
        return torch.mean(es)
    else:
        es, score1, score2, score3 = uncond_energy_score(sample1, sample2, truth1, truth2, full = True, beta = beta)
        return torch.mean(es), torch.mean(score1), torch.mean(score2), torch.mean(score3)
    
def cond_energy_score(truth, sample1, sample2, beta = 1.0, full = True):
    score1 = 0.5 * (dist(sample1, truth, beta = beta) + dist(sample2, truth, beta = beta)) 
    score2 = 0.5 * dist(sample1, sample2, beta = beta) 
    if full:
        return score1 - score2, score1, score2
    else:
        return score1 - score2
    
def cond_energy_score_batch(truth, sample1, sample2, beta = 1.0, full = True):
    if not full:
        es = cond_energy_score(truth, sample1, sample2, full = False, beta = beta)
        return torch.mean(es)
    else:
        es, score1, score2 = cond_energy_score(truth, sample1, sample2, full = True, beta = beta)
        # return torch.mean(es), torch.mean(score1), torch.mean(score2)
        return torch.cat([torch.mean((score1 - score2)).reshape(1), torch.mean(score1).reshape(1), torch.mean(score2).reshape(1)], dim=0) #score1 - score2, score1, score2
    

def vectorize(x, multichannel=False):
    """Vectorize data in any shape.

    Args:
        x (torch.Tensor): input data
        multichannel (bool, optional): whether to keep the multiple channels (in the second dimension). Defaults to False.

    Returns:
        torch.Tensor: data of shape (sample_size, dimension) or (sample_size, num_channel, dimension) if multichannel is True.
    """
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel: # one channel
            return x.reshape(x.shape[0], -1)
        else: # multi-channel
            return x.reshape(x.shape[0], x.shape[1], -1)

def energy_loss_two_sample(x0, x, xp, x0p=None, beta=1, verbose=False):
    """Loss function based on the energy score (estimated based on two samples).
    
    Args:
        x0 (torch.Tensor): iid samples from the true distribution.
        x (torch.Tensor): iid samples from the estimated distribution.
        xp (torch.Tensor): iid samples from the estimated distribution.
        beta (float): power parameter in the energy score.
        verbose (bool):  whether to return two terms of the loss.
    
    Returns:
        loss (torch.Tensor): energy loss.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)
    
    """
    # old version of logit that transforms all the data
    if logit:
        if clip_quantile is not None:
            x0 = torch.clamp(x0, clip_quantile, 1 - clip_quantile)
            x = torch.clamp(x, clip_quantile, 1 - clip_quantile)
            xp = torch.clamp(xp, clip_quantile, 1 - clip_quantile)
        x0 = torch.log(x0/(1-x0))
        x = torch.log(x/(1-x))
        xp = torch.log(xp/(1-xp))
    """
    
    if x0.shape[1] == 1:
        s1 = torch.abs(x - x0).pow(beta).mean() / 2 + torch.abs(xp - x0).pow(beta).mean() / 2
        s2 = torch.abs(x - xp).pow(beta).mean()
        loss = s1 - s2/2
    if x0p is None:
        s1 = (vector_norm(x - x0, 2, dim=1) + EPS).pow(beta).mean() / 2 + (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta).mean() / 2
        s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta).mean() 
        loss = s1 - s2/2
    else:
        x0p = vectorize(x0p)
        s1 = ((vector_norm(x - x0, 2, dim=1) + EPS).pow(beta).mean() + (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta).mean() + 
              (vector_norm(x - x0p, 2, dim=1) + EPS).pow(beta).mean() + (vector_norm(xp - x0p, 2, dim=1) + EPS).pow(beta).mean()) / 4
        s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta).mean() 
        s3 = (vector_norm(x0 - x0p, 2, dim=1) + EPS).pow(beta).mean() 
        loss = s1 - s2/2 - s3/2
    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return loss