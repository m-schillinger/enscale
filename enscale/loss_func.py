import torch
from torch.linalg import vector_norm
import torch.nn.functional as F
import torch.linalg as LA
from utils import vectorize, extract_random_patch
import pdb
import numpy as np

def energy_loss_two_sample(x0, x, xp, beta=1, verbose=False, agg=True, patch_size=None):
    """Loss function based on the energy score (estimated based on two samples).
    
    Args:
        x0 (torch.Tensor): iid samples from the true distribution.
        x (torch.Tensor): iid samples from the estimated distribution.
        xp (torch.Tensor): iid samples from the estimated distribution.
        beta (float): power parameter in the energy score.
        verbose (bool):  whether to return two terms of the loss.
        agg (bool): whether to aggregate the loss over the batch 
            (if yes, mean over batch elements is taken; if not, one loss per batch entry is returned).
        patch_size (int or None): if not None, extract random patches of given size for loss computation.
    
    Returns:
        loss (torch.Tensor): energy loss. (if verbose=True, also returns s1 and s2 terms of the loss, concatenated along dim=0)
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)
    
    if patch_size is not None:
        img_size = int(np.sqrt(x0.shape[1]))        
        top = np.random.randint(0, img_size + patch_size)
        left = np.random.randint(0, img_size + patch_size)
        x0 = extract_random_patch(x0, top=top, left=left, patch_size=patch_size)
        x = extract_random_patch(x, top=top, left=left, patch_size=patch_size)
        xp = extract_random_patch(xp, top=top, left=left, patch_size=patch_size)
            
    s1 = (vector_norm(x - x0, 2, dim=1) + EPS).pow(beta) / 2 + (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta) / 2
    s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta) 
    if agg:
        s1 = s1.mean()
        s2 = s2.mean()
    loss = s1 - s2/2
    if verbose:
        if agg:
            return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
        else:
            return loss, s1, s2
    else:
        return loss
    
def energy_loss_multivariate_summed(x0, x, xp, beta=1, verbose=False, n_vars = 4):
    """
    Multivariate energy loss computed as the sum of univariate energy losses.
    Args:
        x0 (torch.Tensor): iid samples from the true distribution.
        x (torch.Tensor): iid samples from the estimated distribution.
        xp (torch.Tensor): iid samples from the estimated distribution.
        beta (float): power parameter in the energy score.
        verbose (bool):  whether to return two terms of the loss.
        n_vars (int): number of variables (dimensions) in the multivariate data.
        
    Details:
        Dimensions of x0, x, xp can be either (batch_size, n_vars, dim_per_var) or (batch_size, n_vars * dim_per_var).
    
    Returns:
        loss (torch.Tensor): summed energy loss. (if verbose=True, also returns summed s1 and s2, concatenated along dim=0)
    """
    # n_vars = x0.shape[1]
    for i in range(n_vars):
        if len(x0.shape) == 3:
            x0_var = x0[:, i, :]
            x_var = x[:, i, :]
            xp_var = xp[:, i, :]
        elif len(x0.shape) == 2:
            dim_per_var = x0.size(1) // n_vars
            x0_var = x0[:, i * dim_per_var:(i + 1) * dim_per_var]
            x_var = x[:, i * dim_per_var:(i + 1) * dim_per_var]
            xp_var = xp[:, i * dim_per_var:(i + 1) * dim_per_var]
        if i == 0:
            loss, s1, s2 = energy_loss_two_sample(x0_var, x_var, xp_var, verbose=True, beta=beta)
        else:
            loss2, s12, s22 = energy_loss_two_sample(x0_var, x_var, xp_var, verbose=True, beta=beta)
            loss = loss + loss2
            s1 = s1 + s12
            s2 = s2 + s22
    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return loss

def norm_loss(y, gen1, gen2, p_norm_loss_loc, p_norm_loss_batch, beta_norm_loss=1, agg_norm_loss="mean"):
    """
    Compute the norm loss between the true and generated samples.
    Here, norm loss refers to the energy loss computed on the norms of the samples. 
    First, a norm is applied to one of the dimensions in each sample (e.g. across spatial dimension or across batch dimension), 
    then the energy loss is computed based on these norms.
    This is done for multiple p-norms specified in p_norm_loss_loc and p_norm_loss_batch.
    norm_loss_loc refers to applying the norm across the spatial dimension (i.e. per sample in the batch),
    while norm_loss_batch refers to applying the norm across the batch dimension (i.e. per spatial location).
    Norms are computed separately for the positive part (ReLU) and negative part (ReLU(-x)) of the samples.
    Args:
        y (torch.Tensor): true samples (flattened, i.e. shape (batch_size, spatial_dim)).
        gen1 (torch.Tensor): first set of generated samples (flattened, i.e. shape (batch_size, spatial_dim)).
        gen2 (torch.Tensor): second set of generated samples (flattened, i.e. shape (batch_size, spatial_dim)).
        p_norm_loss_loc (list of int): list of p values for the p-norm applied across spatial dimension.
        p_norm_loss_batch (list of int): list of p values for the p-norm applied across batch dimension.
        beta_norm_loss (float): power parameter in the energy score.
        agg_norm_loss (str): aggregation method for batch-wise norm loss ("mean" or "max"). After applying the p-norm across batch dimension in each location,
          energy loss is computed on the norms in each location, and the results are aggregated using the specified method
          (either max across locations or mean across locations).
          
    Returns:
        total_lossnp (torch.Tensor): total norm loss for positive part (ReLU) across spatial dimension.
        total_lossnn (torch.Tensor): total norm loss for negative part (ReLU(-x)) across spatial dimension.
        total_lossrp (torch.Tensor): total norm loss for positive part (ReLU) across batch dimension.
        total_lossrn (torch.Tensor): total norm loss for negative part (ReLU(-x)) across batch dimension.
    """
    if p_norm_loss_loc:
        for i in range(len(p_norm_loss_loc)):
            p_norm_loss = p_norm_loss_loc[i]
            lossnp, s1n, s2n = energy_loss_two_sample(LA.norm(F.relu(y), ord=p_norm_loss, dim=1), LA.norm(F.relu(gen1), ord=p_norm_loss, dim=1), LA.norm(F.relu(gen2), ord=p_norm_loss, dim=1), 
                                                        verbose=True, beta=beta_norm_loss, agg=True)
            lossnn, s1n, s2n = energy_loss_two_sample(LA.norm(F.relu(-y), ord=p_norm_loss, dim=1), LA.norm(F.relu(-gen1), ord=p_norm_loss, dim=1), LA.norm(F.relu(-gen2), ord=p_norm_loss, dim=1), 
                                                        verbose=True, beta=beta_norm_loss, agg=True)
            if i == 0:
                total_lossnp = lossnp
                total_lossnn = lossnn
            else:
                total_lossnp += lossnp
                total_lossnn += lossnn
    else:
        total_lossnp = torch.tensor(0)
        total_lossnn = torch.tensor(0)

    
    if agg_norm_loss == "max":
        mp = torch.nn.MaxPool1d(y.shape[-1], stride=y.shape[-1])
                                    
    if p_norm_loss_batch:
        for i in range(len(p_norm_loss_batch)):
            p_norm_loss = p_norm_loss_batch[i]

            lossrp, s1r, s2r = energy_loss_two_sample(LA.norm(F.relu(y), ord=p_norm_loss, dim=0), LA.norm(F.relu(gen1), ord=p_norm_loss, dim=0), LA.norm(F.relu(gen2), ord=p_norm_loss, dim=0), 
                                                        verbose=True, beta=beta_norm_loss, agg = False)
            lossrn, s1r, s2r = energy_loss_two_sample(LA.norm(F.relu(-y), ord=p_norm_loss, dim=0), LA.norm(F.relu(-gen1), ord=p_norm_loss, dim=0), LA.norm(F.relu(-gen2), ord=p_norm_loss, dim=0), 
                                                        verbose=True, beta=beta_norm_loss, agg = False)

            if agg_norm_loss == "mean":
                lossrp = torch.mean(lossrp)
                lossrn = torch.mean(lossrn)
            elif agg_norm_loss == "max":
                lossrp = mp(lossrp.unsqueeze(0)).squeeze()
                lossrn = mp(lossrn.unsqueeze(0)).squeeze()
            
            if i == 0:
                total_lossrp = lossrp
                total_lossrn = lossrn
            else:
                total_lossrp += lossrp
                total_lossrn += lossrn
    else:
        total_lossrp = torch.tensor(0)
        total_lossrn = torch.tensor(0)
        
    return total_lossnp, total_lossnn, total_lossrp, total_lossrn



def norm_loss_multivariate_summed(x0, x, xp, p_norm_loss_list, beta_norm_loss=1, type = "loc", agg_norm_loss="mean", n_vars = 4):
    """
    Multivariate norm loss computed as the sum of univariate norm losses.
    Compute the norm loss between the true and generated samples.
    Here, norm loss refers to the energy loss computed on the norms of the samples. 
    First, a norm is applied to one of the dimensions in each sample (e.g. across spatial dimension or across batch dimension), 
    then the energy loss is computed based on these norms.
    This is done for multiple p-norms specified in p_norm_loss_loc and p_norm_loss_batch.
    norm_loss_loc refers to applying the norm across the spatial dimension (i.e. per sample in the batch),
    while norm_loss_batch refers to applying the norm across the batch dimension (i.e. per spatial location).
    Norms are computed separately for the positive part (ReLU) and negative part (ReLU(-x)) of the samples.
    Args:
        x0 (torch.Tensor): true samples
        x (torch.Tensor): first set of generated samples
        xp (torch.Tensor): second set of generated samples
        p_norm_loss_list (list of int): list of p values for the p-norm applied.
        beta_norm_loss (float): power parameter in the energy score.
        type (str): "loc" for applying the norm across spatial dimension, "batch" for applying the norm across batch dimension.
        agg_norm_loss (str): aggregation method for batch-wise norm loss ("mean" or "max"). After applying the p-norm across batch dimension in each location,
          energy loss is computed on the norms in each location, and the results are aggregated using the specified method
          (either max across locations or mean across locations).
        n_vars (int): number of variables (dimensions) in the multivariate data.
        
    Details:
        Dimensions of x0, x, xp can be either (batch_size, n_vars, dim_per_var) or (batch_size, n_vars * dim_per_var).
    
    Returns:
        losspvals (torch.Tensor): total norm loss for positive part (ReLU) across specified dimension.
        lossnvals (torch.Tensor): total norm loss for negative part (ReLU(-x)) across specified dimension.
    
    """
    losspvals = []
    lossnvals = []
    mp = torch.nn.MaxPool1d(128*128, stride=128*128)
    for i in range(n_vars):
        if len(x0.shape) == 3:
            x0_var = x0[:, i, :]
            x_var = x[:, i, :]
            xp_var = xp[:, i, :]
        elif len(x0.shape) == 2:
            dim_per_var = x0.size(1) // n_vars
            x0_var = x0[:, i * dim_per_var:(i + 1) * dim_per_var]
            x_var = x[:, i * dim_per_var:(i + 1) * dim_per_var]
            xp_var = xp[:, i * dim_per_var:(i + 1) * dim_per_var]
        
        for j in range(len(p_norm_loss_list)):
            p_norm_loss = p_norm_loss_list[j]
            if type == "loc":
                dim = 1
            elif type == "batch":
                dim = 0
            agg = False
            lossp, s1, s2 = energy_loss_two_sample(LA.norm(F.relu(x0_var), ord=p_norm_loss, dim=dim), 
                                                   LA.norm(F.relu(x_var), ord=p_norm_loss, dim=dim), 
                                                   LA.norm(F.relu(xp_var), ord=p_norm_loss, dim=dim), 
                                                        verbose=True, beta=beta_norm_loss, agg = agg)
            lossn, s1, s2 = energy_loss_two_sample(LA.norm(F.relu(-x0_var), ord=p_norm_loss, dim=dim), 
                                                   LA.norm(F.relu(-x_var), ord=p_norm_loss, dim=dim), 
                                                   LA.norm(F.relu(-xp_var), ord=p_norm_loss, dim=dim), 
                                                        verbose=True, beta=beta_norm_loss, agg = agg)
                        
            if agg_norm_loss == "mean":
                lossp = torch.mean(lossp)
                lossn = torch.mean(lossn)
            elif agg_norm_loss == "max":
                lossp = mp(lossp.unsqueeze(0)).squeeze()
                lossn = mp(lossn.unsqueeze(0)).squeeze()
            
            losspvals.append(lossp)
            lossnvals.append(lossn)
    return sum(losspvals), sum(lossnvals)
    
    
def ridge_loss(x0, x, mse, model, alpha = 0):
    """
    Ridge loss function: MSE plus L2 penalty on model weights.
    Args:
        x0 (torch.Tensor): true samples.
        x (torch.Tensor): generated samples.
        mse (function): mean squared error function.
        model (torch.nn.Module): model with linear layer whose weights are penalized.
        alpha (float): regularization parameter for L2 penalty.
    
    Returns:
        loss (torch.Tensor): ridge loss.
    """
    # print("norm: ", torch.linalg.vector_norm(model.linear.weight))
    # print("norm in numpy: ", np.linalg.norm(model.linear.weight.detach().cpu().numpy()))
    # print("mse torch", mse(outputs, targets))
    # print("mse numpy", mean_squared_error(torch.flatten(outputs, start_dim = 1).detach().cpu().numpy(), torch.flatten(targets, start_dim = 1).detach().cpu().numpy()))
    # weight has shape (20*36, 20*36*5), vector_norm flattens the weight
    return mse(x0, x) + torch.tensor(alpha).to("cuda") * torch.linalg.vector_norm(model.linear.weight)**2 / 20 / 36
    
    # return mse(x0, x) + torch.tensor(alpha).to("cuda") * torch.linalg.vector_norm(model.linear.weight)**2 / 20 / 36

def avg_constraint(xc, gen):
    """
    Average constraint loss between low-resolution and high-resolution samples.
    Calculates difference between low-resolution samples and average-pooled high-resolution samples.
    Args:
        xc (torch.Tensor): low-resolution true samples.
        gen (torch.Tensor): generated high-resolution samples.
    
    Details:
        Assumes square images, i.e. number of pixels is a perfect square.
        Also assumes that the high-resolution images are an integer multiple of the low-resolution images in each dimension, 
        that the upsampling factor is the same in both dimensions,
        and that the grids are aligned (i.e. no shifts).
        
        xc, gen have shape (batch_size, num_pixels).
    
    Returns:
        loss (torch.Tensor): average constraint loss.
    """
    size_low = int(np.sqrt(xc.shape[-1]))
    size = int(np.sqrt(gen.shape[-1]))
    ups_factor = size / size_low
    gen_avg = torch.nn.functional.avg_pool2d(gen.view(-1, 1, size, size), int(ups_factor), int(ups_factor)).view(-1, size_low**2)
    return torch.norm(gen_avg - xc, 2, dim = 1).mean()
    
def avg_constraint_per_var(xc, gen, n_vars = 4):
    """
    Average constraint per variable, for super-resolution setup.
    Args:
        xc (torch.Tensor): low-resolution true samples.
        gen (torch.Tensor): generated high-resolution samples.
        n_vars (int): number of variables (dimensions) in the multivariate data.
        
    Returns:
        loss (torch.Tensor): summed average constraint loss across variables.
    """
    for i in range(n_vars):
        if len(xc.shape) == 3:
            xc_var = xc[:, i, :]
            gen_var = gen[:, i, :]
        elif len(xc.shape) == 2:
            dim_per_var_xc = xc.size(1) // n_vars
            dim_per_var_gen = gen.size(1) // n_vars
            xc_var = xc[:, i * dim_per_var_xc:(i + 1) * dim_per_var_xc]
            gen_var = gen[:, i * dim_per_var_gen:(i + 1) * dim_per_var_gen]
        if i == 0:
            loss = avg_constraint(xc_var, gen_var)
        else:
            loss = loss + avg_constraint(xc_var, gen_var)
    return loss
        
    
# ----- pixel-wise CRPS as a benchmark --------

def crps_pixelwise(x0, x, xp, beta=1, verbose=False):
    
    """Loss function based on the energy score (estimated based on two samples).
    Pixel-wise CRPS.
    
    Args:
        x0 (torch.Tensor): iid samples from the true distribution.
        x (torch.Tensor): iid samples from the estimated distribution.
        xp (torch.Tensor): iid samples from the estimated distribution.
        beta (float): power parameter in the energy score.
        verbose (bool):  whether to return two terms of the loss.
    
    Returns:
        loss (torch.Tensor): averaged pixel-wise loss.
    """
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)
        
    s1 = torch.abs(x - x0).pow(beta).mean() / 2 + torch.abs(xp - x0).pow(beta).mean() / 2
    s2 = torch.abs(x - xp).pow(beta).mean()
    loss = s1 - s2/2
   
    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return loss