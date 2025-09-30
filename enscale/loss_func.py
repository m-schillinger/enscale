import torch
from torch.linalg import vector_norm
import torch.nn.functional as F
import torch.linalg as LA
from utils import vectorize, extract_random_patch
import pdb
import numpy as np

def energy_loss_two_sample(x0, x, xp, x0p=None, beta=1, verbose=False, agg=True, patch_size=None):
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
    
    if patch_size is not None:
        img_size = int(np.sqrt(x0.shape[1]))        
        top = np.random.randint(0, img_size + patch_size)
        left = np.random.randint(0, img_size + patch_size)
        x0 = extract_random_patch(x0, top=top, left=left, patch_size=patch_size)
        x = extract_random_patch(x, top=top, left=left, patch_size=patch_size)
        xp = extract_random_patch(xp, top=top, left=left, patch_size=patch_size)
        
        #x0 = x0.view(-1, img_size, img_size)
        #x = x.view(-1, img_size, img_size)
        #xp = xp.view(-1, img_size, img_size)
        # select patches
        #top = np.random.randint(0, img_size - patch_size)
        #left = np.random.randint(0, img_size - patch_size)

        # n = img_size
        # unnorm_prob = np.concatenate((1/np.cumsum(np.arange(1, n//4 + 2)), np.flip(1/np.cumsum(np.arange(1, n//4 + 1)))))
        # norm_prob = unnorm_prob / unnorm_prob.sum()
        # top = np.random.choice(np.arange(0, n // 2 + 1), p=norm_prob)
        # left = np.random.choice(np.arange(0, n // 2 + 1), p=norm_prob)        
        
    
    if x0.shape[1] == 1:
        s1 = torch.abs(x - x0).pow(beta) / 2 + torch.abs(xp - x0).pow(beta) / 2
        s2 = torch.abs(x - xp).pow(beta)
        if agg:
            s1 = s1.mean()
            s2 = s2.mean()
        loss = s1 - s2/2
    if x0p is None:
        s1 = (vector_norm(x - x0, 2, dim=1) + EPS).pow(beta) / 2 + (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta) / 2
        s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta) 
        if agg:
            s1 = s1.mean()
            s2 = s2.mean()
        loss = s1 - s2/2
    else:
        x0p = vectorize(x0p)
        s1 = ((vector_norm(x - x0, 2, dim=1) + EPS).pow(beta) + (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta) + 
              (vector_norm(x - x0p, 2, dim=1) + EPS).pow(beta) + (vector_norm(xp - x0p, 2, dim=1) + EPS).pow(beta)) / 4
        s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta) 
        s3 = (vector_norm(x0 - x0p, 2, dim=1) + EPS).pow(beta) 
        if agg:
            s1 = s1.mean()
            s2 = s2.mean()
            s3 = s3.mean()
        loss = s1 - s2/2 - s3/2
    if verbose:
        if agg:
            return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
        else:
            return loss, s1, s2
    else:
        return loss
    
    """
    old
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
    """
    
def energy_loss_2step(x0, x, xp, x0_c, xc, xc_p, xcond, xoncdp, beta=1, verbose=False):
    # pdb.set_trace()
    loss, s1, s2 = energy_loss_two_sample(x0, x, xp, verbose=True, beta=beta)
    lossc, s1c, s2c = energy_loss_two_sample(x0_c, xc, xc_p, verbose=True, beta=beta)
    loss_cond, s1_cond, s2_cond = energy_loss_two_sample(x0, xcond, xoncdp, verbose=True, beta=beta)
    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1), lossc.reshape(1), s1c.reshape(1), s2c.reshape(1), loss_cond.reshape(1), s1_cond.reshape(1), s2_cond.reshape(1)], dim=0)
    else:
        return loss + lossc
     
def energy_loss_coarse_wrapper(x_coarse0, x_coarse, x_coarse_p, x_coarse0_t, x_coarse_t, x_coarse_p_t, beta=1, verbose=False):
    # pdb.set_trace()
    loss, s1, s2 = energy_loss_two_sample(x_coarse0, x_coarse, x_coarse_p, verbose=True, beta=beta)
    losst, s1t, s2t = energy_loss_two_sample(x_coarse0_t, x_coarse_t, x_coarse_p_t, verbose=True, beta=beta)
    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1), losst.reshape(1), s1t.reshape(1), s2t.reshape(1)], dim=0)
    else:
        return loss + losst
    
def energy_loss_coarse_wrapper_summed(x_coarse0, x_coarse, x_coarse_p, x_coarse0_t, x_coarse_t, x_coarse_p_t, beta=1, verbose=False,
                                      norm_loss_loc=False, norm_loss_batch=False, p_norm_loss_loc=2, p_norm_loss_batch=2, norm_loss_per_var = False, 
                                      variables = None, transformed_loss_per_var=True):
    # verbose for norm_loss_per_var not yet implemented
    loss, s1, s2 = energy_loss_two_sample(x_coarse0, x_coarse, x_coarse_p, verbose=True, beta=beta)
    
    y = x_coarse0
    gen1 = x_coarse
    gen2 = x_coarse_p
    if norm_loss_loc:
        for j in range(len(p_norm_loss_loc)):
            lossnp, s1np, s2np = energy_loss_two_sample(torch.norm(F.relu(y), p=p_norm_loss_loc[j], dim=1), 
                                                        torch.norm(F.relu(gen1), p=p_norm_loss_loc[j], dim=1),
                                                        torch.norm(F.relu(gen2), p=p_norm_loss_loc[j], dim=1), verbose=True)
            lossnn, s1nn, s2nn = energy_loss_two_sample(torch.norm(F.relu(-y), p=p_norm_loss_loc[j], dim=1), 
                                                        torch.norm(F.relu(-gen1), p=p_norm_loss_loc[j], dim=1),
                                                        torch.norm(F.relu(-gen2), p=p_norm_loss_loc[j], dim=1), verbose=True)
            loss = loss + lossnp + lossnn
            s1 = s1 + s1np + s1nn
            s2 = s2 + s2np + s2nn
    if norm_loss_batch:
        for j in range(len(p_norm_loss_batch)):
            lossrp, s1rp, s2rp = energy_loss_two_sample(torch.norm(F.relu(y), p=p_norm_loss_batch[j], dim=0), 
                                                        torch.norm(F.relu(gen1), p=p_norm_loss_batch[j], dim=0),
                                                        torch.norm(F.relu(gen2), p=p_norm_loss_batch[j], dim=0), verbose=True)
            lossrn, s1rn, s2rn = energy_loss_two_sample(torch.norm(F.relu(-y), p=p_norm_loss_batch[j], dim=0), 
                                                        torch.norm(F.relu(-gen1), p=p_norm_loss_batch[j], dim=0),
                                                        torch.norm(F.relu(-gen2), p=p_norm_loss_batch[j], dim=0), verbose=True)
            loss = loss + lossrp + lossrn
            s1 = s1 + s1rp + s1rn
            s2 = s2 + s2rp + s2rn
    
    if norm_loss_per_var:
        # norm_loss_multivariate_summed(x0, x, xp, p_norm_loss_list, beta_norm_loss=1, type = "loc", agg_norm_loss="mean", n_vars = 4):
        lossnp, lossnn = norm_loss_multivariate_summed(x_coarse0, x_coarse, x_coarse_p, p_norm_loss_loc, beta_norm_loss=1, type = "loc", agg_norm_loss="mean", n_vars = len(variables))
        lossrp, lossrn = norm_loss_multivariate_summed(x_coarse0, x_coarse, x_coarse_p, p_norm_loss_batch, beta_norm_loss=1, type = "batch", agg_norm_loss="mean", n_vars = len(variables))
        loss = loss + lossnp + lossnn + lossrp + lossrn
        
    # losst, s1t, s2t = energy_loss_two_sample(x_coarse0_t, x_coarse_t, x_coarse_p_t, verbose=True, beta=beta)
    if transformed_loss_per_var:
        for i in range(x_coarse0_t.shape[1]):
            if i == 0:
                losst, s1t, s2t = energy_loss_two_sample(x_coarse0_t[:, i, :], x_coarse_t[:, i, :], x_coarse_p_t[:, i, :], verbose=True, beta=beta)
            else:
                loss2, s12, s22 = energy_loss_two_sample(x_coarse0_t[:, i, :], x_coarse_t[:, i, :], x_coarse_p_t[:, i, :], verbose=True, beta=beta)
                losst = losst + loss2
                s1t = s1t + s12
                s2t = s2t + s22 
    else:  
        losst, s1t, s2t = energy_loss_two_sample(x_coarse0_t, x_coarse_t, x_coarse_p_t, verbose=True, beta=beta)
    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1), losst.reshape(1), s1t.reshape(1), s2t.reshape(1)], dim=0)
    else:
        return loss + losst
     
def energy_loss_rk_val_wrapper(x0, x, xp, beta=1, verbose=False, log_odds_transform=False, sep_mean_std=False, lambda_mean_std=1):
    x0_rk = x0[:,:(128*128)]
    x0_val = x0[:,(128*128):]
    x_rk = x[:,:(128*128)]
    x_val = x[:,(128*128):]
    xp_rk = xp[:,:(128*128)]
    xp_val = xp[:,(128*128):]
    if log_odds_transform:
        # re-normalise s.t. between (1/x0_rk.shape[0], 1 - 1/x0_rk.shape[0])
        x0_rk = x0_rk * (x0_rk.shape[0] - 2) / (x0_rk.shape[0] - 1) + 1 / x0_rk.shape[0]
        x_rk = x_rk * (x_rk.shape[0] - 2) / (x_rk.shape[0] - 1) + 1 / x_rk.shape[0]
        xp_rk = xp_rk * (xp_rk.shape[0] - 2) / (xp_rk.shape[0] - 1) + 1 / xp_rk.shape[0]
        x0_rk = torch.log(x0_rk/(1-x0_rk))
        x_rk = torch.log(x_rk/(1-x_rk))
        xp_rk = torch.log(xp_rk/(1-xp_rk))
    if verbose:
        loss_rk, s1_rk, s2_rk = energy_loss_two_sample(x0_rk, x_rk, xp_rk, verbose=True, beta=beta)
        if sep_mean_std:
            loss_val_mean, s1_val_mean, s2_val_mean = energy_loss_two_sample(torch.mean(x0_val, dim = 1, keepdim = True), x_val[:,0].unsqueeze(1), xp_val[:,0].unsqueeze(1), verbose=True, beta=beta)
            loss_val_sd, s1_val_sd, s2_val_sd = energy_loss_two_sample(torch.std(x0_val, dim = 1, keepdim = True), x_val[:,1].unsqueeze(1), xp_val[:,1].unsqueeze(1), verbose=True, beta=beta)
            loss_val_vals, s1_val_vals, s2_val_vals = energy_loss_two_sample(x0_val, x_val[:, 2:], xp_val[:, 2:], verbose=True, beta=beta)
            loss_val = lambda_mean_std* torch.log(loss_val_mean) + lambda_mean_std * torch.log(loss_val_sd) + torch.log(loss_val_vals)
            s1_val = lambda_mean_std* torch.log(s1_val_mean) + lambda_mean_std * torch.log(s1_val_sd) + torch.log(s1_val_vals)
            s2_val = lambda_mean_std* torch.log(s2_val_mean) + lambda_mean_std * torch.log(s2_val_sd) + torch.log(s2_val_vals)
        else:
            loss_val, s1_val, s2_val = energy_loss_two_sample(x0_val, x_val, xp_val, verbose=True, beta=beta)    
        return torch.cat([loss_rk.reshape(1), s1_rk.reshape(1), s2_rk.reshape(1), loss_val.reshape(1), s1_val.reshape(1), s2_val.reshape(1)], dim=0)
    else:
        loss_rk = energy_loss_two_sample(x0_rk, x_rk, xp_rk, beta=beta)
        if sep_mean_std:
            raise NotImplementedError
        loss_val = energy_loss_two_sample(x0_val, x_val, xp_val, beta=beta)
        return torch.cat([loss_rk.reshape(1), loss_val.reshape(1)], dim=0)
    
def energy_loss_multivariate_summed(x0, x, xp, beta=1, verbose=False, n_vars = 4):
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
    # n_vars = x0.shape[1]
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
    # print("norm: ", torch.linalg.vector_norm(model.linear.weight))
    # print("norm in numpy: ", np.linalg.norm(model.linear.weight.detach().cpu().numpy()))
    # print("mse torch", mse(outputs, targets))
    # print("mse numpy", mean_squared_error(torch.flatten(outputs, start_dim = 1).detach().cpu().numpy(), torch.flatten(targets, start_dim = 1).detach().cpu().numpy()))
    # weight has shape (20*36, 20*36*5), vector_norm flattens the weight
    try:
        return mse(x0, x) + torch.tensor(alpha).to("cuda") * torch.linalg.vector_norm(model.linear.weight)**2 / 20 / 36
    except:
        pdb.set_trace()
    
    # return mse(x0, x) + torch.tensor(alpha).to("cuda") * torch.linalg.vector_norm(model.linear.weight)**2 / 20 / 36

def avg_constraint(xc, gen):
    size_low = int(np.sqrt(xc.shape[-1]))
    size = int(np.sqrt(gen.shape[-1]))
    ups_factor = size / size_low
    gen_avg = torch.nn.functional.avg_pool2d(gen.view(-1, 1, size, size), int(ups_factor), int(ups_factor)).view(-1, size_low**2)
    return torch.norm(gen_avg - xc, 2, dim = 1).mean()
    
def avg_constraint_per_var(xc, gen, n_vars = 4):
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
    
# ----- penalty for autoencoder ----------------

def safe_std(x, dim=None):
    if dim is None:
        return torch.std(x) if x.numel() > 1 else torch.tensor(0.0)
    else:
        return torch.std(x, dim=dim) if x.size(dim) > 1 else torch.zeros(x.size(dim))

def group_variance_penalty(gen, label):
    penalty = vector_norm(gen.std(dim=0) - 1) # penalty for variance of all components
    penalty = penalty + vector_norm(gen.mean(dim=0)) # penalty for mean of all components

    unique_labels = torch.unique(label, dim = 0)
    for unique_label in unique_labels:
        row_wise_check = torch.all(torch.eq(label, unique_label), dim=1)
        group = gen[row_wise_check, ...]
        penalty = penalty + vector_norm(safe_std(group, dim=0))
    return penalty