
    
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree


class RectUpsampler(nn.Module):
    """ Module class for upsampling from a low-resolution grid to a high-resolution grid
        using learnable weights and biases.
        The upsampling is performed by interpolating each high-resolution pixel from its
        k nearest neighbors in the low-resolution grid.
        
        Args:
            grid_size_lo : int : size of the low-resolution grid (assumed square)
            grid_size_hi : int : size of the high-resolution grid (assumed square)
            n_features   : int : number of features per pixel (for EnScale: number of climate variables)
            num_classes  : int : number of classes (for class-specific weights/biases; for EnScale: classes are different GCM-RCM pairs)
            num_neighbors: int : number of nearest neighbors to use for interpolation
        
        Input: 
            Low resolution input data, flattened (BS, n_features * number_of_lowres_pixels)
            Optional vector of class IDs (BS, num_classes), class indices in [0, num_classes-1]
            
        Output:
            High resolution output data, not flattened (BS, n_features, grid_size_hi, grid_size_hi)
    """
    
    def __init__(self,
                 grid_size_lo: int,
                 grid_size_hi: int,
                 n_features: int = 4,
                 num_classes: int = 1,
                 num_neighbors: int = 9):
        super().__init__()
        self.grid_lo = grid_size_lo
        self.grid_hi = grid_size_hi
        self.n_features = n_features
        self.num_classes = num_classes
        self.k = num_neighbors
       
        # Total pixels
        self.p_lo = grid_size_lo ** 2
        self.p_hi = grid_size_hi ** 2
       
        # Precompute 2D coordinates for neighborhoods
        self.register_buffer("neighbor_indices", self._compute_2d_neighbors())

        # Learnable weights for upsampling (C, F, P_hi, k)
        self.weight_map = nn.Parameter(
            torch.zeros(num_classes, n_features, self.p_hi, self.k)
        )
       
        # Biases for low and high grids (C, F, P)
        self.bias_low = nn.Parameter(torch.zeros(num_classes, n_features, self.p_lo))
        self.bias_high = nn.Parameter(torch.zeros(num_classes, n_features, self.p_hi))
    
    def _compute_2d_neighbors(self):
        """Pre-compute k nearest neighbors in 2D grid from low-res to high-res."""
        grid_hi = self.grid_hi
        grid_lo = self.grid_lo
        k = self.k
        ratio = grid_hi / grid_lo
        assert ratio == int(ratio), "grid_hi must be divisible by grid_lo"
        ratio = int(ratio)

        # Generate properly scaled coordinates
        lo_coords = self._grid_coords(grid_lo, spacing=ratio)  # (P_lo, 2)
        hi_coords = self._grid_coords(grid_hi, spacing=1.0)    # (P_hi, 2)

        # Use cKDTree to find nearest k low-res neighbors for each high-res pixel
        idx = cKDTree(lo_coords).query(hi_coords, k=k)[1]  # (P_hi, k)
        return torch.tensor(idx, dtype=torch.long)
   
    def _grid_coords(self, size, spacing=1.0):
        xv, yv = torch.meshgrid(
            torch.arange(size) * spacing,
            torch.arange(size) * spacing,
            indexing='ij'
        )
        coords = torch.stack([xv.flatten(), yv.flatten()], dim=1)
        return coords.numpy()


    def forward(self, x: torch.Tensor, cls_ids: torch.Tensor = None) -> torch.Tensor:
        """
        x         : (BS, n_features * p_lo)
        cls_ids  : (BS, num_classes)
        return    : (BS, n_features, grid_hi, grid_hi)
        """
        BS = x.size(0)
        F = self.n_features
       
        # Reshape input to (BS, F, H, W)
        y_low = x.view(BS, F, self.grid_lo, self.grid_lo)
        y_low = y_low.view(BS, F, -1)  # (BS, F, P_lo)
       
        if cls_ids is None:
            cls_ids = torch.zeros(BS, dtype=torch.long)             
            
        idx = self.neighbor_indices 
        
        # // 2  # (P_hi, k)
        # division by 2 is needed because indices of grid_lo have spacing... - MAYBE NOT??
        
        # Debias
        low_bias = self.bias_low[cls_ids]  # (BS, F, P_lo)
        y_low_db = y_low - low_bias

        # Interpolation (upsampling)
        out = y_low.new_zeros(BS, F, self.p_hi)
        for f in range(F):
            neigh = y_low_db[:, f][:, idx]               # (BS, P_hi, k)
            weights = self.weight_map[cls_ids, f]        # (BS, P_hi, k)
            out[:, f] = (neigh * weights).sum(-1)        # (BS, P_hi)

        # Add high bias
        high_bias = self.bias_high[cls_ids]              # (BS, F, P_hi)
        out = out + high_bias                           # (BS, F, P_hi)

        # Reshape to (BS, F, grid_hi, grid_hi)
        return out.view(BS, F, self.grid_hi, self.grid_hi)
    

class LocalResiduals(nn.Module):
    """ Module class for learning residuals on a high-resolution grid
        using local neighborhoods and a shared MLP.
        Each pixel is updated based on its k nearest neighbors in the grid.
        The update is performed by a weighted linear combination of the neighbors,
        followed by a shared MLP that outputs the residual for each pixel.
        
        Args:
            height        : int : height of the grid
            width         : int : width of the grid (note: if combined with RectUpsampler, height=width=grid_size_hi and grid is square)
            n_features    : int : number of features per pixel (for EnScale: number of climate variables)
            num_neighbors : int : number of nearest neighbors to use for local updates
            map_dim       : int : dimension of the intermediate feature map after weighted combination of neighbors, serves as input to MLP
            noise_dim     : int : dimension of the noise vector concatenated to each pixel's features before weighted combination
            mlp_hidden    : int : number of hidden units in the shared MLP (if 0, no MLP is used)
            mlp_depth     : int : number of layers in the shared MLP
            shared_noise  : bool: if True, the noise vector for the MLP is shared across all pixels in the grid; only used if noise_dim_mlp > 0
            noise_dim_mlp : int : dimension of the noise vector for the MLP
            num_classes   : int : number of classes (for class-specific weights; for us: classes are different GCM-RCM pairs)
        """
    def __init__(self, 
                 height, 
                 width, 
                 n_features, 
                 num_neighbors, 
                 map_dim, 
                 noise_dim, 
                 mlp_hidden, 
                 mlp_depth, 
                 shared_noise=False, 
                 noise_dim_mlp=None, 
                 num_classes=1):
        super().__init__()
        self.height = height
        self.width = width
        self.npix = height * width
        self.n_features = n_features
        self.k = num_neighbors
        self.map_dim = map_dim
        self.noise_dim = noise_dim
        self.shared_noise = shared_noise  # If True, noise_dim is shared across all pixels
        if noise_dim_mlp is None:
            self.noise_dim_mlp = noise_dim
        else:
            self.noise_dim_mlp = noise_dim_mlp
        self.num_classes = num_classes
        
        # Weight map: (C, H*W, k, map_dim, n_features + noise_dim)
        if num_classes > 1:
            self.weight_map = nn.Parameter(torch.empty(self.num_classes, self.npix, self.k, map_dim, n_features + noise_dim))
        else:
            self.weight_map = nn.Parameter(torch.empty(self.npix, self.k, map_dim, n_features + noise_dim))
        nn.init.xavier_uniform_(self.weight_map)
        
        # Shared MLP
        self.mlp_hidden = mlp_hidden
        if mlp_hidden > 0:
            mlp_layers = []
            in_dim = map_dim + self.noise_dim_mlp
            for i in range(mlp_depth):
                out_dim = mlp_hidden if i < mlp_depth - 1 else n_features
                mlp_layers.append(nn.Linear(in_dim, out_dim))
                if i < mlp_depth - 1:
                    mlp_layers.append(nn.ReLU())
                in_dim = out_dim
            self.mlp = nn.Sequential(*mlp_layers)
        else:
            self.mlp = nn.Identity()
            
        radius = int((num_neighbors ** 0.5 - 1) // 2)
        # self.register_buffer("neighbor_indices", self._compute_neighbors(radius=radius))
        self.register_buffer("neighbor_indices", self._compute_neighbors_adjusted(radius=radius))
    
    def _compute_neighbors(self, radius=1):
        """
        Compute neighbors for each pixel in a 2D grid.
        Old version that uses uses same radius for all pixels, leading to fewer than K neighbors for edge pixels.
        In this case, out-of-bounds neighbors are replaced by self index.
        Leads to suboptimal performance at the edges.
        Args:
            radius (int): Neighborhood radius. radius=1 means 3x3 (including self), radius=2 means 5x5, etc.

        Returns:
            torch.LongTensor of shape (H*W, K), where K = (2*radius + 1)**2
        """
        neighbors = []
        for i in range(self.height):
            for j in range(self.width):
                inds = []
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.height and 0 <= nj < self.width:
                            inds.append(ni * self.width + nj)
                        else:
                            # If neighbor out of bounds, use self index as fallback
                            inds.append(i * self.width + j)
                neighbors.append(inds)
        return torch.tensor(neighbors, dtype=torch.long)  # (H*W, K)

    def _compute_neighbors_adjusted(self, radius=1):
        """
        Compute neighbors for each pixel in a 2D grid.
        New version that ensures each pixel has exactly K neighbors by expanding the search radius for edge pixels.
        For each pixel in the grid, precomputes K neighbor indices, where K = (2*radius + 1)**2.
        Args:
            radius (int): Neighborhood radius. radius=1 means 3x3 (including self), radius=2 means 5x5, etc.
        Returns:
            torch.LongTensor of shape (H*W, K)
        """ 
        H, W = self.height, self.width
        K = (2 * radius + 1) ** 2
        neighbors = torch.empty((H * W, K), dtype=torch.long)

        base_offsets = torch.stack(torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij"
        ), dim=-1).reshape(-1, 2)  # (K, 2)

        for i in range(H):
            for j in range(W):
                offset_set = base_offsets.clone()
                ni = i + offset_set[:, 0]
                nj = j + offset_set[:, 1]

                valid = (ni >= 0) & (ni < H) & (nj >= 0) & (nj < W)
                valid_inds = (ni[valid] * W + nj[valid]).tolist()

                expansion = 1
                while len(valid_inds) < K:
                    r_ext = radius + expansion
                    ext_offsets = torch.stack(torch.meshgrid(
                        torch.arange(-r_ext, r_ext + 1),
                        torch.arange(-r_ext, r_ext + 1),
                        indexing="ij"
                    ), dim=-1).reshape(-1, 2)

                    offset_set_ext = ext_offsets.tolist()
                    offset_set_ext = [tuple(x) for x in offset_set_ext if tuple(x) not in set(map(tuple, offset_set.tolist()))]

                    offset_set = torch.cat([offset_set, torch.tensor(offset_set_ext, dtype=torch.long)], dim=0)
                    ni_ext = i + torch.tensor([o[0] for o in offset_set_ext])
                    nj_ext = j + torch.tensor([o[1] for o in offset_set_ext])
                    valid_ext = (ni_ext >= 0) & (ni_ext < H) & (nj_ext >= 0) & (nj_ext < W)

                    valid_inds += (ni_ext[valid_ext] * W + nj_ext[valid_ext]).tolist()
                    expansion += 1

                neighbors[i * W + j] = torch.tensor(valid_inds[:K], dtype=torch.long)

        return neighbors  # (H*W, K)
    
    def forward(self, y_in, cls_ids = None, return_latent=False, eps=None):
        """
        Args:
            y_in: (B, n_features, H, W)
            cls_ids: (B) class indices
            return_latent: if True, also return the intermediate feature map before MLP
            eps: optional noise input; reshaped to (B, noise_dim, H, W) 
        Returns: 
            out (B, n_features, H, W)
            or (out, intermediate) if return_latent is True, where intermediate is (B, npix, map_dim)
        """
        B = y_in.shape[0]
        H, W = self.height, self.width

        # Flatten spatial dims
        y_flat = y_in.view(B, self.n_features, -1)           # (B, F, npix)
        
        # Generate noise for neighbors (B, noise_dim, npix)
        if eps is None:
            noise = torch.randn(B, self.noise_dim, self.npix, device=y_in.device)
        else:
            noise = eps.view(B, self.noise_dim, self.npix).to(y_in.device)
        y_with_noise = torch.cat([y_flat, noise], dim=1)  # (B, F + noise_dim, npix)
        
        # Gather neighbors for y_in + noise
        # Shape after gather: (B, F+noise_dim, npix, k)
        gather_y = y_with_noise.permute(0, 2, 1)[:, self.neighbor_indices, :] # (B, npix, k, F + noise_dim)
        
        # Weight map shape: (npix, k, map_dim, F + noise_dim)
        # Compute einsum over k and features: output (B, npix, map_dim)
        
        w = self.weight_map
        # add classes
        if self.num_classes > 1 and cls_ids is not None:
            weights_per_class = w[cls_ids] # shape (B, npix, k, map_dim, F + noise_dim)
            intermediate = torch.einsum("bpkn,bpkmn->bpm", gather_y, weights_per_class)  # (B, npix, map_dim)
        else:
            intermediate = torch.einsum("bpkn,pkmn->bpm", gather_y, w)  # (B, npix, map_dim)
            
        if self.noise_dim_mlp > 0:
            if self.shared_noise:
                # Generate one noise vector per batch element and expand it
                noise2 = torch.randn(B, self.noise_dim_mlp, device=y_in.device)  # (B, noise_dim)
                noise2 = noise2.unsqueeze(1).expand(-1, self.npix, -1)  # (B, npix, noise_dim)

            else:
                noise2 = torch.randn(B, self.noise_dim_mlp , self.npix, device=y_in.device).permute(0, 2, 1)  # (B, npix, noise_dim)
                
            mlp_input = torch.cat([intermediate, noise2], dim=-1)  # (B, npix, map_dim + noise_dim)
            
        else:
            mlp_input = intermediate
                
        # Run shared MLP (apply per pixel)
        mlp_input = mlp_input.contiguous().view(B * self.npix, -1)
        out = self.mlp(mlp_input)  # (B*npix, n_features)
        out = out.view(B, self.npix, self.n_features).permute(0, 2, 1)  # (B, n_features, npix)
        
        # Reshape back to (B, n_features, H, W)
        out = out.view(B, self.n_features, H, W)
        if return_latent:
            return out, intermediate
        else:
            return out
        
class RectUpsampleWithResiduals(nn.Module):
    """ Module class that combines RectUpsampler and LocalResiduals to perform super-resolution.
        First, the low-resolution input is upsampled using RectUpsampler.
        Then, LocalResiduals is applied to refine the upsampled output.
        
        Args:
            grid_size_lo     : int : size of the low-resolution grid (assumed square)
            grid_size_hi     : int : size of the high-resolution grid (assumed square)
            n_features       : int : number of features per pixel (for EnScale: number of climate variables)
            num_classes      : int : number of classes used in RectUpsampler (for class-specific weights/biases; for EnScale: classes are different GCM-RCM pairs)
            num_classes_resid: int : number of classes used in LocalResiduals (for class-specific weights; for us: classes are different GCM-RCM pairs)
            num_neighbors_ups : int : number of nearest neighbors for RectUpsampler
            num_neighbors_res : int : number of nearest neighbors for LocalResiduals
            map_dim          : int : dimension of the intermediate feature map in LocalResiduals
            noise_dim        : int : dimension of the noise vector for LocalResiduals
            mlp_hidden       : int : number of hidden units in the shared MLP of LocalResiduals
            mlp_depth        : int : number of layers in the shared MLP of LocalResiduals
            shared_noise     : bool: if True, the noise vector for the MLP in LocalResiduals is shared across all pixels in the grid
            noise_dim_mlp  : int : dimension of the noise vector for the MLP in LocalResiduals
            double_linear    : bool: if True, use LocalResiduals2 with two MLPs instead of one
            split_residuals  : bool: if True, add the upsampled output to the residuals output before returning; 
                if False, the residuals output is returned directly (and thus represents the full output)
        """
    def __init__(self,
                 grid_size_lo,
                 grid_size_hi,
                 n_features=4,
                 num_classes=1,
                 num_classes_resid=1,
                 num_neighbors_ups=9,
                 num_neighbors_res=25,
                 map_dim=16,
                 noise_dim=4,
                 mlp_hidden=32,
                 mlp_depth=2,
                 shared_noise=False, 
                 noise_dim_mlp=None, 
                 double_linear=False,
                 split_residuals=True):
        super().__init__()
        self.upsampler = RectUpsampler(
            grid_size_lo=grid_size_lo,
            grid_size_hi=grid_size_hi,
            n_features=n_features,
            num_classes=num_classes,
            num_neighbors=num_neighbors_ups
        )
        self.double_linear = double_linear
        if double_linear:
            # If double_linear is True, use LocalResiduals2
            raise NotImplementedError("LocalResiduals2 is not implemented in this code snippet.")
        else:
            self.residuals = LocalResiduals(
                height=grid_size_hi,
                width=grid_size_hi,
                n_features=n_features,
                num_neighbors=num_neighbors_res,
                map_dim=map_dim,
                noise_dim=noise_dim,
                mlp_hidden=mlp_hidden,
                mlp_depth=mlp_depth,
                shared_noise=shared_noise,
                noise_dim_mlp=noise_dim_mlp,
                num_classes=num_classes_resid
            )
        self.split_residuals = split_residuals

    def forward(self, x: torch.Tensor, cls_ids: torch.Tensor = None, 
                return_latent: bool = False, return_mean: bool = False,
                eps: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x         : (B, n_features * p_lo)
            cls_ids  : (B)
            return_latent : if True, also return intermediate feature maps from LocalResiduals
                ()
            return_mean   : if True, also return the upsampled output before residuals
            eps          : (B, n_features, grid_hi, grid_hi) optional noise input for LocalResiduals
            
        Notes:
            map_dim should equal number of features if return_latent is True (such that y_upsampled and intermediate have same dimension)
            
        Returns:
            output    : (B, n_features * grid_hi * grid_hi)
        """
        if self.double_linear:
            raise NotImplementedError("LocalResiduals2 is not implemented in this code snippet.")
        y_upsampled = self.upsampler(x, cls_ids)                # (B, F, H, W)
        if return_latent:
            y_out, intermediate = self.residuals(y_upsampled, cls_ids=cls_ids, return_latent=True, eps=eps)
            intermediate = intermediate.transpose(1,2).reshape(y_upsampled.shape[0], -1, y_upsampled.shape[2], y_upsampled.shape[3])  # (B, map_dim, H, W)
            if self.split_residuals:
                y_interm = (intermediate + y_upsampled).reshape(y_upsampled.shape[0], -1)  # (B, F * H * W)
            else:
                y_interm = intermediate.reshape(y_upsampled.shape[0], -1)  # (B, F * H * W)
        
            if self.split_residuals:
                y_out = y_out + y_upsampled                               # (B, F, H, W)
                
            y_out = y_out.reshape(y_out.shape[0], -1)
            return y_out, y_interm
        else:
            if self.split_residuals:
                y_out = self.residuals(y_upsampled, cls_ids=cls_ids, eps=eps) + y_upsampled        # (B, F, H, W)
            else:
                y_out = self.residuals(y_upsampled, cls_ids=cls_ids, eps=eps)
            y_out = y_out.contiguous().reshape(y_out.shape[0], -1)
            if return_mean:
                return y_out, y_upsampled
            else:
                return y_out

    @torch.no_grad()
    def sample(self, x: torch.Tensor, cls_ids: torch.Tensor = None, sample_size: int = 1, expand_dim: bool = True) -> torch.Tensor:
        """
        Efficiently generates multiple samples from the residual model.
        RectUpsampler is applied once per input, then LocalResiduals is applied `sample_size` times.

        Args:
            x         : (B, n_features * p_lo)
            cls_ids  : (B, num_classes)
            sample_size : int
            expand_dim  : if True, return shape (B, F, H, W, sample_size), else (B * sample_size, F, H, W)

        Returns:
            samples: either (B, F, H, W, sample_size) or (B * sample_size, F, H, W)
        """
        B = x.shape[0]

        # Apply RectUpsampler once
        y_upsampled = self.upsampler(x, cls_ids)     # (B, F, H, W)
        num_features = y_upsampled.shape[1]

        # Repeat upsampled input sample_size times
        y_rep = y_upsampled.unsqueeze(0).expand(sample_size, -1, -1, -1, -1)  # (S, B, F, H, W)
        y_rep = y_rep.contiguous().view(sample_size * B, *y_upsampled.shape[1:])  # (S*B, F, H, W)

        # Apply residual model
        # Repeat cls_ids to match y_rep shape (S*B, ...)
        if cls_ids is not None:
            cls_ids_rep = cls_ids.repeat(sample_size)
        else:
            cls_ids_rep = None

        residuals_out = self.residuals(y_rep, cls_ids=cls_ids_rep)  # (S*B, F, H, W)

        # Add residual to upsampled base
        if self.split_residuals:
            y_upsampled_flat = y_upsampled.repeat(sample_size, 1, 1, 1)  # (S*B, F, H, W)
            samples = residuals_out + y_upsampled_flat  # (S*B, F, H, W)
        else:
            samples = residuals_out

        if not expand_dim:
            return samples  # shape: (S*B, F, H, W)
        else:
            # Reshape to (S, B, F, H, W) and permute to (B, F, H, W, S)
            samples = samples.view(sample_size, B, *samples.shape[1:])  # (S, B, F, H, W)
            samples = samples.permute(1, 2, 3, 4, 0).contiguous()       # (B, F, H, W, S)
            # flatten to (B, F, H * W, S)
            samples = samples.view(B, num_features, -1, sample_size) 
            return samples