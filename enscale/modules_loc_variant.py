
    
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree


class RectUpsampler(nn.Module):
    def __init__(self,
                 grid_size_lo: int,
                 grid_size_hi: int,
                 n_features: int = 4,
                 num_classes: int = 36,
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
        
   
    """
    # was supposed to be a faster version of below
    # but not yet equivalent
        def _compute_2d_neighbors(self):
            ratio = self.grid_hi // self.grid_lo
            hi_coords = self._grid_coords(self.grid_hi)
            hi_coords = torch.tensor(hi_coords, dtype=torch.long)

            lo_i = hi_coords[:, 0] // ratio
            lo_j = hi_coords[:, 1] // ratio

            k = self.k
            P_hi = hi_coords.size(0)
            neighbors = torch.full((P_hi, k), -1, dtype=torch.long)

            count = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if count >= k:
                        break
                    ni = (lo_i + di).clamp(0, self.grid_lo - 1)
                    nj = (lo_j + dj).clamp(0, self.grid_lo - 1)
                    neighbors[:, count] = ni * self.grid_lo + nj
                    count += 1
            return neighbors
        """
    
    def _compute_2d_neighbors(self):
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
    def __init__(self, height, width, n_features, num_neighbors, map_dim, noise_dim, mlp_hidden, mlp_depth, 
                 shared_noise=False, noise_dim_mlp=None, softmax=False, num_classes=1):
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
        self.softmax = softmax
        self.num_classes = num_classes
        
        # Weight map: (H*W, k, map_dim, n_features + noise_dim)
        # self.weight_map = nn.Parameter(torch.empty(self.npix, self.k, map_dim, n_features + noise_dim))
        
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
    
    def forward(self, y_in, cls_ids = None, return_latent=False):
        """
        y_in: (B, n_features, H, W)
        returns: (B, n_features, H, W)
        """
        B = y_in.shape[0]
        H, W = self.height, self.width
        
        if cls_ids is None:
            cls_ids = torch.zeros(B, dtype=torch.long)

        # Flatten spatial dims
        y_flat = y_in.view(B, self.n_features, -1)           # (B, F, npix)
        
        # Generate noise for neighbors (B, noise_dim, npix)
        noise = torch.randn(B, self.noise_dim, self.npix, device=y_in.device)
        y_with_noise = torch.cat([y_flat, noise], dim=1)  # (B, F + noise_dim, npix)
        
        # Gather neighbors for y_in + noise
        # Shape after gather: (B, F+noise_dim, npix, k)
        gather_y = y_with_noise.permute(0, 2, 1)[:, self.neighbor_indices, :] # (B, npix, k, F + noise_dim)
        
        # Weight map shape: (npix, k, map_dim, F + noise_dim)
        # Compute einsum over k and features: output (B, npix, map_dim)
        
        # optional: softmax over k in weight_map
        if self.softmax:
            w = F.softmax(self.weight_map, dim=1)  # (npix, k, map_dim)
        else:
            w = self.weight_map
        # intermediate = torch.einsum("bpkn,pkmc->bpm", gather_y, w)  # (B, npix, map_dim) - WRONG
        # intermediate = torch.einsum("bpkn,pkmn->bpm", gather_y, w)  # (B, npix, map_dim)

        # add classes
        if self.num_classes > 1:
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

# class LocalResiduals2(nn.Module):
#     def __init__(self, height, width, n_features, num_neighbors, map_dim, noise_dim, mlp_hidden, mlp_depth, 
#                  shared_noise=False, noise_dim_mlp=None, softmax=False):
#         super().__init__()
#         self.height = height
#         self.width = width
#         self.npix = height * width
#         self.n_features = n_features
#         self.k = num_neighbors
#         self.map_dim = map_dim
#         self.noise_dim = noise_dim
#         self.shared_noise = shared_noise  # If True, noise_dim is shared across all pixels
#         if noise_dim_mlp is None:
#             self.noise_dim_mlp = noise_dim
#         else:
#             self.noise_dim_mlp = noise_dim_mlp
#         self.softmax = softmax
        
#         # Weight map: (H*W, k, map_dim, n_features + noise_dim)
#         self.weight_map = nn.Parameter(torch.empty(self.npix, self.k, map_dim, n_features + noise_dim))
#         nn.init.xavier_uniform_(self.weight_map)
        
#         self.weight_map2 = nn.Parameter(torch.empty(self.npix, self.k, map_dim, map_dim))
#         nn.init.xavier_uniform_(self.weight_map2)
        
#         # Shared MLP
#         self.mlp_hidden = mlp_hidden
#         if mlp_hidden > 0:
#             mlp_layers = []
#             in_dim = map_dim + self.noise_dim_mlp
#             for i in range(mlp_depth):
#                 out_dim = mlp_hidden if i < mlp_depth - 1 else n_features
#                 mlp_layers.append(nn.Linear(in_dim, out_dim))
#                 if i < mlp_depth - 1:
#                     mlp_layers.append(nn.ReLU())
#                 in_dim = out_dim
#             self.mlp = nn.Sequential(*mlp_layers)
#         else:
#             self.mlp = nn.Identity()
            
#         radius = int((num_neighbors ** 0.5 - 1) // 2)
#         # self.register_buffer("neighbor_indices", self._compute_neighbors(radius=radius))
#         self.register_buffer("neighbor_indices", self._compute_neighbors_adjusted(radius=radius))
    
#     def _compute_neighbors(self, radius=1):
#         """
#         Compute neighbors for each pixel in a 2D grid.

#         Args:
#             radius (int): Neighborhood radius. radius=1 means 3x3 (including self), radius=2 means 5x5, etc.

#         Returns:
#             torch.LongTensor of shape (H*W, K), where K = (2*radius + 1)**2
#         """
#         neighbors = []
#         for i in range(self.height):
#             for j in range(self.width):
#                 inds = []
#                 for di in range(-radius, radius + 1):
#                     for dj in range(-radius, radius + 1):
#                         ni, nj = i + di, j + dj
#                         if 0 <= ni < self.height and 0 <= nj < self.width:
#                             inds.append(ni * self.width + nj)
#                         else:
#                             # If neighbor out of bounds, use self index as fallback
#                             inds.append(i * self.width + j)
#                 neighbors.append(inds)
#         return torch.tensor(neighbors, dtype=torch.long)  # (H*W, K)
    
#     def _compute_neighbors_adjusted(self, radius=1):
#         H, W = self.height, self.width
#         K = (2 * radius + 1) ** 2
#         neighbors = torch.empty((H * W, K), dtype=torch.long)

#         base_offsets = torch.stack(torch.meshgrid(
#             torch.arange(-radius, radius + 1),
#             torch.arange(-radius, radius + 1),
#             indexing="ij"
#         ), dim=-1).reshape(-1, 2)  # (K, 2)

#         for i in range(H):
#             for j in range(W):
#                 offset_set = base_offsets.clone()
#                 ni = i + offset_set[:, 0]
#                 nj = j + offset_set[:, 1]

#                 valid = (ni >= 0) & (ni < H) & (nj >= 0) & (nj < W)
#                 valid_inds = (ni[valid] * W + nj[valid]).tolist()

#                 expansion = 1
#                 while len(valid_inds) < K:
#                     r_ext = radius + expansion
#                     ext_offsets = torch.stack(torch.meshgrid(
#                         torch.arange(-r_ext, r_ext + 1),
#                         torch.arange(-r_ext, r_ext + 1),
#                         indexing="ij"
#                     ), dim=-1).reshape(-1, 2)

#                     offset_set_ext = ext_offsets.tolist()
#                     offset_set_ext = [tuple(x) for x in offset_set_ext if tuple(x) not in set(map(tuple, offset_set.tolist()))]

#                     offset_set = torch.cat([offset_set, torch.tensor(offset_set_ext, dtype=torch.long)], dim=0)
#                     ni_ext = i + torch.tensor([o[0] for o in offset_set_ext])
#                     nj_ext = j + torch.tensor([o[1] for o in offset_set_ext])
#                     valid_ext = (ni_ext >= 0) & (ni_ext < H) & (nj_ext >= 0) & (nj_ext < W)

#                     valid_inds += (ni_ext[valid_ext] * W + nj_ext[valid_ext]).tolist()
#                     expansion += 1

#                 neighbors[i * W + j] = torch.tensor(valid_inds[:K], dtype=torch.long)

#         return neighbors  # (H*W, K)
    
    
#     def forward(self, y_in, return_latent=False):
#         """
#         y_in: (B, n_features, H, W)
#         returns: (B, n_features, H, W)
#         """
#         B = y_in.shape[0]
#         H, W = self.height, self.width
        
#         # Flatten spatial dims
#         y_flat = y_in.view(B, self.n_features, -1)           # (B, F, npix)
        
#         # for Nicolai, gather has # (B,P,k,C); C is n_features + noise_dim
#         # y_in is flat
#         # noise_aug = torch.randn(B, self.noise_dim_aug,
#         #                        self.npix, device=dev)
#         #y_cat   = torch.cat([y_in, noise_aug], 1)              # (B,C,P)
#         #gather  = y_cat.permute(0, 2, 1)[:, self.neighbor_indices, :]   # (B,P,k,C)
#         #sparse  = torch.einsum("bpkn,pkmc->bpm",
#         #                       gather, self.weight_map)     # (B,P,map)
        
#         # Generate noise for neighbors (B, noise_dim, npix)
#         noise = torch.randn(B, self.noise_dim, self.npix, device=y_in.device)
#         y_with_noise = torch.cat([y_flat, noise], dim=1)  # (B, F + noise_dim, npix)
        
#         # Gather neighbors for y_in + noise
#         # Shape after gather: (B, F+noise_dim, npix, k)
#         gather_y = y_with_noise.permute(0, 2, 1)[:, self.neighbor_indices, :] # (B, npix, k, F + noise_dim)
        
#         # Weight map shape: (npix, k, map_dim, F + noise_dim)
#         # Compute einsum over k and features: output (B, npix, map_dim)
        
#         # optional: softmax over k in weight_map
#         if self.softmax:
#             w = F.softmax(self.weight_map, dim=1)  # (npix, k, map_dim)
#             w2 = F.softmax(self.weight_map2, dim=1)  # (npix, k, map_dim)
#         else:
#             w = self.weight_map
#             w2 = self.weight_map2
                
#         #intermediate = torch.einsum("bpkn,pkmc->bpm", gather_y, w)  # (B, npix, map_dim)
#         #intermediate2 = torch.einsum("bpkn,pkmc->bpm", intermediate[:,self.neighbor_indices,:], w2)  # (B, npix, map_dim)

#         intermediate = torch.einsum("bpkn,pkmn->bpm", gather_y, w)  # (B, npix, map_dim)
#         # intermediate[:,self.neighbor_indices,:] is (B, npix, k, map_dim)
#         # weightmap2 has shape (npix, k, map_dim, map_dim)
#         intermediate2 = torch.einsum("bpkn,pkmn->bpm", intermediate[:,self.neighbor_indices,:], w2)  # (B, npix, map_dim)
        
#         if self.noise_dim_mlp > 0:
#             if self.shared_noise:
#                 # Generate one noise vector per batch element and expand it
#                 noise2 = torch.randn(B, self.noise_dim_mlp, device=y_in.device)  # (B, noise_dim)
#                 noise2 = noise2.unsqueeze(1).expand(-1, self.npix, -1)  # (B, npix, noise_dim)

#             else:
#                 noise2 = torch.randn(B, self.noise_dim_mlp , self.npix, device=y_in.device).permute(0, 2, 1)  # (B, npix, noise_dim)
                
#             mlp_input = torch.cat([intermediate2, noise2], dim=-1)  # (B, npix, map_dim + noise_dim)
            
#         else:
#             mlp_input = intermediate2
                
#         # Run shared MLP (apply per pixel)
#         mlp_input = mlp_input.contiguous().view(B * self.npix, -1)
#         out = self.mlp(mlp_input)  # (B*npix, n_features)
#         out = out.view(B, self.npix, self.n_features).permute(0, 2, 1)  # (B, n_features, npix)
        
#         # Reshape back to (B, n_features, H, W)
#         out = out.view(B, self.n_features, H, W)
#         if return_latent:
#             return out, intermediate, intermediate2
#         else:
#             return out

class RectUpsampleWithResiduals(nn.Module):
    def __init__(self,
                 grid_size_lo,
                 grid_size_hi,
                 n_features=4,
                 num_classes=1,
                 num_neighbors_ups=9,
                 num_neighbors_res=25,
                 map_dim=16,
                 noise_dim=4,
                 mlp_hidden=32,
                 mlp_depth=2,
                 shared_noise=False, 
                 noise_dim_mlp=None, 
                 double_linear=False,
                 softmax=False,
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
            self.residuals = LocalResiduals2(
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
                softmax=softmax
            )
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
                softmax=softmax,
                num_classes=num_classes
            )
        self.split_residuals = split_residuals

    def forward(self, x: torch.Tensor, cls_ids: torch.Tensor = None, 
                return_latent: bool = False, return_mean: bool = False) -> torch.Tensor:
        """
        Args:
            x         : (B, n_features * p_lo)
            cls_ids  : (B)
        Returns:
            output    : (B, n_features, grid_hi, grid_hi)
        """
        if return_latent:
            y_upsampled = self.upsampler(x, cls_ids)                # (B, F, H, W)
            if self.double_linear:
                y_out, intermediate, intermediate2 = self.residuals(y_upsampled, cls_ids=cls_ids, return_latent=True)
                # shape intermediate (B, npix, map_dim)
                # reshape to match shape of y_upsampled
                intermediate = intermediate.transpose(1,2).reshape(y_upsampled.shape[0], -1, y_upsampled.shape[2], y_upsampled.shape[3])  # (B, map_dim, H, W)
                intermediate2 = intermediate2.transpose(1,2).reshape(y_upsampled.shape[0], -1, y_upsampled.shape[2], y_upsampled.shape[3])
                
                if self.split_residuals:
                    y_interm = (intermediate + y_upsampled).reshape(y_upsampled.shape[0], -1)  # (B, F * H * W)
                    y_interm2 = (intermediate2 + y_upsampled).reshape(y_upsampled.shape[0], -1)  # (B, F * H * W)
                else:
                    y_interm = intermediate.reshape(y_upsampled.shape[0], -1)  # (B, F * H * W)
                    y_interm2 = intermediate2.reshape(y_upsampled.shape[0], -1)  # (B, F * H * W)
            else:
                y_out, intermediate = self.residuals(y_upsampled, cls_ids=cls_ids, return_latent=True)
                intermediate = intermediate.transpose(1,2).reshape(y_upsampled.shape[0], -1, y_upsampled.shape[2], y_upsampled.shape[3])  # (B, map_dim, H, W)
                if self.split_residuals:
                    y_interm = (intermediate + y_upsampled).reshape(y_upsampled.shape[0], -1)  # (B, F * H * W)
                else:
                    y_interm = intermediate.reshape(y_upsampled.shape[0], -1)  # (B, F * H * W)
            
            if self.split_residuals:
                y_out = y_out + y_upsampled                               # (B, F, H, W)
                
            y_out = y_out.reshape(y_out.shape[0], -1)
            if self.double_linear:
                return y_out, y_interm, y_interm2
            else:
                return y_out, y_interm
        else:
            y_upsampled = self.upsampler(x, cls_ids)                # (B, F, H, W)
            if self.split_residuals:
                y_out = self.residuals(y_upsampled, cls_ids=cls_ids) + y_upsampled        # (B, F, H, W)
            else:
                y_out = self.residuals(y_upsampled, cls_ids=cls_ids)
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