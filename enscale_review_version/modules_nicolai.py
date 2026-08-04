


class LB2(nn.Module):
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        nside_lo: int,
        nside_hi: int,
        *,
        n_features: int = 4,
        num_neighbors: int = 16,
        num_max: int = 35,       # highest cls index
    ):
        super().__init__()

        # ─── geometry ---------------------------------------------------
        self.nside_lo, self.nside_hi = nside_lo, nside_hi
        self.npix_hi = hp.nside2npix(nside_hi)
        self.npix_lo = hp.nside2npix(nside_lo) if nside_lo >= 1 else 0

        # ─── parameters -------------------------------------------------
        self.n_features = n_features
        self.num_cls    = num_max + 1        # include 0-class
        self.num_mon    = 13                 # 1 … 12

        # ─── neighbour indices (low→high) ------------------------------
        if self.npix_lo > 0:
            k = min(num_neighbors, self.npix_lo)
            lo_vec = np.vstack(hp.pix2vec(nside_lo, np.arange(self.npix_lo))).T
            hi_vec = np.vstack(hp.pix2vec(nside_hi, np.arange(self.npix_hi))).T
            idx_low = cKDTree(lo_vec).query(hi_vec, k=k)[1]             # (P_hi,k)
            self.register_buffer("neighbor_indices",
                                 torch.tensor(idx_low, dtype=torch.long))
            self.k = k
        else:                       # degenerate (no low-res grid)
            self.k = 0
            self.register_buffer("neighbor_indices",
                                 torch.empty(self.npix_hi, 0, dtype=torch.long))

        # ─── weight map: class × feature specific ----------------------
        if self.k > 0:
            self.weight_map = nn.Parameter(           # (C,F,P_hi,k)
                torch.zeros(self.num_cls, n_features, self.npix_hi, self.k)
            )
        else:
            self.weight_map = None

        # ─── biases: cls × mon × feat × pix ----------------------------
        if self.npix_lo > 0:
            self.bias_low = nn.Parameter(
                torch.zeros(self.num_cls, self.num_mon,
                            n_features, self.npix_lo)
            )                                        # (C,M,F,P_lo)
        else:
            self.bias_low = None

        self.bias_high = nn.Parameter(
            torch.zeros(self.num_cls, self.num_mon,
                        n_features, self.npix_hi)
        )                                            # (C,M,F,P_hi)

    # ──────────────────────────────────────────────────────────────────
    def _upsample(self,
                  y_low: torch.Tensor,
                  cls:   torch.Tensor) -> torch.Tensor:
        """
        y_low : (B,F,P_lo)  
        cls   : (B,)        
        return: (B,F,P_hi)
        """
        B, F, _ = y_low.shape
        dev = y_low.device

        # start with zeros; bias will be added outside
        out = y_low.new_zeros(B, F, self.npix_hi)

        if self.k == 0:
            return out                     # nothing to up-sample

        idx = self.neighbor_indices        # (P_hi,k)

        for f in range(self.n_features):
            neigh = y_low[:, f][:, idx]                       # (B,P_hi,k)
            w     = self.weight_map[cls, f]                   # (B,P_hi,k)
            out[:, f] = (neigh * w).sum(-1)                   # (B,P_hi)

        return out                                            # (B,F,P_hi)

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,                   # (…,≥2) – only first two used
        y_low: torch.Tensor | None = None, # (…,F,P_lo) or None
        *,
        high: bool = True,                 # for bias-only mode
    ) -> torch.Tensor:

        # --- extract class & month ------------------------------------
        if x.dim() > 1:
            cls  = x[..., 0].long().clamp_(0, self.num_cls - 1)
            mon  = x[..., 1].long().clamp_(0, self.num_mon - 1)
        else:              
            raise RuntimeError("x needs at least first two dimensions (cls and month)")


        # flatten batch dims into leading dim “B”
        cls_flat = cls.view(-1)
        mon_flat = mon.view(-1)

        # ---------- bias–only path ------------------------------------
        if y_low is None:
            bias = (self.bias_high if high else self.bias_low)
            if bias is None:
                raise RuntimeError("Low-res grid absent → no low bias")
            out = bias[cls_flat, mon_flat]                 # (B,F,P)
            return out.view(*cls.shape, self.n_features,
                            -1)                            # restore batch-shape

        # ---------- full path (y_low given) ---------------------------
        if y_low.dim() != 3:
            raise ValueError("y_low must be 3-D (B,F,P_lo)")

        if self.bias_low is None:
            raise RuntimeError("Model was initialised without low-res grid")

        B = y_low.size(0)

        low_bias = self.bias_low[cls_flat, mon_flat].view_as(y_low)  # (B,F,P_lo)
        y_low_db = y_low - low_bias                                  # debiased

        up = self._upsample(y_low_db, cls_flat)                      # (B,F,P_hi)

        high_bias = self.bias_high[cls_flat, mon_flat]               # (B,F,P_hi)
        return up + high_bias  





class L2fl(nn.Module):
    """
    Original sparse-map network (high resolution) **plus** an optional
    autoregressive latent branch identical to the one in class *L2*.
    """
    # -----------------------------------------------------------------
    def __init__(
        self,
        nside: int,
        *,
        # ---------- original L2f arguments ----------------------------
        n_features: int = 4,
        x_features: int = 10,
        x_dim: int = 5,
        map_dim: int = 12,
        noise_dim_aug: int = 5,
        noise_dim_high: int = 5,
        num_neighbors: int = 25,
        mlp_hidden: int = 32,
        mlp_depth: int = 4,
        depth_mlp_x: int = 3,
        hidden_x: int = 16,
        noise_x: int = 5,
        variab: int = 0,
        num_max: int = 35,
        # ---------- NEW (to match L2) ---------------------------------
        add_latent_dim: int | None = None,
        noise_dim_prev: int | None = None,
        lags: list[int] = (1, 2),
    ):
        super().__init__()

        # ====== keep all old config unchanged =========================
        self.var_dims = max(0, int(variab))
        self.num_max  = int(num_max)

        self.nside = nside
        self.npix  = hp.nside2npix(nside)

        self.n_features     = n_features
        self.x_features     = x_features
        self.x_dim          = x_dim
        self.map_dim        = map_dim
        self.noise_dim_aug  = noise_dim_aug
        self.noise_dim_high = noise_dim_high

        # ===== neighbour indices (high-res) ===========================
        self.k = min(num_neighbors, self.npix)
        vec = np.vstack(hp.pix2vec(nside, np.arange(self.npix))).T
        idx = cKDTree(vec).query(vec, k=self.k)[1]
        self.register_buffer("neighbor_indices",
                             torch.tensor(idx.clip(0, self.npix-1),
                                          dtype=torch.long))          # (P,k)

        # ===== learnable sparse-map weights ==========================
        self.weight_map = nn.Parameter(
            torch.empty(self.npix, self.k, map_dim,
                        n_features + noise_dim_aug))
        nn.init.xavier_uniform_(self.weight_map)

        # ===== optional class scaling =================================
        if self.var_dims > 0:
            self.scale_raw = nn.Parameter(
                torch.zeros(self.num_max+1, self.var_dims))
        else:
            self.register_parameter("scale_raw", None)

        # ===== global-x encoder =======================================
        self.noise_x = noise_x
        self.mlp_x_reduce = make_mlp(
            (x_features-1) + noise_x, x_dim, hidden_x, depth_mlp_x)

        # ===== pixel angles ===========================================
        theta, phi = hp.pix2ang(nside, np.arange(self.npix))
        self.register_buffer(
            "angles",
            torch.from_numpy(np.stack([theta, phi], 1).astype(np.float32)))

        
        self.lags   = list(lags)
        self.n_prev = len(self.lags)

        self.add_latent_dim = (n_features if add_latent_dim is None
                               else add_latent_dim)
        self.noise_dim_prev = (n_features if noise_dim_prev is None
                               else noise_dim_prev)

        self.D_prev = (x_dim
                       + n_features                # current y_in average
                       + self.noise_dim_prev
                       + self.n_prev * self.k * n_features)

        self.weight_prev_latent = nn.Parameter(
            torch.empty(self.npix, self.k,
                        self.add_latent_dim, self.D_prev))
        nn.init.xavier_uniform_(self.weight_prev_latent)

        self.mlp_in_dim = x_dim + map_dim + noise_dim_high + 2 + self.add_latent_dim
        self.latent_mlp = make_mlp(self.mlp_in_dim,
                                   n_features,
                                   mlp_hidden,
                                   mlp_depth)

    # -----------------------------------------------------------------
    # helper for neighbour aggregation of previous residuals
    # -----------------------------------------------------------------
    def _aggregate_prev(self, y_prev):
        # y_prev: (B,F,P) → (B,P,k*F)
        y_perm = y_prev.permute(0, 2, 1)                       # (B,P,F)
        gathered = y_perm[:, self.neighbor_indices, :]         # (B,P,k,F)
        return gathered.reshape(gathered.size(0), gathered.size(1), -1)

    # -----------------------------------------------------------------
    # forward for a single step  (unchanged maths + extra latent)
    # -----------------------------------------------------------------
    def _forward_single(self, x, y_in, *, prev_list=None):
        """
        x      : (B, x_features)
        y_in   : (B, n_features, P)     – low-res input for this step
        prev_list : list[Tensor|None] of length n_prev,    each (B,F,P)
        """
        B, dev = x.size(0), x.device
        prev_list = prev_list or [None] * self.n_prev

        # ===== sparse-map from y_in ==================================
        noise_aug = torch.randn(B, self.noise_dim_aug,
                                self.npix, device=dev)
        y_cat   = torch.cat([y_in, noise_aug], 1)              # (B,C,P)
        gather  = y_cat.permute(0, 2, 1)[:, self.neighbor_indices, :]   # (B,P,k,C)
        sparse  = torch.einsum("bpkn,pkmc->bpm",
                               gather, self.weight_map)        # (B,P,map)

        # ===== encode x (drop first col) =============================
        noise_vec = torch.randn(B, self.noise_x, device=dev)
        x_red = self.mlp_x_reduce(torch.cat([x[:, 1:], noise_vec], 1))   # (B,x_dim)
        x_exp = x_red.unsqueeze(1).expand(B, self.npix, self.x_dim)      # (B,P,x_dim)

        # ===== optional class-vector scaling =========================
        if self.var_dims > 0:
            cls   = x[:, 0].long().clamp_(0, self.num_max)
            scale = torch.exp(self.scale_raw[cls])                       # (B,var)
            d     = min(self.var_dims, self.x_dim)
            mask  = torch.ones_like(x_exp)
            mask[:, :, :d] = scale.view(B, 1, d)
            x_exp = x_exp * mask

        # ===== high-level noise vector ===============================
        noise_high = None
        if self.noise_dim_high:
            noise_high = (torch.randn(B, self.noise_dim_high,
                                      self.npix, device=dev)
                          .permute(0, 2, 1))                # (B,P,noise)

        # ===== additional latent branch (lags) =======================
        if any(p is None for p in prev_list):
            add_latent = (2*torch.rand(B, self.npix,
                                       self.add_latent_dim, device=dev) - 1)
        else:
            x_lat = x_exp                                      # (B,P,x_dim)

            noise_prev = (2*torch.rand(B, self.npix,
                                        self.noise_dim_prev, device=dev) - 1)
            y_naive = y_in.permute(0, 2, 1)

            prev_concat = torch.cat([self._aggregate_prev(p)
                                     for p in prev_list], 2)   # (B,P,k*F*n_prev)

            add_in = torch.cat([x_lat, y_naive, noise_prev, prev_concat], 2)  # (B,P,D_prev)
            gathered = add_in[:, self.neighbor_indices, :]           # (B,P,k,D_prev)
            add_latent = torch.einsum("bpkn,pkmn->bpm",
                                      gathered, self.weight_prev_latent)
            add_latent = torch.tanh(add_latent)                      # (B,P,add_latent)

        # ===== residual prediction ===================================
        parts = [x_exp, sparse, self.angles.expand(B, -1, -1), add_latent]
        if noise_high is not None:
            parts.append(noise_high)

        mlp_in  = torch.cat(parts, 2)                         # (B,P,Dtot)
        residual = self.latent_mlp(mlp_in).permute(0, 2, 1)   # (B,F,P)
        return residual

    # -----------------------------------------------------------------
    # public forward (single step or autoregressive sequence)
    # -----------------------------------------------------------------
    def forward(
        self,
        x,              # (..., x_features)
        y_in,           # (..., n_features, P)
        *,
        dependence: bool = False,
        prev_tensors: dict[int, torch.Tensor] | None = None,
    ):
        # --- single step --------------------------------------------
        if not dependence:
            prev_list = None
            if prev_tensors is not None:
                prev_list = [prev_tensors.get(l) for l in self.lags]
            return self._forward_single(x, y_in, prev_list=prev_list)

        # --- autoregressive sequence --------------------------------
        orig_2d = x.dim() == 2
        if orig_2d:
            x, y_in = x.unsqueeze(1), y_in.unsqueeze(1)       # add time dim

        T = x.size(0)
        outs = []
        for t in range(T):
            prev_list = [outs[t - l] if t >= l else None for l in self.lags]
            outs.append(self._forward_single(x[t], y_in[t],
                                             prev_list=prev_list))

        out = torch.stack(outs, 0)                            # (T,B,F,P)
        return out.squeeze(1) if orig_2d else out
    