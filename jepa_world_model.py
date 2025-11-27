import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvEncoder(nn.Module):
    """Encoder f: maps raw observation x_t → latent z_t."""
    
    def __init__(self, in_channels=3, latent_dim=512, channels=[32, 64, 128, 256]):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Build conv blocks
        layers = []
        in_ch = in_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.GELU()
            ])
            in_ch = out_ch
        
        self.conv_blocks = nn.Sequential(*layers)
        
        # Use Max Pooling instead of Avg Pooling to preserve small features (like the ball)
        # AdaptiveAvgPool washes out small bright objects against dark background
        self.adaptive_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.flatten_dim = channels[-1] * 4 * 4
        self.fc = nn.Linear(self.flatten_dim, latent_dim)
    
    def forward(self, x):
        # x: (B, C, H, W)
        h = self.conv_blocks(x)
        h = self.adaptive_pool(h)  # (B, channels[-1], 4, 4)
        h = h.flatten(1)  # (B, channels[-1] * 4 * 4)
        z = self.fc(h)  # (B, latent_dim)
        return z


class ContextEncoder(nn.Module):
    """Transformer-based context encoder over past latents and actions."""
    
    def __init__(self, latent_dim=512, action_dim=None, d_action_embed=64, 
                 n_layers=3, n_heads=8, d_ff_scale=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.use_actions = action_dim is not None
        
        if self.use_actions:
            self.action_embed = nn.Linear(action_dim, d_action_embed)
            d_token = latent_dim + d_action_embed
        else:
            d_token = latent_dim
        
        self.d_token = d_token
        
        # Positional encoding (learned)
        self.pos_embed = nn.Parameter(torch.randn(1, 100, d_token))  # max 100 steps
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ff_scale * d_token,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, z_ctx, a_ctx=None):
        """
        Args:
            z_ctx: (B, K_ctx, latent_dim) - past latents
            a_ctx: (B, K_ctx-1, action_dim) or None - past actions
        Returns:
            h_ctx: (B, d_token) - context representation
        """
        B, K_ctx, _ = z_ctx.shape
        
        if self.use_actions and a_ctx is not None:
            # Embed actions
            a_emb = self.action_embed(a_ctx)  # (B, K_ctx-1, d_action_embed)
            
            # Align: z_ctx has K_ctx steps, a_ctx has K_ctx-1
            # We'll use z_t and a_{t-1} for step t
            # For first step, use zero action embedding
            a_padded = torch.zeros(B, 1, a_emb.shape[-1], device=a_emb.device)
            a_emb = torch.cat([a_padded, a_emb], dim=1)  # (B, K_ctx, d_action_embed)
            
            # Concatenate
            tokens = torch.cat([z_ctx, a_emb], dim=-1)  # (B, K_ctx, d_token)
        else:
            tokens = z_ctx  # (B, K_ctx, d_token)
        
        # Add positional encoding
        tokens = tokens + self.pos_embed[:, :K_ctx, :]
        
        # Transformer encode
        tokens_ctx = self.transformer(tokens)  # (B, K_ctx, d_token)
        
        # Use last token output
        h_ctx = tokens_ctx[:, -1, :]  # (B, d_token)
        
        return h_ctx


class Predictor(nn.Module):
    """Predictor g: maps (context, future_actions) → predicted latent z_hat_{t+k}."""
    
    def __init__(self, ctx_dim, latent_dim=512, d_offset=64, max_offset=10, 
                 action_dim=None, d_action_embed=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_offset = d_offset
        self.action_dim = action_dim
        self.use_actions = action_dim is not None
        
        # Offset embedding
        self.offset_embed = nn.Embedding(max_offset + 1, d_offset)
        
        # Action embedding
        if self.use_actions:
            self.action_embed = nn.Linear(action_dim, d_action_embed)
            input_dim = ctx_dim + d_offset + d_action_embed
        else:
            input_dim = ctx_dim + d_offset
        
        # Deeper Residual MLP
        hidden_dim = 2 * latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, h_ctx, k, a_pred=None):
        """
        Args:
            h_ctx: (B, ctx_dim) - context representation
            k: int or tensor of shape (B,) - prediction offset(s)
            a_pred: (B, k, action_dim) or None - future actions to condition on
        Returns:
            z_hat: (B, latent_dim) - predicted latent
        """
        if isinstance(k, int):
            k_tensor = torch.full((h_ctx.shape[0],), k, device=h_ctx.device, dtype=torch.long)
        else:
            k_tensor = k
        
        # Embed offset
        e_k = self.offset_embed(k_tensor)  # (B, d_offset)
        
        # Build input list
        inputs = [h_ctx, e_k]
        
        if self.use_actions and a_pred is not None:
            # For k-step prediction, we'll use the first action in the sequence
            # This is a simplification but enough to make the paddle move!
            
            if a_pred.ndim == 3:
                # Take the first action (immediate next move)
                action = a_pred[:, 0, :] 
            else:
                action = a_pred
                
            a_emb = self.action_embed(action) # (B, d_action_embed)
            inputs.append(a_emb)
        elif self.use_actions and a_pred is None:
            # If we expect actions but none provided (e.g. inference without action plan)
            # Use zero embedding
            B = h_ctx.shape[0]
            a_zeros = torch.zeros(B, self.action_dim, device=h_ctx.device)
            a_emb = self.action_embed(a_zeros)
            inputs.append(a_emb)
        
        # Concatenate context, offset, and action
        h = torch.cat(inputs, dim=-1)
        
        # Predict latent
        z_hat = self.net(h)  # (B, latent_dim)
        
        return z_hat


class ProjectionHead(nn.Module):
    """Projection head for JEPA loss."""
    
    def __init__(self, input_dim, proj_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


def variance_loss(embeddings, eps=1e-4):
    """
    Prevent collapse by ensuring embeddings have high variance.
    Based on VICReg regularization.
    
    Args:
        embeddings: (B, D) tensor of embeddings
        eps: small constant for numerical stability
    
    Returns:
        loss: scalar tensor penalizing low variance
    """
    B = embeddings.shape[0]
    if B <= 1:
        # Can't compute variance with batch size <= 1
        return torch.tensor(0.0, device=embeddings.device)
    
    # Use unbiased=False to avoid issues with small batches
    std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + eps)
    return torch.mean(torch.relu(1 - std))  # Penalize if std < 1


def covariance_loss(embeddings):
    """
    Decorrelate dimensions to prevent redundancy.
    Simplified VICReg-style covariance regularization.
    
    Args:
        embeddings: (B, D) tensor of embeddings
    
    Returns:
        loss: scalar tensor penalizing correlated dimensions
    """
    B, D = embeddings.shape
    if B <= 1:
        # Can't compute covariance with batch size <= 1
        return torch.tensor(0.0, device=embeddings.device)
    
    # Center the embeddings
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov = (embeddings.T @ embeddings) / (B - 1)
    
    # Extract off-diagonal elements (simpler approach)
    # Sum of squared off-diagonal elements, normalized by D
    cov_off_diag = cov - torch.diag(torch.diagonal(cov))
    return cov_off_diag.pow(2).sum() / D


class JEPAWorldModel(nn.Module):
    """Joint-Embedding Predictive Architecture world model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = ConvEncoder(
            in_channels=config.get('in_channels', 3),
            latent_dim=config['latent_dim'],
            channels=config.get('encoder_channels', [32, 64, 128, 256])
        )
        
        # Target encoder (EMA copy)
        self.encoder_ema = ConvEncoder(
            in_channels=config.get('in_channels', 3),
            latent_dim=config['latent_dim'],
            channels=config.get('encoder_channels', [32, 64, 128, 256])
        )
        
        # Initialize EMA encoder as copy
        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        
        # Freeze EMA encoder (no gradients)
        for param in self.encoder_ema.parameters():
            param.requires_grad = False
        
        # Context encoder
        ctx_dim = config['latent_dim'] + (config.get('d_action_embed', 64) 
                                          if config.get('action_dim') is not None 
                                          else 0)
        self.ctx_encoder = ContextEncoder(
            latent_dim=config['latent_dim'],
            action_dim=config.get('action_dim'),
            d_action_embed=config.get('d_action_embed', 64),
            n_layers=config.get('n_layers_ctx', 3),
            n_heads=config.get('n_heads_ctx', 8),
            d_ff_scale=config.get('d_ff_scale', 4)
        )
        
        # Predictor
        self.predictor = Predictor(
            ctx_dim=ctx_dim,
            latent_dim=config['latent_dim'],
            d_offset=config.get('d_offset', 64),
            max_offset=config.get('max_offset', 10),
            action_dim=config.get('action_dim'),
            d_action_embed=config.get('d_action_embed', 64)
        )
        
        # Projection heads
        self.proj_pred = ProjectionHead(
            input_dim=config['latent_dim'],
            proj_dim=config.get('proj_dim', 256)
        )
        self.proj_targ = ProjectionHead(
            input_dim=config['latent_dim'],
            proj_dim=config.get('proj_dim', 256)
        )
        
        # EMA decay rate
        self.tau_ema = config.get('tau_ema', 0.99)
    
    @torch.no_grad()
    def update_ema(self):
        """Update EMA encoder parameters."""
        for param_ema, param in zip(self.encoder_ema.parameters(), 
                                     self.encoder.parameters()):
            param_ema.data = (self.tau_ema * param_ema.data + 
                             (1 - self.tau_ema) * param.data)
    
    def encode_online(self, x):
        """Encode with online encoder."""
        return self.encoder(x)
    
    @torch.no_grad()
    def encode_target(self, x):
        """Encode with target (EMA) encoder."""
        return self.encoder_ema(x)
    
    def forward_step(self, x_seq, a_seq, t, prediction_offsets, normalize_proj=True):
        B, T, C, H, W = x_seq.shape
        K_ctx = self.config['K_ctx']
        
        # ---- 1. Encode CONTEXT ONLY (online encoder) ----
        z_ctx = []
        for i in range(t-K_ctx+1, t+1):
            z_ctx.append(self.encoder(x_seq[:, i]))  # (B, D)
        z_ctx = torch.stack(z_ctx, dim=1)  # (B, K_ctx, D)
        
        # Encode ACTIONS if present
        if a_seq is not None:
            A_ctx = a_seq[:, t-K_ctx+1:t, :]  # (B, K_ctx-1, action_dim)
        else:
            A_ctx = None
        
        # ---- 2. Encode FUTURE TARGETS ONLY (EMA encoder, stop-grad) ----
        z_tgt = {}
        with torch.no_grad():
            for k in prediction_offsets:
                z_tgt[k] = self.encoder_ema(x_seq[:, t+k])  # (B, D)
        
        # ---- 3. Get context representation ----
        h_ctx = self.ctx_encoder(z_ctx, A_ctx)  # (B, ctx_dim)
        
        # ---- 4. Predictor + JEPA loss ----
        losses = {}
        final_losses = []
        
        for k in prediction_offsets:
            # Get future actions for conditioning
            if a_seq is not None:
                # Actions from t to t+k-1
                # We need k actions to predict k steps ahead
                a_pred = a_seq[:, t:t+k, :]
            else:
                a_pred = None
                
            z_hat = self.predictor(h_ctx, k, a_pred)  # (B, latent_dim)
            y_pred = self.proj_pred(z_hat)            # (B, proj_dim)
            y_targ = self.proj_targ(z_tgt[k])         # (B, proj_dim)
            
            # Check for NaN in projections
            if torch.isnan(y_pred).any() or torch.isnan(y_targ).any():
                print(f"Warning: NaN detected in projections at k={k}")
                print(f"  y_pred has NaN: {torch.isnan(y_pred).any()}")
                print(f"  y_targ has NaN: {torch.isnan(y_targ).any()}")
                continue
            
            # CRITICAL: Apply variance AND covariance loss to PREDICTIONS ONLY
            # Don't regularize EMA targets (they're stop-grad anyway)
            var_loss_latent_pred = variance_loss(z_hat)
            cov_loss_latent_pred = covariance_loss(z_hat)
            
            # Also apply to predicted projections only
            var_loss_proj_pred = variance_loss(y_pred)
            cov_loss_proj_pred = covariance_loss(y_pred)
            
            if normalize_proj:
                eps = 1e-6
                y_pred_norm = F.normalize(y_pred, p=2, dim=-1, eps=eps)
                y_targ_norm = F.normalize(y_targ, p=2, dim=-1, eps=eps)
            else:
                y_pred_norm = y_pred
                y_targ_norm = y_targ
            
            # MSE loss on normalized embeddings
            loss_mse = F.mse_loss(y_pred_norm, y_targ_norm)
            
            # Regularization: minimal weights - just enough to prevent collapse
            # VICReg-style: variance prevents collapse, covariance prevents redundancy
            var_weight_latent = 0.01  # Very small - MSE should be primary signal
            cov_weight_latent = 0.001 # Covariance is per-dimension so very small
            var_weight_proj = 0.005   # Minimal on projections
            cov_weight_proj = 0.0005  # Tiny covariance weight
            
            total_reg_loss = (var_weight_latent * var_loss_latent_pred + 
                             cov_weight_latent * cov_loss_latent_pred +
                             var_weight_proj * var_loss_proj_pred +
                             cov_weight_proj * cov_loss_proj_pred)
            
            loss_k = loss_mse + total_reg_loss
            
            # Safety check
            if torch.isnan(loss_k) or torch.isinf(loss_k):
                print(f"Warning: Invalid loss at k={k}: {loss_k.item()}")
                print(f"  loss_mse: {loss_mse.item()}")
                print(f"  total_reg_loss: {total_reg_loss.item()}")
                continue
            
            losses[f"loss_k{k}"] = loss_k.item()
            losses[f"mse_k{k}"] = loss_mse.item()
            losses[f"var_latent_k{k}"] = var_loss_latent_pred.item()
            losses[f"cov_latent_k{k}"] = cov_loss_latent_pred.item()
            losses[f"var_proj_k{k}"] = var_loss_proj_pred.item()
            losses[f"cov_proj_k{k}"] = cov_loss_proj_pred.item()
            final_losses.append(loss_k)
        
        if len(final_losses) == 0:
            print("Warning: All prediction offsets produced invalid losses!")
            # Create a dummy loss that's connected to the model parameters
            # Use a small term from the encoder to maintain gradient flow
            dummy_loss = 0.0 * self.encoder.fc.weight.sum()
            return dummy_loss, losses
        
        loss = sum(final_losses) / len(final_losses)

        # ---- 5. Return ----
        losses['loss'] = loss.item()
        return loss, losses

    def forward_batch(self, x_seq, a_seq, prediction_offsets, sample_times=True):
        """
        Forward pass over batch, optionally sampling multiple time steps.
        
        Args:
            x_seq: (B, T, C, H, W) - sequence of observations
            a_seq: (B, T-1, action_dim) or None - sequence of actions
            prediction_offsets: list of ints - offsets to predict
            sample_times: bool - if True, sample one t per sequence; if False, use t = K_ctx
        Returns:
            loss: scalar tensor
            diagnostics: dict
        """
        B, T, C, H, W = x_seq.shape
        K_ctx = self.config['K_ctx']
        max_offset = max(prediction_offsets)
        
        # Valid time range: [K_ctx, T - max_offset)
        t_min = K_ctx
        t_max = T - max_offset
        
        if sample_times:
            # Sample one time step per sequence
            t_values = torch.randint(t_min, t_max, (B,), device=x_seq.device)
            
            # Build batch by extracting the sampled timestep from each sequence
            # This allows proper variance/covariance computation across batch
            z_ctx_list = []
            z_tgt_dict = {k: [] for k in prediction_offsets}
            a_ctx_list = [] if a_seq is not None else None
            a_pred_dict = {k: [] for k in prediction_offsets} if a_seq is not None else None
            
            for b in range(B):
                t = t_values[b].item()
                
                # Encode context
                z_ctx_b = []
                for i in range(t-K_ctx+1, t+1):
                    z_ctx_b.append(self.encoder(x_seq[b:b+1, i]))
                z_ctx_list.append(torch.stack(z_ctx_b, dim=1))  # (1, K_ctx, D)
                
                # Encode targets (EMA encoder)
                with torch.no_grad():
                    for k in prediction_offsets:
                        z_tgt_dict[k].append(self.encoder_ema(x_seq[b:b+1, t+k]))
                
                # Get actions if present
                if a_seq is not None:
                    # Context actions: t-K_ctx+1 to t-1
                    a_ctx_list.append(a_seq[b:b+1, t-K_ctx+1:t, :])
                    
                    # Future actions: t to t+k-1
                    for k in prediction_offsets:
                        a_pred_dict[k].append(a_seq[b:b+1, t:t+k, :])
            
            # Stack into batches
            z_ctx_batch = torch.cat(z_ctx_list, dim=0)  # (B, K_ctx, D)
            z_tgt_batch = {k: torch.cat(z_tgt_dict[k], dim=0) for k in prediction_offsets}  # (B, D) for each k
            a_ctx_batch = torch.cat(a_ctx_list, dim=0) if a_ctx_list is not None else None
            a_pred_batch = {k: torch.cat(a_pred_dict[k], dim=0) for k in prediction_offsets} if a_pred_dict is not None else None
            
            # Get context representation
            h_ctx = self.ctx_encoder(z_ctx_batch, a_ctx_batch)  # (B, ctx_dim)
            
            # Compute loss with regularization on full batch
            losses = {}
            final_losses = []
            
            for k in prediction_offsets:
                a_pred = a_pred_batch[k] if a_pred_batch is not None else None
                z_hat = self.predictor(h_ctx, k, a_pred)  # (B, latent_dim)
                y_pred = self.proj_pred(z_hat)     # (B, proj_dim)
                y_targ = self.proj_targ(z_tgt_batch[k])  # (B, proj_dim)
                
                # Regularization losses on PREDICTIONS ONLY (not EMA targets)
                var_loss_latent_pred = variance_loss(z_hat)
                cov_loss_latent_pred = covariance_loss(z_hat)
                
                var_loss_proj_pred = variance_loss(y_pred)
                cov_loss_proj_pred = covariance_loss(y_pred)
                
                # Normalize projections
                if self.config.get('normalize_proj', True):
                    eps = 1e-6
                    y_pred_norm = F.normalize(y_pred, p=2, dim=-1, eps=eps)
                    y_targ_norm = F.normalize(y_targ, p=2, dim=-1, eps=eps)
                else:
                    y_pred_norm = y_pred
                    y_targ_norm = y_targ
                
                # MSE loss
                loss_mse = F.mse_loss(y_pred_norm, y_targ_norm)
                
                # Regularization: minimal weights - just enough to prevent collapse
                var_weight_latent = 0.01  # Very small - MSE should be primary signal
                cov_weight_latent = 0.001 # Covariance is per-dimension so very small
                var_weight_proj = 0.005
                cov_weight_proj = 0.0005
                
                total_reg_loss = (var_weight_latent * var_loss_latent_pred + 
                                 cov_weight_latent * cov_loss_latent_pred +
                                 var_weight_proj * var_loss_proj_pred +
                                 cov_weight_proj * cov_loss_proj_pred)
                
                loss_k = loss_mse + total_reg_loss
                
                # Store diagnostics
                losses[f"loss_k{k}"] = loss_k.item()
                losses[f"mse_k{k}"] = loss_mse.item()
                losses[f"var_latent_k{k}"] = var_loss_latent_pred.item()
                losses[f"cov_latent_k{k}"] = cov_loss_latent_pred.item()
                losses[f"var_proj_k{k}"] = var_loss_proj_pred.item()
                losses[f"cov_proj_k{k}"] = cov_loss_proj_pred.item()
                final_losses.append(loss_k)
            
            loss = sum(final_losses) / len(final_losses)
            losses['loss'] = loss.item()
            diagnostics = losses
        else:
            # Use fixed time step (e.g., middle of sequence)
            t = (t_min + t_max) // 2
            loss, diagnostics = self.forward_step(
                x_seq, a_seq, t, prediction_offsets
            )
        
        return loss, diagnostics