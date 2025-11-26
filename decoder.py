"""
Simple decoder to visualize JEPA predictions.
Maps latent vectors back to pixel frames.
"""
import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    """Decoder: maps latent z_t → reconstructed frame x_t."""
    
    def __init__(self, latent_dim=512, out_channels=3, img_size=84, channels=[256, 128, 64, 32]):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Start from 4x4 spatial size (matching encoder's adaptive pool)
        self.start_size = 4
        self.start_channels = channels[0]
        
        # Linear to get to starting spatial size
        self.fc = nn.Linear(latent_dim, self.start_channels * self.start_size * self.start_size)
        
        # Transpose convolutions to upsample
        layers = []
        in_ch = self.start_channels
        for out_ch in channels[1:]:
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.GELU()
            ])
            in_ch = out_ch
        
        self.deconv_blocks = nn.Sequential(*layers)
        
        # Final layer to output channels
        # After deconv_blocks: 4 -> 8 -> 16 -> 32 -> 64
        # We need to get to img_size (84), so we'll use interpolation
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        self.img_size = img_size
    
    def forward(self, z):
        """
        Args:
            z: (B, latent_dim) - latent vector
        Returns:
            x_recon: (B, C, H, W) - reconstructed frame
        """
        # Linear projection
        h = self.fc(z)  # (B, start_channels * 4 * 4)
        h = h.view(-1, self.start_channels, self.start_size, self.start_size)
        
        # Transpose convolutions
        h = self.deconv_blocks(h)
        
        # Final output
        x_recon = self.final_conv(h)  # (B, C, H, W) - might be 64x64
        
        # Interpolate to exact target size if needed
        if x_recon.shape[-1] != self.img_size:
            x_recon = nn.functional.interpolate(
                x_recon, size=(self.img_size, self.img_size), 
                mode='bilinear', align_corners=False
            )
        
        return x_recon


def train_decoder(jepa_model, dataloader, num_epochs=10, device='cuda', lr=1e-3):
    """
    Train a decoder to reconstruct frames from JEPA latents.
    
    This lets you visualize what the model is learning.
    """
    decoder = ConvDecoder(
        latent_dim=jepa_model.config['latent_dim'],
        out_channels=jepa_model.config.get('in_channels', 3),
        img_size=84
    ).to(device)
    
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Freeze JEPA encoder
    jepa_model.encoder.eval()
    for param in jepa_model.encoder.parameters():
        param.requires_grad = False
    
    print("Training decoder...")
    for epoch in range(num_epochs):
        decoder.train()
        total_loss = 0
        
        for batch in dataloader:
            x_seq = batch['x'].to(device)  # (B, T, C, H, W)
            B, T, C, H, W = x_seq.shape
            
            # Encode with frozen JEPA encoder
            with torch.no_grad():
                x_flat = x_seq.view(B * T, C, H, W)
                z = jepa_model.encode_online(x_flat)  # (B*T, latent_dim)
            
            # Decode
            x_recon = decoder(z)  # (B*T, C, H, W)
            x_recon = x_recon.view(B, T, C, H, W)
            
            # Reconstruction loss
            loss = criterion(x_recon, x_seq)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return decoder


def visualize_predictions(jepa_model, decoder, x_seq, a_seq, t, device='cuda'):
    """
    Visualize JEPA predictions by decoding them to pixels.
    
    Args:
        jepa_model: Trained JEPA model
        decoder: Trained decoder
        x_seq: (1, T, C, H, W) - input sequence
        a_seq: (1, T-1, action_dim) - actions
        t: time step to predict from
    """
    jepa_model.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Get context and predict
        x_seq = x_seq.to(device)
        if a_seq is not None:
            a_seq = a_seq.to(device)
        
        prediction_offsets = jepa_model.config.get('prediction_offsets', [1, 2, 3])
        
        # Encode input frames
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        z_all = jepa_model.encode_online(x_flat)
        z_all = z_all.view(B, T, -1)
        
        # Get context
        K_ctx = jepa_model.config['K_ctx']
        Z_ctx = z_all[:, t - K_ctx + 1 : t + 1, :]
        
        if a_seq is not None:
            A_ctx = a_seq[:, t - K_ctx + 1 : t, :]
        else:
            A_ctx = None
        
        # Get context representation
        h_ctx = jepa_model.ctx_encoder(Z_ctx, A_ctx)
        
        # Predict future latents
        predictions = {}
        for k in prediction_offsets:
            z_hat = jepa_model.predictor(h_ctx, k)  # (1, latent_dim)
            x_pred = decoder(z_hat)  # (1, C, H, W)
            predictions[f't+{k}'] = x_pred.cpu()
        
        # Get ground truth frames
        ground_truth = {}
        for k in prediction_offsets:
            if t + k < T:
                ground_truth[f't+{k}'] = x_seq[:, t + k, :, :, :].cpu()
        
        return predictions, ground_truth

