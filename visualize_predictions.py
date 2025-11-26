"""
Visualize JEPA predictions by decoding them to pixel space.
Shows side-by-side comparisons of predicted vs actual frames.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import os
from pathlib import Path

from jepa_world_model import JEPAWorldModel
from decoder import ConvDecoder
from config import default_config
from dataset import AtariDataset


def denormalize_frame(frame):
    """Convert frame from [0, 1] to [0, 255] for display."""
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    # Convert from CHW to HWC
    if frame.shape[0] == 3:
        frame = frame.transpose(1, 2, 0)
    return frame


def visualize_predictions(jepa_model, decoder, dataset, num_samples=5, 
                          device='cuda', save_path='./checkpoints/predictions.png'):
    """
    Visualize JEPA predictions by decoding them to pixels.
    
    Args:
        jepa_model: Trained JEPA model
        decoder: Trained decoder
        dataset: Dataset to sample from
        num_samples: Number of sequences to visualize
        device: Device to use
        save_path: Where to save the visualization
    """
    jepa_model.eval()
    decoder.eval()
    
    # Sample random sequences
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    prediction_offsets = jepa_model.config.get('prediction_offsets', [1, 2, 3])
    K_ctx = jepa_model.config['K_ctx']
    
    # Create figure
    n_cols = 1 + len(prediction_offsets)  # Context + predictions
    fig = plt.figure(figsize=(4 * n_cols, 4 * num_samples))
    gs = gridspec.GridSpec(num_samples, n_cols, figure=fig, hspace=0.3, wspace=0.2)
    
    with torch.no_grad():
        for sample_idx, dataset_idx in enumerate(indices):
            # Get sequence
            sample = dataset[dataset_idx]
            x_seq = sample['x'].unsqueeze(0).to(device)  # (1, T, C, H, W)
            a_seq = sample.get('a')
            if a_seq is not None:
                a_seq = a_seq.unsqueeze(0).to(device)  # (1, T-1, action_dim)
            
            B, T, C, H, W = x_seq.shape
            
            # Choose prediction time step (need at least K_ctx frames before end)
            min_t = K_ctx
            max_t = T - max(prediction_offsets) - 1
            if max_t <= min_t:
                max_t = min_t + 1
            t = np.random.randint(min_t, max_t)
            
            # Encode all frames
            x_flat = x_seq.view(B * T, C, H, W)
            z_all = jepa_model.encode_online(x_flat)  # (B*T, latent_dim)
            z_all = z_all.view(B, T, -1)
            
            # Get context
            Z_ctx = z_all[:, t - K_ctx + 1 : t + 1, :]  # (1, K_ctx, latent_dim)
            
            if a_seq is not None:
                A_ctx = a_seq[:, t - K_ctx + 1 : t, :]  # (1, K_ctx-1, action_dim)
            else:
                A_ctx = None
            
            # Get context representation
            h_ctx = jepa_model.ctx_encoder(Z_ctx, A_ctx)  # (1, ctx_dim)
            
            # Plot context frame (the last frame we're predicting from)
            ax = fig.add_subplot(gs[sample_idx, 0])
            context_frame = denormalize_frame(x_seq[0, t, :, :, :])
            ax.imshow(context_frame)
            ax.set_title(f'Sample {sample_idx+1}\nContext (t={t})', fontsize=10)
            ax.axis('off')
            
            # Predict and visualize for each offset
            for col_idx, k in enumerate(prediction_offsets, start=1):
                if t + k >= T:
                    continue
                
                # Predict future latent
                z_hat = jepa_model.predictor(h_ctx, k)  # (1, latent_dim)
                
                # Decode to pixel space
                x_pred = decoder(z_hat)  # (1, C, H, W)
                x_pred_frame = denormalize_frame(x_pred[0])
                
                # Get ground truth
                x_gt = x_seq[0, t + k, :, :, :].cpu()
                x_gt_frame = denormalize_frame(x_gt)
                
                # Create side-by-side comparison
                comparison = np.hstack([x_pred_frame, x_gt_frame])
                
                # Plot
                ax = fig.add_subplot(gs[sample_idx, col_idx])
                ax.imshow(comparison)
                ax.axvline(x=W-0.5, color='red', linewidth=2, linestyle='--')
                ax.text(W//4, H-5, 'Predicted', color='white', fontsize=8, 
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                ax.text(W + W//4, H-5, 'Actual', color='white', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                ax.set_title(f't+{k} steps', fontsize=10)
                ax.axis('off')
    
    plt.suptitle('JEPA Predictions: Predicted (left) vs Actual (right)', 
                 fontsize=14, y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    plt.close()


def analyze_embeddings(jepa_model, dataset, device='cuda'):
    """Analyze if embeddings are collapsing."""
    jepa_model.eval()
    
    print("\nAnalyzing embeddings...")
    
    all_embeddings = []
    with torch.no_grad():
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            x_seq = sample['x'].to(device)  # (T, C, H, W)
            T = x_seq.shape[0]
            
            # Encode
            z = jepa_model.encode_online(x_seq)  # (T, latent_dim)
            all_embeddings.append(z.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, latent_dim)
    
    # Compute statistics
    mean_emb = all_embeddings.mean(dim=0)
    std_emb = all_embeddings.std(dim=0)
    
    # Compute pairwise distances
    # Sample a subset for efficiency
    sample_size = min(1000, all_embeddings.shape[0])
    indices = torch.randperm(all_embeddings.shape[0])[:sample_size]
    sampled = all_embeddings[indices]
    
    # Compute pairwise L2 distances
    distances = torch.cdist(sampled, sampled, p=2)
    # Remove diagonal
    mask = ~torch.eye(sample_size, dtype=bool)
    distances = distances[mask]
    
    print(f"\nEmbedding Statistics:")
    print(f"  Shape: {all_embeddings.shape}")
    print(f"  Mean norm: {all_embeddings.norm(dim=1).mean():.4f}")
    print(f"  Std norm: {all_embeddings.norm(dim=1).std():.4f}")
    print(f"  Mean per-dimension std: {std_emb.mean():.4f}")
    print(f"  Min per-dimension std: {std_emb.min():.4f}")
    print(f"  Max per-dimension std: {std_emb.max():.4f}")
    print(f"\nPairwise Distance Statistics (sample of {sample_size}):")
    print(f"  Mean distance: {distances.mean():.4f}")
    print(f"  Std distance: {distances.std():.4f}")
    print(f"  Min distance: {distances.min():.4f}")
    print(f"  Max distance: {distances.max():.4f}")
    
    # Check for collapse
    if distances.mean() < 0.1:
        print(f"\n⚠️  WARNING: Very small pairwise distances ({distances.mean():.4f})")
        print(f"   This suggests embedding collapse!")
    elif std_emb.mean() < 0.01:
        print(f"\n⚠️  WARNING: Very low variance in embeddings ({std_emb.mean():.4f})")
        print(f"   This suggests embedding collapse!")
    else:
        print(f"\n✅ Embeddings show reasonable diversity")


def main():
    parser = argparse.ArgumentParser(description='Visualize JEPA predictions')
    parser.add_argument('--jepa_checkpoint', type=str, default='./checkpoints/best_checkpoint.pt',
                        help='Path to trained JEPA checkpoint')
    parser.add_argument('--decoder_checkpoint', type=str, default='./checkpoints/decoder.pt',
                        help='Path to trained decoder checkpoint')
    parser.add_argument('--env', type=str, default='ALE/Pong-v5',
                        help='Environment name')
    parser.add_argument('--num_episodes', type=int, default=50,
                        help='Maximum number of episodes to use (None = use all)')
    parser.add_argument('--dataset_name', type=str, default='atari/pong/expert-v0',
                        help='Minari dataset name (e.g., "atari/pong/expert-v0"). If None, will search for env_name')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of sequences to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--analyze', action='store_true',
                        help='Also analyze embeddings for collapse')
    
    args = parser.parse_args()
    
    # Load JEPA model
    print(f"Loading JEPA model from {args.jepa_checkpoint}...")
    checkpoint = torch.load(args.jepa_checkpoint, map_location=args.device)
    config = checkpoint.get('config', default_config)
    model = JEPAWorldModel(config).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ JEPA model loaded")
    
    # Load decoder
    print(f"\nLoading decoder from {args.decoder_checkpoint}...")
    decoder = ConvDecoder(
        latent_dim=config['latent_dim'],
        out_channels=config.get('in_channels', 3),
        img_size=84
    ).to(args.device)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location=args.device))
    decoder.eval()
    print("✅ Decoder loaded")
    
    # Create test dataset
    print(f"\nLoading Minari dataset...")
    # Need longer sequences for visualization: K_ctx + max prediction offset
    K_ctx = config.get('K_ctx', 10)
    max_offset = max(config.get('prediction_offsets', [1, 2, 3, 4, 5]))
    seq_length = K_ctx + max_offset + 2
    dataset = AtariDataset(
        dataset_name=args.dataset_name,
        env_name=args.env,
        seq_length=seq_length,
        frame_size=84,
        use_actions=config.get('action_dim') is not None,
        max_episodes=args.num_episodes
    )
    print(f"Test dataset: {len(dataset)} sequences")
    
    # Analyze embeddings if requested
    if args.analyze:
        analyze_embeddings(model, dataset, args.device)
    
    # Visualize predictions
    print(f"\nGenerating visualizations for {args.num_samples} samples...")
    visualize_predictions(
        jepa_model=model,
        decoder=decoder,
        dataset=dataset,
        num_samples=args.num_samples,
        device=args.device,
        save_path='./checkpoints/predictions.png'
    )
    
    print("\n🎉 Visualization complete!")


if __name__ == '__main__':
    main()

