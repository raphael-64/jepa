"""
Test encoding and decoding by comparing reconstructed images to originals.
No JEPA predictions - just encoder→decoder pipeline test.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from pathlib import Path

from jepa_world_model import JEPAWorldModel, ConvEncoder
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


def compute_reconstruction_metrics(original, reconstructed):
    """Compute metrics to evaluate reconstruction quality."""
    # MSE
    mse = torch.mean((original - reconstructed) ** 2).item()
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
    else:
        psnr = float('inf')
    
    # MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(original - reconstructed)).item()
    
    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae
    }


def visualize_reconstructions(encoder, decoder, dataset, num_samples=10, 
                               device='cuda', save_path='./checkpoints/reconstructions.png'):
    """
    Visualize encoder-decoder reconstructions.
    
    Args:
        encoder: The encoder
        decoder: Trained decoder
        dataset: Dataset to sample from
        num_samples: Number of frames to visualize
        device: Device to use
        save_path: Where to save the visualization
    """
    encoder.eval()
    decoder.eval()
    
    # Sample random frames
    all_originals = []
    all_reconstructed = []
    all_metrics = []
    
    with torch.no_grad():
        frames_collected = 0
        dataset_idx = 0
        
        while frames_collected < num_samples and dataset_idx < len(dataset):
            sample = dataset[dataset_idx]
            x_seq = sample['x'].to(device)  # (T, C, H, W)
            T = x_seq.shape[0]
            
            # Pick a random frame from this sequence
            t = np.random.randint(0, T)
            x_original = x_seq[t:t+1]  # (1, C, H, W)
            
            # Encode and decode
            z = encoder(x_original)
            x_reconstructed = decoder(z)
            
            # Compute metrics
            metrics = compute_reconstruction_metrics(x_original, x_reconstructed)
            
            all_originals.append(x_original)
            all_reconstructed.append(x_reconstructed)
            all_metrics.append(metrics)
            
            frames_collected += 1
            dataset_idx += 1
    
    # Create visualization
    fig = plt.figure(figsize=(8, 4 * num_samples))
    gs = gridspec.GridSpec(num_samples, 1, figure=fig, hspace=0.3)
    
    for idx in range(num_samples):
        ax = fig.add_subplot(gs[idx, 0])
        
        # Get frames
        original_frame = denormalize_frame(all_originals[idx][0])
        recon_frame = denormalize_frame(all_reconstructed[idx][0])
        
        # Create side-by-side comparison
        H, W = original_frame.shape[:2]
        comparison = np.hstack([original_frame, recon_frame])
        
        # Plot
        ax.imshow(comparison)
        ax.axvline(x=W-0.5, color='red', linewidth=2, linestyle='--')
        
        # Add labels
        ax.text(W//2, H-5, 'Original', color='white', fontsize=10, 
               ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        ax.text(W + W//2, H-5, 'Reconstructed', color='white', fontsize=10,
               ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Add metrics
        metrics = all_metrics[idx]
        title = f"Sample {idx+1} | MSE: {metrics['mse']:.6f} | PSNR: {metrics['psnr']:.2f} dB | MAE: {metrics['mae']:.6f}"
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    plt.suptitle('Encoder-Decoder Reconstruction Test: Original (left) vs Reconstructed (right)', 
                 fontsize=14, y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    plt.close()


def test_encode_decode(encoder, decoder, dataset, num_samples=100, device='cuda'):
    """
    Test encoding and decoding by comparing reconstructions to originals.
    
    Args:
        encoder: The encoder (just the ConvEncoder part)
        decoder: Trained decoder
        dataset: Dataset to sample from
        num_samples: Number of frames to test
        device: Device to use
    """
    encoder.eval()
    decoder.eval()
    
    all_metrics = []
    
    print(f"Testing encode-decode on {num_samples} frames...")
    print("-" * 60)
    
    with torch.no_grad():
        frames_tested = 0
        dataset_idx = 0
        
        while frames_tested < num_samples and dataset_idx < len(dataset):
            # Get sequence
            sample = dataset[dataset_idx]
            x_seq = sample['x'].to(device)  # (T, C, H, W)
            T, C, H, W = x_seq.shape
            
            # Test all frames in this sequence
            for t in range(T):
                if frames_tested >= num_samples:
                    break
                
                # Get original frame
                x_original = x_seq[t:t+1]  # (1, C, H, W)
                
                # Encode
                z = encoder(x_original)  # (1, latent_dim)
                
                # Decode
                x_reconstructed = decoder(z)  # (1, C, H, W)
                
                # Compute metrics
                metrics = compute_reconstruction_metrics(x_original, x_reconstructed)
                all_metrics.append(metrics)
                
                frames_tested += 1
                
                # Print progress every 10 frames
                if frames_tested % 10 == 0 or frames_tested == 1:
                    print(f"Frame {frames_tested}/{num_samples}: "
                          f"MSE={metrics['mse']:.6f}, "
                          f"PSNR={metrics['psnr']:.2f} dB, "
                          f"MAE={metrics['mae']:.6f}")
            
            dataset_idx += 1
    
    # Compute aggregate statistics
    print("\n" + "=" * 60)
    print("RECONSTRUCTION STATISTICS")
    print("=" * 60)
    
    mse_values = [m['mse'] for m in all_metrics]
    psnr_values = [m['psnr'] for m in all_metrics if m['psnr'] != float('inf')]
    mae_values = [m['mae'] for m in all_metrics]
    
    print(f"\nMean Squared Error (MSE):")
    print(f"  Mean: {np.mean(mse_values):.6f}")
    print(f"  Std:  {np.std(mse_values):.6f}")
    print(f"  Min:  {np.min(mse_values):.6f}")
    print(f"  Max:  {np.max(mse_values):.6f}")
    
    print(f"\nPeak Signal-to-Noise Ratio (PSNR):")
    if psnr_values:
        print(f"  Mean: {np.mean(psnr_values):.2f} dB")
        print(f"  Std:  {np.std(psnr_values):.2f} dB")
        print(f"  Min:  {np.min(psnr_values):.2f} dB")
        print(f"  Max:  {np.max(psnr_values):.2f} dB")
    
    print(f"\nMean Absolute Error (MAE):")
    print(f"  Mean: {np.mean(mae_values):.6f}")
    print(f"  Std:  {np.std(mae_values):.6f}")
    print(f"  Min:  {np.min(mae_values):.6f}")
    print(f"  Max:  {np.max(mae_values):.6f}")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    mean_mse = np.mean(mse_values)
    mean_psnr = np.mean(psnr_values) if psnr_values else 0
    
    if mean_mse < 0.001:
        print("✅ EXCELLENT: Very low reconstruction error (MSE < 0.001)")
        print("   Encoder→Decoder pipeline is working very well.")
    elif mean_mse < 0.01:
        print("✅ GOOD: Low reconstruction error (MSE < 0.01)")
        print("   Encoder→Decoder pipeline is working well.")
    elif mean_mse < 0.05:
        print("⚠️  MODERATE: Moderate reconstruction error (MSE < 0.05)")
        print("   Encoder→Decoder pipeline is working but could be improved.")
    else:
        print("❌ POOR: High reconstruction error (MSE >= 0.05)")
        print("   Encoder→Decoder pipeline may have issues.")
    
    if mean_psnr > 30:
        print(f"✅ PSNR is good (>{30} dB) - high quality reconstruction")
    elif mean_psnr > 20:
        print(f"⚠️  PSNR is moderate ({mean_psnr:.1f} dB) - acceptable reconstruction")
    else:
        print(f"❌ PSNR is low ({mean_psnr:.1f} dB) - poor reconstruction")
    
    print("\nNote: Images are normalized to [0, 1] range.")
    print("=" * 60)
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Test encoder-decoder reconstruction')
    parser.add_argument('--jepa_checkpoint', type=str, default='./checkpoints/best_checkpoint.pt',
                        help='Path to JEPA checkpoint (to load encoder)')
    parser.add_argument('--decoder_checkpoint', type=str, default='./checkpoints/decoder.pt',
                        help='Path to trained decoder checkpoint')
    parser.add_argument('--env', type=str, default='ALE/Pong-v5',
                        help='Environment name')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Maximum number of episodes to use')
    parser.add_argument('--dataset_name', type=str, default='atari/pong/expert-v0',
                        help='Minari dataset name')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of frames to test for metrics')
    parser.add_argument('--num_visualize', type=int, default=10,
                        help='Number of frames to visualize')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load JEPA checkpoint to get the encoder
    print(f"Loading encoder from JEPA checkpoint: {args.jepa_checkpoint}...")
    checkpoint = torch.load(args.jepa_checkpoint, map_location=args.device)
    config = checkpoint.get('config', default_config)
    
    # We only need the encoder, not the full JEPA model
    # But it's easiest to just load the full model and extract the encoder
    full_model = JEPAWorldModel(config).to(args.device)
    full_model.load_state_dict(checkpoint['model_state_dict'])
    encoder = full_model.encoder  # Extract just the encoder
    encoder.eval()
    print("✅ Encoder loaded")
    
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
    dataset = AtariDataset(
        dataset_name=args.dataset_name,
        env_name=args.env,
        seq_length=10,
        frame_size=84,
        use_actions=False,  # We don't need actions for encode-decode test
        max_episodes=args.num_episodes
    )
    print(f"Test dataset: {len(dataset)} sequences")
    
    # Visualize reconstructions
    print(f"\nGenerating visualization for {args.num_visualize} samples...")
    visualize_reconstructions(
        encoder=encoder,
        decoder=decoder,
        dataset=dataset,
        num_samples=args.num_visualize,
        device=args.device,
        save_path='./checkpoints/reconstructions.png'
    )
    
    # Test encoding and decoding
    print(f"\nTesting encode-decode reconstruction on {args.num_samples} frames...\n")
    test_encode_decode(
        encoder=encoder,
        decoder=decoder,
        dataset=dataset,
        num_samples=args.num_samples,
        device=args.device
    )
    
    print("\n🎉 Testing complete!")


if __name__ == '__main__':
    main()