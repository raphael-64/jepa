"""
Train a decoder to visualize JEPA predictions.
This decoder maps latent embeddings back to pixel space.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from pathlib import Path

from jepa_world_model import JEPAWorldModel
from decoder import ConvDecoder
from config import default_config
from dataset import AtariDataset


def collate_fn(batch):
    """Custom collate function that handles None values."""
    from torch.utils.data._utils.collate import default_collate
    # Filter out None values from each item
    filtered_batch = []
    for item in batch:
        filtered_item = {k: v for k, v in item.items() if v is not None}
        filtered_batch.append(filtered_item)
    
    return default_collate(filtered_batch)


def train_decoder(jepa_model, dataloader, num_epochs=20, device='cuda', lr=1e-3, 
                  save_path='./checkpoints/decoder.pt'):
    """
    Train a decoder to reconstruct frames from JEPA latents.
    
    Args:
        jepa_model: Trained JEPA model (encoder will be frozen)
        dataloader: DataLoader with sequences
        num_epochs: Number of training epochs
        device: Device to train on
        lr: Learning rate
        save_path: Where to save the trained decoder
    """
    # Create decoder
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
    
    print(f"Training decoder for {num_epochs} epochs...")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        decoder.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            x_seq = batch['x'].to(device)  # (B, T, C, H, W)
            B, T, C, H, W = x_seq.shape
            
            # Encode with frozen JEPA encoder
            with torch.no_grad():
                x_flat = x_seq.view(B * T, C, H, W)
                z = jepa_model.encode_online(x_flat)  # (B*T, latent_dim)
            
            # Decode
            x_recon = decoder(z)  # (B*T, C, H, W)
            
            # Reconstruction loss
            loss = criterion(x_recon, x_flat)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} complete - Avg Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(decoder.state_dict(), save_path)
            print(f"  ✨ Saved best decoder (loss: {best_loss:.6f})")
    
    print(f"\n✅ Decoder training complete!")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Saved to: {save_path}")
    
    return decoder


def main():
    parser = argparse.ArgumentParser(description='Train decoder for JEPA visualization')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_checkpoint.pt',
                        help='Path to trained JEPA checkpoint')
    parser.add_argument('--env', type=str, default='ALE/Pong-v5',
                        help='Environment name')
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='Maximum number of episodes to use (None = use all)')
    parser.add_argument('--dataset_name', type=str, default='atari/pong/expert-v0',
                        help='Minari dataset name (e.g., "atari/pong/expert-v0"). If None, will search for env_name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs('./checkpoints', exist_ok=True)
    
    # Load trained JEPA model
    print(f"Loading JEPA model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Reconstruct config from checkpoint
    config = checkpoint.get('config', default_config)
    if 'config' not in checkpoint:
        print("⚠️  No config in checkpoint, using default_config")
    
    # Create model
    model = JEPAWorldModel(config).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ JEPA model loaded")
    
    # Create dataset
    print(f"\nLoading Minari dataset...")
    dataset = AtariDataset(
        dataset_name=args.dataset_name,
        env_name=args.env,
        seq_length=10,
        frame_size=84,
        use_actions=False,  # Decoder doesn't need actions
        max_episodes=args.num_episodes
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False,
        collate_fn=collate_fn
    )
    
    print(f"Dataset: {len(dataset)} sequences")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Train decoder
    decoder = train_decoder(
        jepa_model=model,
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        device=args.device,
        lr=args.lr,
        save_path='./checkpoints/decoder.pt'
    )
    
    print("\n🎉 Done! You can now use visualize_predictions.py to see the results.")


if __name__ == '__main__':
    main()

