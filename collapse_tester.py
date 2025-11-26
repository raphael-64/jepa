"""
Check if encoder embeddings have collapsed.
"""
import torch
import numpy as np
from jepa_world_model import JEPAWorldModel
from config import default_config
from dataset import AtariDataset

def check_encoder_collapse(checkpoint_path='./checkpoints/best_checkpoint.pt'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print("Loading JEPA model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', default_config)
    model = JEPAWorldModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    encoder = model.encoder
    encoder.eval()
    
    # Load dataset
    print("Loading dataset...")
    dataset = AtariDataset(
        dataset_name='atari/pong/expert-v0',
        env_name='ALE/Pong-v5',
        seq_length=10,
        frame_size=84,
        use_actions=False,
        max_episodes=10
    )
    
    # Collect embeddings from diverse frames
    all_embeddings = []
    print("\nCollecting embeddings from 100 frames...")
    with torch.no_grad():
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            x_seq = sample['x'].to(device)  # (T, C, H, W)
            
            # Take first frame
            x = x_seq[0:1]
            z = encoder(x)
            all_embeddings.append(z.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)  # (100, latent_dim)
    
    print("\n" + "="*60)
    print("ENCODER EMBEDDING ANALYSIS")
    print("="*60)
    
    # Statistics
    mean_emb = all_embeddings.mean(dim=0)
    std_emb = all_embeddings.std(dim=0)
    
    print(f"\nEmbedding shape: {all_embeddings.shape}")
    print(f"Mean norm: {all_embeddings.norm(dim=1).mean():.4f}")
    print(f"Std of norms: {all_embeddings.norm(dim=1).std():.4f}")
    print(f"Mean per-dim std: {std_emb.mean():.6f}")
    print(f"Min per-dim std: {std_emb.min():.6f}")
    print(f"Max per-dim std: {std_emb.max():.6f}")
    print(f"Num dims with std < 0.01: {(std_emb < 0.01).sum().item()}/{len(std_emb)}")
    
    # Pairwise distances
    print("\nComputing pairwise distances...")
    sample_size = min(50, all_embeddings.shape[0])
    sampled = all_embeddings[:sample_size]
    distances = torch.cdist(sampled, sampled, p=2)
    mask = ~torch.eye(sample_size, dtype=bool)
    distances = distances[mask]
    
    print(f"Mean pairwise distance: {distances.mean():.6f}")
    print(f"Std pairwise distance: {distances.std():.6f}")
    print(f"Min pairwise distance: {distances.min():.6f}")
    print(f"Max pairwise distance: {distances.max():.6f}")
    
    # Check for collapse
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    collapsed = False
    if distances.mean() < 0.1:
        print("❌ SEVERE COLLAPSE: Pairwise distances < 0.1")
        print("   All embeddings are nearly identical!")
        collapsed = True
    elif distances.mean() < 1.0:
        print("⚠️  MODERATE COLLAPSE: Pairwise distances < 1.0")
        print("   Embeddings lack diversity")
        collapsed = True
    elif std_emb.mean() < 0.01:
        print("❌ VARIANCE COLLAPSE: Per-dimension variance < 0.01")
        print("   Encoder outputs don't vary across inputs")
        collapsed = True
    else:
        print("✅ Embeddings show good diversity")
    
    if collapsed:
        print("\n💡 SOLUTION: Your JEPA encoder collapsed during training.")
        print("   This is why reconstructions all look the same.")
        print("\n   To fix:")
        print("   1. Retrain JEPA with variance regularization")
        print("   2. Use smaller learning rate")
        print("   3. Add embedding normalization")
        print("   4. Check if EMA target encoder is updating properly")
    
    print("="*60)

if __name__ == '__main__':
    check_encoder_collapse()