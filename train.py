import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
from collections import defaultdict

from jepa_world_model import JEPAWorldModel
from config import default_config
from trainer import JEPAWorldModelTrainer
from dataset import AtariDataset


def collate_fn(batch):
    """Custom collate function that handles None values."""
    from torch.utils.data._utils.collate import default_collate
    # Filter out None values from each item
    # If 'a' is None, don't include it in that item
    filtered_batch = []
    for item in batch:
        filtered_item = {k: v for k, v in item.items() if v is not None}
        filtered_batch.append(filtered_item)
    
    return default_collate(filtered_batch)


def plot_training_curves(history, save_path='./training_curves.png'):
    """Plot training curves from history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Progress', fontsize=16)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Training loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Per-offset losses
    for k in sorted(history.keys()):
        if k.startswith('train_loss_k'):
            offset = k.replace('train_loss_k', '')
            axes[0, 1].plot(epochs, history[k], label=f'k={offset}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss by Prediction Offset')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate (if available)
    if 'lr' in history and history['lr']:
        axes[1, 0].plot(epochs, history['lr'], 'g-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].grid(True)
    
    # Loss trend (smoothed)
    if len(history['train_loss']) > 1:
        window = min(5, len(history['train_loss']) // 2)
        smoothed = []
        for i in range(len(history['train_loss'])):
            start = max(0, i - window)
            end = min(len(history['train_loss']), i + window + 1)
            smoothed.append(sum(history['train_loss'][start:end]) / (end - start))
        axes[1, 1].plot(epochs, history['train_loss'], 'b-', alpha=0.3, label='Raw')
        axes[1, 1].plot(epochs, smoothed, 'b-', linewidth=2, label='Smoothed')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Loss Trend (Smoothed)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()


def train(config, num_epochs=100, save_dir='./checkpoints', device='cuda', 
          env_name='ALE/Pong-v5', num_episodes=100, val_split=0.2, plot=True, dataset_name=None):
    """Main training function."""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history for plotting
    history = defaultdict(list)
    
    # Create model
    print("Creating model...")
    model = JEPAWorldModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = JEPAWorldModelTrainer(model, config=config, device=device)
    
    # Create dataset and dataloader
    print(f"\nLoading Minari dataset...")
    dataset = AtariDataset(
        dataset_name=dataset_name,  # Will auto-search for env_name if None
        env_name=env_name,
        seq_length=config['K_ctx'] + max(config['prediction_offsets']) + 2,
        frame_size=84,
        use_actions=config.get('action_dim') is not None,
        max_episodes=num_episodes
    )
    
    # Update config with detected action_dim if needed
    if config.get('action_dim') is None and dataset.use_actions:
        config['action_dim'] = dataset.action_dim
        print(f"Auto-detected action_dim: {config['action_dim']}")
        # Recreate model with correct action_dim
        model = JEPAWorldModel(config)
        trainer = JEPAWorldModelTrainer(model, config=config, device=device)
    
    # Split into train/val sets
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Sequence length: {config['K_ctx'] + max(config['prediction_offsets']) + 2}")
    print(f"Prediction offsets: {config['prediction_offsets']}")
    print("-" * 60)
    
    best_loss = float('inf')
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_losses = []
        epoch_diagnostics = {}
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Training step
            loss, diagnostics = trainer.training_step(batch)
            epoch_losses.append(loss)
            
            # Accumulate diagnostics
            for k, v in diagnostics.items():
                if k not in epoch_diagnostics:
                    epoch_diagnostics[k] = []
                epoch_diagnostics[k].append(v)
            
            # Update progress bar
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg': f'{avg_loss:.4f}'
            })
            
        # Training epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_diagnostics = {k: sum(v)/len(v) for k, v in epoch_diagnostics.items()}
        
        # Store training metrics
        history['train_loss'].append(avg_loss)
        for k, v in avg_diagnostics.items():
            if k.startswith('loss_k'):
                history[f'train_{k}'].append(v)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_diagnostics = {}
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                x_seq = batch['x'].to(device)
                a_seq = batch.get('a')
                if a_seq is not None:
                    a_seq = a_seq.to(device)
                
                prediction_offsets = config.get('prediction_offsets', [1, 2, 3])
                loss, diagnostics = model.forward_batch(
                    x_seq, a_seq, prediction_offsets, sample_times=True
                )
                val_losses.append(loss.item())
                
                for k, v in diagnostics.items():
                    if k not in val_diagnostics:
                        val_diagnostics[k] = []
                    val_diagnostics[k].append(v)
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        avg_val_diagnostics = {k: sum(v)/len(v) for k, v in val_diagnostics.items()}
        
        # Store validation metrics
        history['val_loss'].append(avg_val_loss)
        for k, v in avg_val_diagnostics.items():
            if k.startswith('loss_k'):
                history[f'val_{k}'].append(v)
        
        # Learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} complete")
        print(f"  Total Loss:   {avg_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        # Show MSE (actual prediction quality) prominently
        avg_mse_train = avg_diagnostics.get('mse_k1', 0)
        avg_mse_val = avg_val_diagnostics.get('mse_k1', 0)
        print(f"  MSE (k=1):    {avg_mse_train:.6f} | Val: {avg_mse_val:.6f}  ← PREDICTION QUALITY")
        
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Show detailed breakdown for first offset
        if 'var_latent_k1' in avg_diagnostics:
            print(f"  Regularization breakdown (k=1):")
            print(f"    Var(latent): {avg_diagnostics.get('var_latent_k1', 0):.4f}")
            print(f"    Cov(latent): {avg_diagnostics.get('cov_latent_k1', 0):.4f}")
        
        print(f"{'='*60}\n")
        
        # Save metrics to JSON
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot every 5 epochs or at the end
        if plot and ((epoch + 1) % 5 == 0 or epoch == num_epochs - 1):
            plot_training_curves(history, os.path.join(save_dir, 'training_curves.png'))
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or avg_val_loss < best_val_loss:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': config,
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'diagnostics': avg_diagnostics,
                'history': dict(history),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model based on validation loss (prevents overfitting!)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(save_dir, 'best_checkpoint.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'config': config,
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'diagnostics': avg_diagnostics,
                    'history': dict(history),
                }, best_path)
                print(f"✨ New best model (val_loss={avg_val_loss:.4f}) saved to {best_path}")
    
    print("Training complete!")
    
    # Final plot
    if plot:
        plot_training_curves(history, os.path.join(save_dir, 'training_curves_final.png'))
        print(f"\n📊 Final training curves saved!")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Training Summary:")
    print(f"  Best Train Loss: {min(history['train_loss']):.4f}")
    print(f"  Best Val Loss: {min(history['val_loss']):.4f}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")
    
    # Check for overfitting
    if len(history['train_loss']) > 5:
        recent_train = sum(history['train_loss'][-5:]) / 5
        recent_val = sum(history['val_loss'][-5:]) / 5
        gap = recent_val - recent_train
        if gap > recent_train * 0.2:  # Val loss >20% higher than train
            print(f"\n⚠️  Warning: Possible overfitting detected!")
            print(f"   Train-Val gap: {gap:.4f} ({gap/recent_train*100:.1f}%)")
        else:
            print(f"\n✅ Training looks healthy (gap: {gap:.4f})")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train JEPA World Model on Atari')
    parser.add_argument('--env', type=str, default='ALE/Pong-v5', 
                       help='Atari environment name (default: ALE/Pong-v5)')
    parser.add_argument('--dataset-name', type=str, default=None,
                       help='Minari dataset name (e.g., "atari/pong/expert-v0"). If None, will search for env_name')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Maximum number of episodes to use from dataset (None = use all)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--no-actions', action='store_true',
                       help='Disable action conditioning')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    
    args = parser.parse_args()
    
    # Setup config
    config = default_config.copy()
    config['action_dim'] = None if args.no_actions else 6  # Will be auto-detected
    config['prediction_offsets'] = [1, 2, 3, 4, 5]  # Predict 5 steps ahead
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['in_channels'] = 3  # RGB frames
    
    # Device selection
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print("=" * 60)
    print("JEPA World Model Training")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {device}")
    print(f"Use actions: {not args.no_actions}")
    print("=" * 60)
    
    # Train
    train(
        config=config,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        device=device,
        env_name=args.env,
        num_episodes=args.episodes,
        val_split=args.val_split,
        plot=not args.no_plot,
        dataset_name=args.dataset_name
    )