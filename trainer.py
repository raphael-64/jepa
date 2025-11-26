import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class JEPAWorldModelTrainer:
    """Trainer for JEPA world model."""
    
    def __init__(self, model, optimizer=None, config=None, device='cuda'):
        self.model = model
        self.config = config or model.config
        self.device = device
        self.model.to(device)
        
        # Optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.get('lr', 3e-4),
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
        
        # Scheduler (optional)
        self.scheduler = None
        
        # Training state
        self.step = 0
    
    def training_step(self, batch):
        """
        Single training step.
        
        Args:
            batch: dict with keys:
                - 'x': (B, T, C, H, W) - observations
                - 'a': (B, T-1, action_dim) or None - actions
        Returns:
            loss: scalar tensor
            diagnostics: dict
        """
        x_seq = batch['x'].to(self.device)  # (B, T, C, H, W)
        a_seq = batch.get('a')
        if a_seq is not None:
            a_seq = a_seq.to(self.device)
        
        prediction_offsets = self.config.get('prediction_offsets', [1, 2, 3])
        
        # Forward pass
        loss, diagnostics = self.model.forward_batch(
            x_seq, 
            a_seq, 
            prediction_offsets,
            sample_times=True
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Optional gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # EMA update
        self.model.update_ema()
        
        self.step += 1
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item(), diagnostics
    
    def set_scheduler(self, scheduler):
        """Set learning rate scheduler."""
        self.scheduler = scheduler