"""Default configuration for JEPA world model."""

default_config = {
    # Architecture
    'in_channels': 3,
    'latent_dim': 512,
    'proj_dim': 256,
    'encoder_channels': [32, 64, 128, 256],
    
    # Context encoder
    'K_ctx': 10,  # Number of input frames (context window)
    'action_dim': None,  # Set to action dimension if using actions
    'd_action_embed': 64,
    'n_layers_ctx': 3,
    'n_heads_ctx': 8,
    'd_ff_scale': 4,
    
    # Predictor
    'd_offset': 64,
    'max_offset': 10,
    
    # Training
    'K_pred': 5,
    'prediction_offsets': [1, 2, 3, 4, 5],  # Predict 5 steps ahead
    'tau_ema': 0.99,
    'lr': 3e-4,
    'batch_size': 128,
    'normalize_proj': True,
}