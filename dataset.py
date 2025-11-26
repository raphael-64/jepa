"""
Atari dataset using Minari pre-recorded datasets.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import minari


class AtariDataset(Dataset):
    """Atari dataset for JEPA training using Minari."""
    
    def __init__(self, dataset_name=None, env_name='ALE/Pong-v5', seq_length=10, 
                 frame_size=84, use_actions=True, max_episodes=None):
        """
        Args:
            dataset_name: Minari dataset name (e.g., 'atari/pong/expert-v0'). 
                         If None, will try to find a dataset for env_name.
            env_name: Environment name (used if dataset_name is None)
            seq_length: Length of sequences to sample
            frame_size: Size to resize frames to (e.g., 84)
            use_actions: Whether to include actions
            max_episodes: Maximum number of episodes to use (None = use all)
        """
        self.seq_length = seq_length
        self.frame_size = frame_size
        self.use_actions = use_actions
        
        # Load Minari dataset
        if dataset_name is None:
            # Try to find dataset for this environment
            print(f"Searching for Minari dataset for {env_name}...")
            try:
                available = minari.list_remote_datasets()
            except Exception as e:
                print(f"Could not list remote datasets: {e}")
                available = []
            
            # Look for matching dataset
            dataset_name = None
            env_short = env_name.replace('ALE/', '').replace('-v5', '').lower()
            for ds in available:
                if env_short in ds.lower():
                    dataset_name = ds
                    break
            
            if dataset_name is None:
                raise ValueError(
                    f"Could not find Minari dataset for {env_name}. "
                    f"Available datasets: {available[:10]}...\n"
                    f"Please specify dataset_name explicitly or download a dataset first:\n"
                    f"  minari download <dataset_name>"
                )
        
        print(f"Loading Minari dataset: {dataset_name}")
        try:
            self.minari_dataset = minari.load_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            print("\nTo download a dataset, run:")
            print(f"  minari download {dataset_name}")
            raise
        
        print(f"Dataset loaded: {self.minari_dataset.total_episodes} episodes, {self.minari_dataset.total_steps} steps")
        
        # Extract episodes
        self.episodes = []
        self.action_dim = None
        
        num_episodes = (
            self.minari_dataset.total_episodes
            if max_episodes is None
            else min(max_episodes, self.minari_dataset.total_episodes)
        )
        
        # Iterate through episodes using Minari API
        episode_count = 0
        for episode in self.minari_dataset:
            if episode_count >= num_episodes:
                break
            
            if (episode_count + 1) % 100 == 0:
                print(f"  Processing {episode_count + 1}/{num_episodes} episodes...")
            
            # Extract observations and actions from EpisodeData
            # Observations are numpy arrays: Box(0, 255, (210, 160, 3), uint8)
            observations = episode.observations
            actions = episode.actions if self.use_actions else None
            
            # Infer action space info once from first episode with actions
            if self.use_actions and actions is not None and self.action_dim is None:
                if isinstance(actions, np.ndarray):
                    if actions.ndim == 1:
                        # Discrete actions - get max value + 1
                        self.action_dim = int(actions.max() + 1) if len(actions) > 0 else 6
                    else:
                        # e.g., continuous or multi-dimensional actions (T, A)
                        self.action_dim = actions.shape[-1]
            
            # Preprocess frames
            frames = [self._preprocess_frame(obs) for obs in observations]
            
            if len(frames) >= self.seq_length:
                self.episodes.append({
                    'frames': np.stack(frames),
                    'actions': np.array(actions) if self.use_actions and actions is not None else None
                })
            
            episode_count += 1
        
        # Fallbacks
        if self.action_dim is None and self.use_actions:
            # Reasonable default for Atari discrete actions
            self.action_dim = 6
        
        if len(self.episodes) == 0:
            raise RuntimeError(
                "No episodes with at least seq_length frames were found in the dataset. "
                "Try lowering seq_length or checking the dataset."
            )
        
        print(f"Dataset ready: {len(self.episodes)} episodes, action_dim={self.action_dim}")
    
    def _preprocess_frame(self, frame):
        """Preprocess frame: resize to frame_size x frame_size, normalize."""
        # Minari observations are Box(0, 255, (210, 160, 3), uint8)
        # Convert to numpy if needed
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # Handle different shapes
        if len(frame.shape) == 3:
            if frame.shape[2] == 3:
                # RGB - resize
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            elif frame.shape[2] == 4:
                # RGBA - convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            else:
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
        elif len(frame.shape) == 2:
            # Grayscale
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            # Convert to 3-channel (repeat channels)
            frame = np.stack([frame] * 3, axis=-1)
        else:
            # Handle other formats
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            if len(frame.shape) == 2:
                frame = np.stack([frame] * 3, axis=-1)
        
        # Normalize to [0, 1] (frames are uint8 0-255)
        if frame.max() > 1.0:
            frame = frame.astype(np.float32) / 255.0
        else:
            frame = frame.astype(np.float32)
        
        return frame
    
    def __len__(self):
        # Return multiple samples per episode
        return len(self.episodes) * 50  # 50 random samples per episode
    
    def __getitem__(self, idx):
        ep_idx = idx % len(self.episodes)
        episode = self.episodes[ep_idx]
        
        frames = episode['frames']
        actions = episode['actions']
        
        # Sample random start position
        max_start = len(frames) - self.seq_length
        if max_start < 0:
            # Pad if episode too short
            padding = np.zeros((self.seq_length - len(frames),) + frames.shape[1:], dtype=frames.dtype)
            frames = np.concatenate([frames, padding], axis=0)
            max_start = 0
        
        start = np.random.randint(0, max_start + 1)
        
        # Extract sequence
        x_seq = frames[start:start + self.seq_length]  # (T, H, W, C)
        x_seq = torch.from_numpy(x_seq).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        
        if self.use_actions and actions is not None and len(actions) > 0:
            a_seq = actions[start:start + self.seq_length - 1]  # (T-1,)
            a_seq = torch.from_numpy(a_seq).long()
            
            # One-hot encode discrete actions
            if self.action_dim is not None and self.action_dim > 1 and a_seq.ndim == 1:
                a_seq_onehot = torch.zeros(self.seq_length - 1, self.action_dim)
                a_seq_onehot.scatter_(1, a_seq.unsqueeze(1), 1.0)
                a_seq = a_seq_onehot.float()
            else:
                a_seq = a_seq.float()
        else:
            a_seq = None
        
        return {'x': x_seq, 'a': a_seq}
