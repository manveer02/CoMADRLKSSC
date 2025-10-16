import numpy as np
import torch

def collate(batch):
    """
    Collates a batch of transitions from the replay buffer into tensors.
    """
    # --- FIX IS HERE ---
    obs = np.asarray([b['obs'] for b in batch], dtype=np.float32)
    next_obs = np.asarray([b['next_obs'] for b in batch], dtype=np.float32) # This was missing
    
    actions = torch.stack([b['actions'] for b in batch])
    logp = np.asarray([b['logp'] for b in batch], dtype=np.float32)
    
    states = np.asarray([b['states'] for b in batch], dtype=np.float32)
    next_states = np.asarray([b['next_states'] for b in batch], dtype=np.float32)

    rewards = torch.stack([b['rewards'] for b in batch])
    costs = torch.stack([b['costs'] for b in batch])
    dones = torch.stack([b['dones'] for b in batch])

    return {
        'obs': torch.tensor(obs, dtype=torch.float32),
        'next_obs': torch.tensor(next_obs, dtype=torch.float32), # This was missing
        'actions': actions,
        'rewards': rewards,
        'costs': costs,
        'dones': dones,
        'logp': torch.tensor(logp, dtype=torch.float32),
        'states': torch.tensor(states, dtype=torch.float32),
        'next_states': torch.tensor(next_states, dtype=torch.float32),
    }