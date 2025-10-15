import torch, numpy as np
def collate(batch):
    obs = torch.tensor([b['obs'] for b in batch], dtype=torch.float32)
    actions = torch.tensor([b['actions'] for b in batch], dtype=torch.float32)
    rewards = torch.tensor([b['rewards'] for b in batch], dtype=torch.float32)
    next_obs = torch.tensor([b['next_obs'] for b in batch], dtype=torch.float32)
    costs = torch.tensor([b['costs'] for b in batch], dtype=torch.float32)
    dones = torch.tensor([b['dones'] for b in batch], dtype=torch.float32)
    logp = torch.tensor([b['logp'] for b in batch], dtype=torch.float32)
    return {'obs': obs,'actions': actions,'rewards': rewards,'next_obs': next_obs,'costs': costs,'dones': dones,'logp': logp}
