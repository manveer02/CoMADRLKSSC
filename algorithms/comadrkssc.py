import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim,hidden), nn.ReLU(), nn.Linear(hidden,hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        # log_std is a learnable parameter, clamped to maintain stability
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.net(x)
        # Clamp log_std to avoid numerical instability
        log_std = torch.clamp(self.log_std, min=-20, max=2)
        return self.mu(h), torch.exp(log_std)

    def _get_logp_for_action(self, obs, action_env):
        # This function correctly calculates the log-probability of a given environment action [0, 1]
        
        # 1. Inverse scaling: action_env [0, 1] -> action_tanh [-1, 1]
        action_tanh = action_env * 2.0 - 1.0
        
        # Clamp for stable arctanh
        action_tanh = torch.clamp(action_tanh, -0.999999, 0.999999) 
        
        # 2. Inverse Tanh: action_tanh -> epsilon (Gaussian sample)
        epsilon = torch.atanh(action_tanh)
        
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)

        # 3. Calculate log N(epsilon | mu, std)
        logp_epsilon = dist.log_prob(epsilon).sum(dim=-1)

        # 4. Add Tanh correction (log(pi_tanh) = log(N(epsilon)) - log(|d(tanh)/d(epsilon)|))
        logp_tanh = logp_epsilon - torch.log(1.0 - action_tanh.pow(2) + 1e-6).sum(dim=-1)
        
        # 5. Add Rescaling correction (for the [0, 1] range shift)
        # The scaling transformation is linear: y = 0.5 * x + 0.5. Derivative is 0.5.
        # The correction term is log(|1 / derivative|) * act_dim = log(1 / 0.5) * act_dim = log(2) * act_dim
        act_dim = mu.shape[-1]
        LOG_TWO = torch.log(torch.tensor(2.0, dtype=torch.float32)).to(obs.device)
        logp_env = logp_tanh + act_dim * LOG_TWO
        
        return logp_env

    def get_action(self, obs):
        import torch
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.dim() == 1: 
            obs = obs.unsqueeze(0)
            
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        
        # 1. Sample epsilon (the Gaussian sample)
        epsilon = dist.sample()
        logp_epsilon = dist.log_prob(epsilon).sum(dim=-1)
        
        # 2. Tanh squash to get action in [-1, 1] range
        action_tanh = torch.tanh(epsilon)

        # 3. Rescale action from [-1, 1] to [0.0, 1.0] (FIXES CLIPPING)
        action_env = (action_tanh + 1.0) / 2.0
        
        # 4. Calculate log probability of the sampled action_env (FIXES LOG-PROB ERROR)
        # Tanh correction
        logp_tanh = logp_epsilon - torch.log(1.0 - action_tanh.pow(2) + 1e-6).sum(dim=-1)
        
        # Rescaling correction
        act_dim = mu.shape[-1]
        LOG_TWO = torch.log(torch.tensor(2.0, dtype=torch.float32)).to(obs.device)
        logp_env = logp_tanh + act_dim * LOG_TWO
        
        return action_env, logp_env

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(obs_dim+act_dim,hidden), nn.ReLU(), nn.Linear(hidden,hidden), nn.ReLU(), nn.Linear(hidden,1))
    def forward(self, s,a):
        import torch
        if not isinstance(s, torch.Tensor): s = torch.tensor(s, dtype=torch.float32)
        if not isinstance(a, torch.Tensor): a = torch.tensor(a, dtype=torch.float32)
        if s.dim() == 1: 
            s = s.unsqueeze(0)
        if a.dim() == 1: 
            a = a.unsqueeze(0)
        return self.net(torch.cat([s,a], dim=-1)).squeeze(-1)

class Value(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(obs_dim,hidden), nn.ReLU(), nn.Linear(hidden,hidden), nn.ReLU(), nn.Linear(hidden,1))
    def forward(self, s):
        import torch
        if not isinstance(s, torch.Tensor): 
            s = torch.tensor(s, dtype=torch.float32)
        if s.dim() == 1: 
            s = s.unsqueeze(0)
        return self.net(s).squeeze(-1)

class CoMADRLKSSC:
    def __init__(self, cfg):
        self.cfg = cfg
        obs_dim = cfg.get('obs_dim', 8)
        act_dim = cfg.get('act_dim', 2)
        self.actor = Actor(obs_dim, act_dim)
        self.critic1 = Critic(obs_dim, act_dim)
        self.critic2 = Critic(obs_dim, act_dim)
        self.cost_critic = Critic(obs_dim, act_dim)
        self.value = Value(obs_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.get('actor_lr',3e-4))
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters())+list(self.critic2.parameters())+list(self.value.parameters()), lr=cfg.get('critic_lr',1e-3))
        self.cost_opt = torch.optim.Adam(self.cost_critic.parameters(), lr=cfg.get('cost_lr',1e-3))
        self.gamma = cfg.get('gamma',0.99); 
        self.lam = cfg.get('lam',0.95); 
        self.clip = cfg.get('clip',0.2)
        self.lambda_c = cfg.get('lambda_c',1.0); 
        self.lambda_c_lr = cfg.get('lambda_c_lr',1e-2)
    def compute_gae(self, rewards, values, masks, gamma=0.99, lam=0.95):
        advs=[]; gae=0; values = values + [0]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma*values[step+1]*masks[step] - values[step]
            gae = delta + gamma*lam*masks[step]*gae
            advs.insert(0, gae)
        return advs
    def update(self, batch):
        import torch.nn.functional as F, torch
        obs = batch['obs']; 
        actions = batch['actions']; 
        rewards = batch['rewards']; 
        costs = batch['costs']; 
        dones = batch['dones']; 
        old_logp = batch['logp']
        with torch.no_grad(): 
            values = self.value(obs).cpu().numpy().tolist()
        masks = (1-dones).cpu().numpy().tolist()
        advs = self.compute_gae(rewards.cpu().numpy().tolist(), values, masks, self.gamma, self.lam)
        advs = torch.tensor(advs, dtype=torch.float32); 
        returns = advs + torch.tensor(values, dtype=torch.float32)
        with torch.no_grad():
            next_values = self.value(batch['next_obs']); 
            td_target = rewards + self.gamma * next_values * (1-dones)
        q1 = self.critic1(obs, actions); 
        q2 = self.critic2(obs, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        self.critic_opt.zero_grad(); 
        critic_loss.backward(); 
        self.critic_opt.step()
        
        # Cost Critic Update
        # Note: self.actor.get_action(batch['next_obs'])[0] still uses the new, correctly scaled action.
        next_cost = self.cost_critic(batch['next_obs'], self.actor.get_action(batch['next_obs'])[0]); 
        cost_target = costs + 0.99 * next_cost * (1-dones)
        cost_pred = self.cost_critic(obs, actions); 
        cost_loss = F.mse_loss(cost_pred, cost_target)
        self.cost_opt.zero_grad(); 
        cost_loss.backward(); 
        self.cost_opt.step()
        
        # Policy Update
        qcoop = 0.5*(q1+q2); 
        safe_adv = qcoop - self.value(obs) - self.lambda_c * cost_pred
        adv_norm = (safe_adv - safe_adv.mean())/(safe_adv.std()+1e-8)
        
        # --- FIX: Correctly calculate new_logp using the inverse transformations ---
        new_logp = self.actor._get_logp_for_action(obs, actions) 
        
        ratio = torch.exp(new_logp - old_logp); 
        surrogate1 = ratio * adv_norm.detach(); 
        surrogate2 = torch.clamp(ratio,1-self.clip,1+self.clip)*adv_norm.detach()
        policy_loss = -torch.min(surrogate1, surrogate2).mean();
        
        # For entropy, use the Gaussian entropy for simplicity (common in PPO implementations)
        mu, std = self.actor.forward(obs);
        dist = torch.distributions.Normal(mu, std); 
        entropy = dist.entropy().sum(dim=-1).mean(); 
        
        loss = policy_loss - 0.01*entropy
        self.actor_opt.zero_grad(); 
        loss.backward(); 
        self.actor_opt.step()
        
        # Value Update
        value_loss = F.mse_loss(self.value(obs), returns); 
        self.critic_opt.zero_grad(); value_loss.backward(); 
        self.critic_opt.step()
        
        # Lambda Update
        avg_cost = costs.mean().item(); 
        cost_threshold = self.cfg.get('cost_threshold', 0.1); 
        self.lambda_c = max(0.0, self.lambda_c + self.lambda_c_lr*(avg_cost-cost_threshold))
        
        return {'critic_loss': critic_loss.item(), 'cost_loss': cost_loss.item(), 'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'entropy': entropy.item(), 'lambda_c': self.lambda_c, 'avg_cost': avg_cost}