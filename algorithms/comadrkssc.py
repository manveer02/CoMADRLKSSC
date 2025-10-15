import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim,hidden), nn.ReLU(), nn.Linear(hidden,hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    def forward(self, x):
        h = self.net(x)
        return self.mu(h), torch.exp(self.log_std)
    def get_action(self, obs):
        import torch
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.dim() == 1: 
            obs = obs.unsqueeze(0)
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a).sum(dim=-1)
        return torch.tanh(a), logp

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
        next_cost = self.cost_critic(batch['next_obs'], self.actor.get_action(batch['next_obs'])[0]); 
        cost_target = costs + 0.99 * next_cost * (1-dones)
        cost_pred = self.cost_critic(obs, actions); 
        cost_loss = F.mse_loss(cost_pred, cost_target)
        self.cost_opt.zero_grad(); 
        cost_loss.backward(); 
        self.cost_opt.step()
        qcoop = 0.5*(q1+q2); 
        safe_adv = qcoop - self.value(obs) - self.lambda_c * cost_pred
        adv_norm = (safe_adv - safe_adv.mean())/(safe_adv.std()+1e-8)
        mu, std = self.actor.forward(obs);
        dist = torch.distributions.Normal(mu, std); 
        new_logp = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(new_logp - old_logp); 
        surrogate1 = ratio * adv_norm.detach(); 
        surrogate2 = torch.clamp(ratio,1-self.clip,1+self.clip)*adv_norm.detach()
        policy_loss = -torch.min(surrogate1, surrogate2).mean();
        entropy = dist.entropy().sum(dim=-1).mean(); 
        loss = policy_loss - 0.01*entropy
        self.actor_opt.zero_grad(); 
        loss.backward(); 
        self.actor_opt.step()
        value_loss = F.mse_loss(self.value(obs), returns); 
        self.critic_opt.zero_grad(); value_loss.backward(); 
        self.critic_opt.step()
        avg_cost = costs.mean().item(); 
        cost_threshold = self.cfg.get('cost_threshold', 0.1); 
        self.lambda_c = max(0.0, self.lambda_c + self.lambda_c_lr*(avg_cost-cost_threshold))
        return {'critic_loss': critic_loss.item(), 'cost_loss': cost_loss.item(), 'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'entropy': entropy.item(), 'lambda_c': self.lambda_c, 'avg_cost': avg_cost}
