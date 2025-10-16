import argparse, os, torch
from utils.grid_env import MultiAgentGridEnv
from algorithms.comadrkssc import CoMADRLKSSC
from utils.mpe_trainer import MPETrainer

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=5000)
    
    # --- FIX IS HERE ---
    p.add_argument('--num_agents', type=int, default=3) # Corrected from 'add_gument'
    
    p.add_argument('--grid_size', type=int, default=10)
    p.add_argument('--log_dir', type=str, default='results/ours_gridworld_curiosity')
    p.add_argument('--render', action='store_true', help='Enable live visualization')
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    
    env = MultiAgentGridEnv(n_agents=args.num_agents, grid_size=args.grid_size, render_mode="rgb_array")
    env.reset()

    obs_dim = env.observation_spaces[env.agents[0]].shape[0] * env.observation_spaces[env.agents[0]].shape[1]
    act_dim = env.action_spaces[env.agents[0]].n
    state_dim = env.state_space.shape[0]

    cfg = {
        'obs_dim': obs_dim, 'act_dim': act_dim, 'state_dim': state_dim,
        'actor_lr': 3e-4, 'critic_lr': 1e-3, 'lambda_c_lr': 1e-3, 'icm_lr': 3e-4,
        'batch_size': 1024, 'gamma': 0.99, 'lam': 0.95, 'clip': 0.2,
        'entropy_coef': 0.01, 'cost_threshold': 0.05, 'lambda_c': 1.0, 'max_grad_norm': 0.5,
        'icm_beta': 0.2,
        'intrinsic_reward_weight': 0.01
    }
    alg = CoMADRLKSSC(cfg)

    trainer = MPETrainer(
        env, alg, log_dir=args.log_dir, save_every=500,
        max_cycles=100, live_render=args.render
    )
    trainer.train(num_episodes=args.episodes)