import argparse, os, torch
from pettingzoo.mpe import simple_spread_v3
from algorithms.comadrkssc import CoMADRLKSSC
from utils.mpe_trainer import MPETrainer

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--num_agents', type=int, default=3)
    p.add_argument('--log_dir', type=str, default='results/ours_mpe')
    p.add_argument('--render', action='store_true', help='Enable live visualization')
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    env = simple_spread_v3.parallel_env(N=args.num_agents, local_ratio=0.5, max_cycles=25, continuous_actions=True, render_mode="rgb_array")
    env.reset()

    # Get dimensions for algorithm setup
    sample_agent = env.agents[0]
    obs_dim = env.observation_space(sample_agent).shape[0]
    act_dim = env.action_space(sample_agent).shape[0]
    state_dim = env.state_space.shape[0] # SOTA CHANGE: Get global state dimension

    cfg = {
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'state_dim': state_dim, # SOTA CHANGE: Pass global state dimension
        'actor_lr': 1e-4,
        'critic_lr': 3e-4,
        'lambda_c_lr': 1e-3,
        'batch_size': 1024, # Larger batch size for more stable centralized learning
        'gamma': 0.99,
        'lam': 0.95,
        'clip': 0.2,
        'entropy_coef': 0.01,
        'cost_threshold': 0.1,
        'lambda_c': 1.0,
        'max_grad_norm': 1.0,
    }
    alg = CoMADRLKSSC(cfg)

    trainer = MPETrainer(env, alg, log_dir=args.log_dir, save_every=100, live_render=args.render)
    trainer.train(num_episodes=args.episodes)