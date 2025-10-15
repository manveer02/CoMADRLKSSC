# train_mpe.py
import argparse, os, torch
from pettingzoo.mpe import simple_spread_v3
from algorithms.comadrkssc import CoMADRLKSSC
from utils.mpe_trainer import MPETrainer

if __name__ == '__main__':
    # ---------------- Arguments ----------------
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--num_agents', type=int, default=3)
    p.add_argument('--max_cycles', type=int, default=25)
    p.add_argument('--log_dir', type=str, default='results/ours_mpe')
    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # ---------------- Environment ----------------
    env = simple_spread_v3.parallel_env(
        N=args.num_agents,
        local_ratio=0.5,
        max_cycles=args.max_cycles,
        continuous_actions=True
    )
    env.reset()

    # PettingZoo deprecates observation_spaces/action_spaces dicts
    sample_agent = list(env.observation_spaces.keys())[0]
    obs_dim = env.observation_space(sample_agent).shape[0]
    act_dim = env.action_space(sample_agent).shape[0]

    # ---------------- Agent Configuration ----------------
    cfg = {
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'actor_lr': 3e-4,
        'critic_lr': 1e-3,
        'cost_lr': 1e-3,
        'batch_size': 128,
        'cost_threshold': 0.1
    }
    alg = CoMADRLKSSC(cfg)

    # ---------------- Trainer ----------------
    trainer = MPETrainer(env, alg, log_dir=args.log_dir, save_every=50, max_cycles=args.max_cycles)
    trainer.train(num_episodes=args.episodes)
