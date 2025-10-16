# demo_visualize.py
import argparse, os, time
import torch, numpy as np
try:
    from pettingzoo.mpe import simple_spread_v3
except Exception:
    from pettingzoo.mpe2 import simple_spread_v3

# load your Actor class (import path may change)
from algorithms.comadrkssc import Actor

def load_actor(actor_path, obs_dim, act_dim, device='cpu'):
    model = Actor(obs_dim, act_dim)
    sd = torch.load(actor_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model

def run_episode_render(actor, env, max_cycles=50, render=True, sleep=0.02):
    # Use AEC API (non-parallel) for nicer rendering if available
    try:
        aec = env.env  # some wrappers expose .env
    except Exception:
        aec = None

    # we will use the parallel env API here: we will call step on parallel_env and call render()
    obs = env.reset()
    if isinstance(obs, tuple) and len(obs) == 2:
        obs, _ = obs

    done = False
    steps = 0
    while True:
        actions = {}
        for agent, ob in obs.items():
            # prepare obs tensor
            import torch
            o = torch.tensor(ob, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mu, _ = actor.forward(o)
            raw = mu.squeeze(0).cpu().numpy()
            act = np.tanh(raw)
            # scale to env action space if needed
            act_space = env.action_space(agent)
            if hasattr(act_space, 'low'):
                low = np.asarray(act_space.low, dtype=float)
                high = np.asarray(act_space.high, dtype=float)
                act_scaled = low + (act + 1.0) * 0.5 * (high - low)
                act_scaled = np.clip(act_scaled, low, high)
            else:
                act_scaled = act
            actions[agent] = act_scaled

        next_ret = env.step(actions)
        if len(next_ret) == 5:
            obs, rewards, terms, truncs, infos = next_ret
            dones = {k: terms.get(k, False) or truncs.get(k, False) for k in terms.keys()}
        else:
            obs, rewards, dones, infos = next_ret
            dones = dones

        if render:
            try:
                env.render()
                time.sleep(sleep)
            except Exception:
                pass

        steps += 1
        if all(dones.values()) or steps >= max_cycles:
            break

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='results/ours_mpe/models/actor_ep100.pt')
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--render', action='store_true')
    p.add_argument('--num_agents', type=int, default=3)
    p.add_argument('--max_cycles', type=int, default=50)
    args = p.parse_args()

    env = simple_spread_v3.parallel_env(N=args.num_agents, max_cycles=args.max_cycles, continuous_actions=True)
    # sample agent to get dims
    env.reset()
    agent0 = env.agents[0]
    obs_dim = env.observation_space(agent0).shape[0]
    act_dim = env.action_space(agent0).shape[0] if hasattr(env.action_space(agent0), 'shape') else 1

    actor = load_actor(args.model, obs_dim, act_dim)
    for ep in range(args.episodes):
        run_episode_render(actor, env, max_cycles=args.max_cycles, render=args.render)
