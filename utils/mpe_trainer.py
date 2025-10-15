# utils/mpe_trainer.py
import numpy as np, torch, os
from collections import deque
from utils.runner import collate

class SimpleBuffer:
    def __init__(self, max_size=200000):
        self.buf = deque(maxlen=max_size)
    def add(self, tr):
        self.buf.append(tr)
    def size(self):
        return len(self.buf)
    def sample(self, batch_size):
        import numpy as np
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        return collate(batch)

class MPETrainer:
    def __init__(self, env, alg, log_dir='results/ours_mpe', save_every=50, max_cycles=25):
        self.env = env
        self.alg = alg
        self.log_dir = log_dir
        self.save_every = save_every
        self.max_cycles = max_cycles
        self.buffer = SimpleBuffer()
        self.batch_size = alg.cfg.get('batch_size', 128)
        os.makedirs(log_dir, exist_ok=True)
        open(os.path.join(log_dir,'metrics.csv'),'w').write('episode,avg_return,avg_cost,success_rate\n')

    def train(self, num_episodes=50):
        for ep in range(1, num_episodes + 1):
            obs, info = self.env.reset()
            ep_rewards = {a:0.0 for a in obs.keys()}
            ep_costs = {a:0.0 for a in obs.keys()}
            steps = 0

            while True:
                actions = {}
                logps = {}
                for agent, ob in obs.items():
                    at, logp = self.alg.actor.get_action(ob)
                    act = at.detach().cpu().numpy().squeeze(0)
                    actions[agent] = act
                    logps[agent] = float(logp.detach().cpu().numpy())

                next_obs, rewards, terms, truncs, infos = self.env.step(actions)
                dones = {k: terms.get(k, False) or truncs.get(k, False) for k in terms.keys()}

                for agent in obs.keys():
                    o = obs[agent]
                    a = actions[agent]
                    r = rewards.get(agent, 0.0)
                    no = next_obs.get(agent, o)
                    done_flag = dones.get(agent, False)
                    info = infos.get(agent, {})

                    # --- Robust cost computation ---
                    if 'collision' in info:
                        cost = 1.0 if info['collision'] else 0.0
                    else:
                        cost = 0.0
                        if 'agent_positions' in info:
                            positions = info['agent_positions']
                            for i in range(len(positions)):
                                for j in range(i + 1, len(positions)):
                                    dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                                    if dist < 0.08:
                                        cost += 1.0
                                        break

                    ep_rewards[agent] += r
                    ep_costs[agent] += cost

                    self.buffer.add({
                        'obs': o,
                        'actions': a,
                        'rewards': r,
                        'next_obs': no,
                        'costs': cost,
                        'dones': float(done_flag),
                        'logp': logps[agent]
                    })

                obs = next_obs
                steps += 1

                # Stop if all agents done or max cycles reached
                if all(dones.values()) or steps >= self.max_cycles:
                    break

                if self.buffer.size() >= max(256, self.batch_size):
                    batch = self.buffer.sample(self.batch_size)
                    self.alg.update(batch)

            avg_return = np.mean(list(ep_rewards.values()))
            avg_cost = np.mean(list(ep_costs.values()))
            success = 1.0 if avg_cost == 0.0 else 0.0

            with open(os.path.join(self.log_dir,'metrics.csv'),'a') as f:
                f.write(f"{ep},{avg_return},{avg_cost},{success}\n")

            if ep % 10 == 0:
                print(f"Episode {ep}: avg_return={avg_return:.3f}, avg_cost={avg_cost:.3f}")
