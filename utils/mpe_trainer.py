import numpy as np
import torch
import os
from collections import deque
import imageio
import csv
import pygame

from utils.runner import collate
# Renderer is kept here for simplicity with the grid world
def render_grid_for_paper(env, screen, font, title_font, episode, step):
    AGENT_COLORS = [(86, 180, 233), (230, 159, 0), (0, 158, 115)]
    window_size, grid_size = screen.get_size()[0], env.unwrapped.grid_size
    pix_size = window_size / grid_size
    
    screen.fill((255, 255, 255))
    for i in range(0, window_size, int(pix_size)):
        pygame.draw.line(screen, (230, 230, 230), (i, 0), (i, window_size), 1)
        pygame.draw.line(screen, (230, 230, 230), (0, i), (window_size, i), 1)

    for agent, pos in env.unwrapped.goal_positions.items():
        agent_idx = env.unwrapped.agents.index(agent)
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        pygame.draw.rect(screen, color, (pos[1] * pix_size, pos[0] * pix_size, pix_size, pix_size), 5)

    for r in range(grid_size):
        for c in range(grid_size):
            if env.unwrapped.grid[r, c] == 1:
                pygame.draw.rect(screen, (50, 50, 50), (c * pix_size, r * pix_size, pix_size, pix_size))

    for agent, pos in env.unwrapped.agent_positions.items():
        agent_idx = env.unwrapped.agents.index(agent)
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]
        center = ((pos[1] + 0.5) * pix_size, (pos[0] + 0.5) * pix_size)
        pygame.draw.circle(screen, color, center, pix_size / 2.5)
        label = font.render(f"A{agent_idx}", True, (255, 255, 255))
        screen.blit(label, label.get_rect(center=center))
        
    title = title_font.render(f"Episode: {episode}, Step: {step}", True, (0, 0, 0))
    screen.blit(title, (10, 5))
    return np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))


class SimpleBuffer:
    def __init__(self, max_size=200000):
        self.buf = deque(maxlen=max_size)
    def add(self, tr): self.buf.append(tr)
    def size(self): return len(self.buf)
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        return collate([self.buf[i] for i in idx])

class MPETrainer:
    def __init__(self, env, alg, log_dir='results/ours_gridworld', save_every=100, max_cycles=100, live_render=False):
        self.env, self.alg = env, alg
        self.log_dir, self.save_every, self.max_cycles = log_dir, save_every, max_cycles
        self.buffer, self.batch_size = SimpleBuffer(), alg.cfg.get('batch_size', 1024)
        self.live_render = live_render
        self.pygame_screen = self.pygame_font = self.pygame_title_font = None
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_path = os.path.join(log_dir, 'metrics.csv')
        self.paper_img_dir = os.path.join(log_dir, 'paper_images'); os.makedirs(self.paper_img_dir, exist_ok=True)
        with open(self.metrics_path, 'w', newline='') as f:
            csv.writer(f).writerow(['episode', 'avg_return', 'avg_cost_ep', 'success_rate', 'critic_loss', 'cost_loss', 'policy_loss', 'value_loss', 'entropy', 'lambda_c', 'avg_cost_batch'])

    def _init_pygame(self):
        if self.pygame_screen: return
        pygame.init(); pygame.font.init()
        size = self.env.unwrapped.window_size
        self.pygame_screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("MARL Grid World Visualizer")
        self.pygame_font = pygame.font.SysFont('Arial', 16, bold=True)
        self.pygame_title_font = pygame.font.SysFont('Arial', 20, bold=True)
    
    def _capture_paper_image(self, ep, step, name):
        self._init_pygame()
        render_grid_for_paper(self.env, self.pygame_screen, self.pygame_font, self.pygame_title_font, ep, step)
        path = os.path.join(self.paper_img_dir, f"ep{ep:04d}_step{step:03d}_{name}.png")
        pygame.image.save(self.pygame_screen, path); print(f"🖼️ Saved paper-quality image: {os.path.basename(path)}")
    
    def _record_video_episode(self, ep):
        try:
            frames, (obs, _) = [], self.env.reset(); self._init_pygame()
            for step in range(self.max_cycles):
                frames.append(render_grid_for_paper(self.env, self.pygame_screen, self.pygame_font, self.pygame_title_font, ep, step))
                actions = {a: self.alg.actor.get_action(o, deterministic=True)[0].item() for a, o in obs.items()}
                if not actions: break
                obs, _, terms, truncs, _ = self.env.step(actions)
                if all(terms.values()) or all(truncs.values()):
                    frames.append(render_grid_for_paper(self.env, self.pygame_screen, self.pygame_font, self.pygame_title_font, ep, step + 1)); break
            path = os.path.join(self.log_dir, f"episode_{ep:04d}.mp4")
            imageio.mimsave(path, frames, fps=10); print(f"📹 Saved video: {os.path.basename(path)}")
        except Exception as e: print(f"⚠️ Skipped video saving (reason: {e})")

    def train(self, num_episodes=2000):
        if self.live_render: self._init_pygame()
        
        for ep in range(1, num_episodes + 1):
            obs, _ = self.env.reset()
            ep_rewards, ep_costs = {a: 0.0 for a in self.env.agents}, {a: 0.0 for a in self.env.agents}
            metrics = {k: [] for k in ['critic_loss', 'cost_loss', 'policy_loss', 'value_loss', 'entropy', 'lambda_c', 'avg_cost']}
            is_milestone = (ep % self.save_every == 0)
            if is_milestone: self._capture_paper_image(ep, 0, "start")

            for step in range(self.max_cycles):
                if self.live_render:
                    render_grid_for_paper(self.env, self.pygame_screen, self.pygame_font, self.pygame_title_font, ep, step)
                    pygame.display.flip(); pygame.time.wait(50)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: pygame.quit(); print("🛑 Training stopped."); return

                actions, logps = {}, {}
                for agent, ob in obs.items():
                    action, logp = self.alg.actor.get_action(ob)
                    actions[agent], logps[agent] = action.item(), logp.item()
                if not actions: break
                
                state = self.env.state()
                next_obs, rewards, terms, truncs, _ = self.env.step(actions)
                next_state = self.env.state()
                
                current_pos = [tuple(p) for p in self.env.unwrapped.agent_positions.values()]
                costs = {a: float(current_pos.count(tuple(self.env.unwrapped.agent_positions[a])) > 1) for a in self.env.agents}

                for agent in self.env.agents:
                    reward = rewards.get(agent, 0.0)
                    ep_rewards[agent] += reward; ep_costs[agent] += costs.get(agent, 0.0)
                    # --- FIX IS HERE ---
                    self.buffer.add({
                        'obs': obs[agent], 
                        'next_obs': next_obs[agent], # This was missing
                        'actions': torch.tensor(actions.get(agent, 0), dtype=torch.long), 
                        'rewards': torch.tensor(reward, dtype=torch.float32), 
                        'costs': torch.tensor(costs.get(agent, 0.0), dtype=torch.float32), 
                        'dones': torch.tensor(float(terms.get(agent, False) or truncs.get(agent, False)), dtype=torch.float32),
                        'logp': logps.get(agent, 0), 
                        'states': state, 
                        'next_states': next_state
                    })
                obs = next_obs
                
                if self.buffer.size() >= self.batch_size:
                    for _ in range(2):
                        batch = self.buffer.sample(self.batch_size)
                        update_metrics = self.alg.update(batch)
                        for k, v in update_metrics.items(): metrics[k].append(v)
                
                if all(terms.values()) or all(truncs.values()): break

            if is_milestone:
                self._capture_paper_image(ep, step + 1, "end")
                self._record_video_episode(ep)

            avg_metrics = {k: np.mean(v) if v else 0 for k, v in metrics.items()}
            avg_return, avg_cost = np.mean(list(ep_rewards.values())), np.mean(list(ep_costs.values()))
            success = 1.0 if all(self.env.unwrapped.terminations.values()) else 0.0

            if ep % 10 == 0:
                print(f"Ep {ep}: Ret={avg_return:.2f}, Cost={avg_cost:.2f}, λ={avg_metrics['lambda_c']:.2f}, V_Loss={avg_metrics['value_loss']:.2f}")

            with open(self.metrics_path, 'a', newline='') as f:
                csv.writer(f).writerow([ep, avg_return, avg_cost, success] + [avg_metrics[k] for k in metrics.keys()])
        
        if self.live_render: pygame.quit()