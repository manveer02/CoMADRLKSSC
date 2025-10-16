import numpy as np
import pygame
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces

class MultiAgentGridEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "MultiAgentGridEnv-v0"}

    def __init__(self, n_agents=3, grid_size=10, render_mode="rgb_array"):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.render_mode = render_mode

        # Define agents and possible agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        
        # Action space: 0:stay, 1:up, 2:down, 3:left, 4:right
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}
        
        # Observation space: The full grid is observed by each agent
        # Each cell can be: 0=empty, 1=wall, 2=goal, 3+ = agent_id + 3
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=n_agents + 2, shape=(grid_size, grid_size), dtype=np.uint8) 
            for agent in self.agents
        }
        
        # Centralized state space (for the critic)
        self.state_space = spaces.Box(
            low=0, high=n_agents + 2, shape=(grid_size * grid_size,), dtype=np.uint8
        )

        self.window = None
        self.window_size = 512

    def reset(self, seed=None, options=None):
        self.agent_positions = {}
        self.goal_positions = {}
        
        # Create the grid: 0 for empty, 1 for walls
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.grid[2:4, 2] = 1 # Vertical wall
        self.grid[7, 4:7] = 1 # Horizontal wall
        self.grid[5, 8] = 1   # Single block

        # Place agents and goals randomly, ensuring no overlaps
        occupied_positions = set()
        for i, agent in enumerate(self.agents):
            # Place agent
            pos = self._get_random_empty_pos(occupied_positions)
            self.agent_positions[agent] = pos
            occupied_positions.add(tuple(pos))
            # Place goal
            goal_pos = self._get_random_empty_pos(occupied_positions)
            self.goal_positions[agent] = goal_pos
            occupied_positions.add(tuple(goal_pos))

        self.steps = 0
        self.terminations = {agent: False for agent in self.agents}
        
        observations = self._get_obs()
        infos = self._get_info()
        return observations, infos

    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        
        # Move agents
        for agent, action in actions.items():
            if self.terminations[agent]: continue

            new_pos = self.agent_positions[agent].copy()
            if action == 1: new_pos[0] -= 1 # Up
            elif action == 2: new_pos[0] += 1 # Down
            elif action == 3: new_pos[1] -= 1 # Left
            elif action == 4: new_pos[1] += 1 # Right

            # Check for collisions with walls or grid boundaries
            if (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size and 
                self.grid[tuple(new_pos)] != 1):
                self.agent_positions[agent] = new_pos

        # Calculate rewards and terminations
        at_goal_count = 0
        for i, agent in enumerate(self.agents):
            if self.terminations[agent]:
                at_goal_count +=1
                continue

            pos = self.agent_positions[agent]
            goal_pos = self.goal_positions[agent]
            
            # Reward for getting closer to goal
            dist_to_goal = np.linalg.norm(pos - goal_pos)
            rewards[agent] -= dist_to_goal * 0.1

            # Penalty for collision with other agents
            for other_agent, other_pos in self.agent_positions.items():
                if agent != other_agent and np.array_equal(pos, other_pos):
                    rewards[agent] -= 1.0

            # Large reward for reaching goal
            if np.array_equal(pos, goal_pos):
                rewards[agent] += 10.0
                self.terminations[agent] = True
                at_goal_count +=1

        self.steps += 1
        truncations = {agent: self.steps >= 100 for agent in self.agents}
        
        if all(self.terminations.values()):
            for agent in self.agents:
                rewards[agent] += 50.0 # Team completion bonus

        observations = self._get_obs()
        infos = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observations, rewards, self.terminations, truncations, infos

    def render(self):
        if self.render_mode is None: return
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Multi-Agent Grid World")
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_size = self.window_size / self.grid_size

        # Draw goals
        for agent, pos in self.goal_positions.items():
            agent_idx = self.agents.index(agent)
            color = pygame.Color(AGENT_COLORS[agent_idx % len(AGENT_COLORS)])
            pygame.draw.rect(canvas, color, (pos[1] * pix_size, pos[0] * pix_size, pix_size, pix_size), 5)

        # Draw walls
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == 1:
                    pygame.draw.rect(canvas, (50, 50, 50), (c * pix_size, r * pix_size, pix_size, pix_size))

        # Draw agents
        for agent, pos in self.agent_positions.items():
            agent_idx = self.agents.index(agent)
            color = pygame.Color(AGENT_COLORS[agent_idx % len(AGENT_COLORS)])
            pygame.draw.circle(canvas, color, ((pos[1] + 0.5) * pix_size, (pos[0] + 0.5) * pix_size), pix_size / 2.5)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            return

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _get_random_empty_pos(self, occupied):
        while True:
            pos = np.random.randint(0, self.grid_size, size=2)
            if self.grid[tuple(pos)] == 0 and tuple(pos) not in occupied:
                return pos

    def _get_obs(self):
        full_grid = self.grid.copy()
        for i, agent in enumerate(self.agents):
            goal_pos = self.goal_positions[agent]
            full_grid[tuple(goal_pos)] = 2 # Goal marker
            
        for i, agent in enumerate(self.agents):
            agent_pos = self.agent_positions[agent]
            full_grid[tuple(agent_pos)] = i + 3 # Agent marker
            
        return {agent: full_grid for agent in self.agents}
    
    def state(self):
        return self._get_obs()[self.agents[0]].flatten()

    def _get_info(self):
        return {agent: {} for agent in self.agents}

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()

AGENT_COLORS = ["blue", "red", "green", "purple", "orange"]