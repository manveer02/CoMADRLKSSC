import pygame
import numpy as np

# A professional, colorblind-friendly palette
AGENT_COLORS = [
    (86, 180, 233),   # Sky Blue
    (230, 159, 0),    # Orange
    (0, 158, 115),    # Bluish Green
    (240, 228, 66),   # Yellow
    (204, 121, 167),  # Reddish Purple
]

def render_for_paper(env, obs, screen, font, title_font, episode, step):
    """
    Renders a high-quality, informative frame suitable for a research paper.
    """
    screen_size = screen.get_size()[0]
    
    # 1. Clear screen and draw a reference grid
    screen.fill((255, 255, 255)) # White background
    for i in range(0, screen_size, 25):
        pygame.draw.line(screen, (230, 230, 230), (i, 0), (i, screen_size), 1)
        pygame.draw.line(screen, (230, 230, 230), (0, i), (screen_size, i), 1)

    # 2. Parse entities and check their status
    entities_to_draw = []
    for i, agent_name in enumerate(env.agents):
        if agent_name not in obs: continue

        agent_obs = obs[agent_name]
        agent_pos = agent_obs[2:4]
        landmark_relative_pos = agent_obs[4:6]
        
        # Calculate the landmark's true absolute position.
        landmark_abs_pos = agent_pos + landmark_relative_pos

        is_at_goal = np.linalg.norm(agent_pos - landmark_abs_pos) < 0.05
        
        is_colliding = False
        for other_name in env.agents:
            if agent_name == other_name or other_name not in obs: continue
            other_pos = obs[other_name][2:4]
            if np.linalg.norm(agent_pos - other_pos) < 0.1:
                is_colliding = True
                break

        entities_to_draw.append({
            'id': i,
            'agent_pos': ((agent_pos + 1) / 2 * screen_size).astype(int),
            'landmark_pos': ((landmark_abs_pos + 1) / 2 * screen_size).astype(int),
            'color': AGENT_COLORS[i % len(AGENT_COLORS)],
            'is_at_goal': is_at_goal,
            'is_colliding': is_colliding,
        })

    # 3. Perform drawing in layers for a clean look
    # Layer 1: Target lines
    for entity in entities_to_draw:
        pygame.draw.line(screen, (200, 200, 200), entity['agent_pos'], entity['landmark_pos'], 2)

    # Layer 2: Landmarks (Squares)
    for entity in entities_to_draw:
        pos, color = entity['landmark_pos'], entity['color']
        landmark_rect = pygame.Rect(pos[0] - 8, pos[1] - 8, 16, 16)
        pygame.draw.rect(screen, color, landmark_rect, border_radius=3)
        pygame.draw.rect(screen, (0,0,0), landmark_rect, 2, border_radius=3)
        label = font.render(f"L{entity['id']}", True, (50, 50, 50))
        screen.blit(label, (pos[0] + 12, pos[1] + 12))

    # Layer 3: Agents (Circles)
    for entity in entities_to_draw:
        pos, color = entity['agent_pos'], entity['color']
        
        if entity['is_colliding']:
            pygame.draw.circle(screen, (255, 0, 0, 100), pos, 15)

        pygame.draw.circle(screen, color, pos, 10)
        pygame.draw.circle(screen, (0,0,0), pos, 10, 2)
        
        if entity['is_at_goal']:
            pygame.draw.polygon(screen, (255,255,255), [(pos[0]-5, pos[1]), (pos[0]-2, pos[1]+4), (pos[0]+5, pos[1]-3)])

        label = font.render(f"A{entity['id']}", True, (0, 0, 0))
        screen.blit(label, (pos[0] + 12, pos[1] - 24))

    # 4. Add a title
    title_surface = title_font.render(f"Episode: {episode}, Step: {step}", True, (0, 0, 0))
    screen.blit(title_surface, (10, 5))

    return np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))