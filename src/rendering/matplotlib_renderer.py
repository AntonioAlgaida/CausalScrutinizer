# File: src/evaluation/matplotlib_renderer.py
# (Showing the full updated file for clarity)

import matplotlib.pylab as plt
import numpy as np
from typing import Dict
import matplotlib.patches as patches

def render_matplotlib_snapshot(scenario_data: Dict[str, np.ndarray], timestep: int, output_path: str):
    """
    Renders a full snapshot of the scenario (map + agents) at a specific
    timestep using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    
    # --- 1. Draw the Static Map (same logic as before) ---
    polylines = scenario_data['map_polylines']
    polyline_types = scenario_data['map_polyline_types']
    ROAD_GRAPH_COLORS = {
        0: '#E6E6E6', 1: '#E6E6E6', 2: '#E6E6E6', 3: '#E6E6E6',
        6: '#FFFFFF', 7: '#FFFFFF', 8: '#FFFFFF',
        9: '#FFFF00', 10: '#FFFF00', 13: '#FFFF00',
        11: '#FFD700', 12: '#FFD700',
        15: '#505050', 16: '#505050',
        17: '#FF0000', 18: '#C8C8C8', 19: '#C8C8C8',
    }
    DEFAULT_COLOR = '#E6E6E6'
    
    for i, p_type in enumerate(polyline_types):
        polyline = polylines[i]
        if polyline.shape[0] > 0:
            color = ROAD_GRAPH_COLORS.get(p_type, DEFAULT_COLOR)
            ax.plot(polyline[:, 0], polyline[:, 1], '.', color=color, ms=2)

    # --- 2. Draw the Dynamic Agents at the specified timestep ---
    traj_data = scenario_data['all_agent_trajectories']
    valid_mask = scenario_data['valid_mask']
    sdc_index = scenario_data['sdc_track_index']
    states_at_t = traj_data[:, timestep, :]
    valid_agents_mask = valid_mask[:, timestep]
    valid_indices = np.where(valid_agents_mask)[0]

    for agent_idx in valid_indices:
        agent_state = states_at_t[agent_idx]
        center_x, center_y = agent_state[0], agent_state[1]
        length, width = agent_state[3], agent_state[4]
        heading_rad = agent_state[6]
        
        # Matplotlib's Rectangle needs the bottom-left corner, not the center.
        # We calculate this by rotating the half-dimensions.
        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        bottom_left_offset = np.array([
            -length / 2 * cos_h + width / 2 * sin_h,
            -length / 2 * sin_h - width / 2 * cos_h
        ])
        bottom_left_corner = np.array([center_x, center_y]) + bottom_left_offset
        
        color = 'g' if agent_idx == sdc_index else 'b'
        
        rect = patches.Rectangle(
            xy=bottom_left_corner,
            width=length,
            height=width,
            angle=np.rad2deg(heading_rad),
            facecolor=color,
            edgecolor='k', # Add a black edge for clarity
            linewidth=0.5,
            zorder=10 # Ensure cars are drawn on top of the map
        )
        ax.add_patch(rect)

    # --- 3. Set the view and save ---
    sdc_traj = scenario_data['all_agent_trajectories'][sdc_index]
    center_x, center_y = sdc_traj[timestep, :2]
    view_range_meters = 800 / 5
    ax.set_xlim(center_x - view_range_meters / 2, center_x + view_range_meters / 2)
    ax.set_ylim(center_y - view_range_meters / 2, center_y + view_range_meters / 2)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor='black')
    plt.close(fig)
    print(f"✅ Matplotlib snapshot saved to: {output_path}")