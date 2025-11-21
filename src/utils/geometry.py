# File: src/utils/geometry.py
# Contains all deterministic geometric calculations for the Causal Scrutinizer.

import numpy as np
from scipy.spatial import ConvexHull
from typing import Tuple, List, Dict

# --- Agent Type Mapping ---
AGENT_TYPE_MAP = {
    1: "VEHICLE",
    2: "PEDESTRIAN",
    3: "CYCLIST",
    4: "OTHER"
}

def get_bounding_box_vertices(agent_state: np.ndarray) -> np.ndarray:
    """Helper function to get the 4 corners of an agent's bounding box."""
    x, y, length, width, heading = agent_state[0], agent_state[1], agent_state[3], agent_state[4], agent_state[6]
    
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    half_l, half_w = length / 2, width / 2
    
    corners = np.array([
        [-half_l, -half_w], [-half_l, +half_w],
        [+half_l, +half_w], [+half_l, -half_w]
    ])
    
    rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    rotated_corners = corners @ rot_matrix.T
    
    return rotated_corners + np.array([x, y])

def check_for_path_conflict(
    scenario_data: Dict,
    current_timestep: int,
    time_horizon_s: float = 2.0
) -> Tuple[bool, str, int | None]:
    """
    Performs a geometric conflict check between the AV's future path and
    the current position of all other agents using AABB intersection.
    """
    try:
        sdc_index = scenario_data['sdc_track_index']
        sdc_traj = scenario_data.get('sdc_route', scenario_data['all_agent_trajectories'][sdc_index])
        
        if current_timestep >= len(sdc_traj): return (False, "NONE", None)
        
        current_av_state = sdc_traj[current_timestep]
        future_timestep = min(current_timestep + int(time_horizon_s * 10), len(sdc_traj) - 1)
        future_av_state = sdc_traj[future_timestep]

        current_av_box = get_bounding_box_vertices(current_av_state)
        future_av_box = get_bounding_box_vertices(future_av_state)
        
        all_vertices = np.vstack([current_av_box, future_av_box])
        av_swept_path_polygon_indices = ConvexHull(all_vertices).vertices
        av_swept_path_points = all_vertices[av_swept_path_polygon_indices]
        
        av_min_x, av_min_y = np.min(av_swept_path_points, axis=0)
        av_max_x, av_max_y = np.max(av_swept_path_points, axis=0)
        
        all_agents_now = scenario_data['all_agent_trajectories'][:, current_timestep, :]
        valid_mask = scenario_data['valid_mask'][:, current_timestep]

        for agent_idx, agent_state in enumerate(all_agents_now):
            if agent_idx == sdc_index or not valid_mask[agent_idx]:
                continue

            other_agent_box = get_bounding_box_vertices(agent_state)
            other_min_x, other_min_y = np.min(other_agent_box, axis=0)
            other_max_x, other_max_y = np.max(other_agent_box, axis=0)
            
            if (av_min_x < other_max_x and av_max_x > other_min_x and
                av_min_y < other_max_y and av_max_y > other_min_y):
                
                agent_id = int(scenario_data['object_ids'][agent_idx])
                agent_type_id = int(scenario_data['object_types'][agent_idx])
                agent_type_str = AGENT_TYPE_MAP.get(agent_type_id, "OTHER")
                
                return (True, agent_type_str, agent_id)

    except (ValueError, IndexError):
        return (False, "NONE", None)

    return (False, "NONE", None)

# --- NEW FULLY IMPLEMENTED FUNCTION ---

def get_top_k_threats(
    scenario_data: Dict,
    current_timestep: int,
    k: int = 3
) -> List[str]:
    """
    Identifies the top k most threatening agents to the AV at a given timestep.
    Threat is defined by a combination of proximity and convergence rate.
    
    Args:
        scenario_data: The full scenario data dictionary.
        current_timestep: The current timestep to analyze.
        k: The number of top threats to return.
        
    Returns:
        A list of formatted strings describing the top k threats.
    """
    threats = []
    
    try:
        sdc_index = scenario_data['sdc_track_index']
        all_agents_now = scenario_data['all_agent_trajectories'][:, current_timestep, :]
        valid_mask = scenario_data['valid_mask'][:, current_timestep]
        
        # Get the AV's current state for relative calculations
        if not valid_mask[sdc_index]:
            return [] # Cannot calculate threats if AV is not valid
        av_state = all_agents_now[sdc_index]
        av_pos = av_state[:2]
        av_vel = av_state[7:9]

        for agent_idx, agent_state in enumerate(all_agents_now):
            # Skip the AV itself and any invalid agents
            if agent_idx == sdc_index or not valid_mask[agent_idx]:
                continue

            # --- 1. Calculate Core Metrics ---
            agent_pos = agent_state[:2]
            agent_vel = agent_state[7:9]
            
            distance = np.linalg.norm(agent_pos - av_pos)
            
            # Don't consider agents that are very far away
            if distance > 100.0:
                continue

            relative_pos = agent_pos - av_pos
            relative_vel = agent_vel - av_vel
            
            # Rate of closure. Positive means getting closer.
            convergence_score = -np.dot(relative_pos, relative_vel)

            # --- 2. Calculate the "Threat Score" ---
            # A simple heuristic that heavily weights convergence, with proximity as a secondary factor.
            # We only consider agents that are actually getting closer (convergence > 0).
            if convergence_score > 0:
                # The threat score increases with convergence and decreases with distance.
                # Adding a small epsilon to distance to avoid division by zero.
                threat_score = convergence_score / (distance + 1e-6)
            else:
                threat_score = 0
            
            if threat_score > 0:
                agent_id = int(scenario_data['object_ids'][agent_idx])
                agent_type_id = int(scenario_data['object_types'][agent_idx])
                agent_type_str = AGENT_TYPE_MAP.get(agent_type_id, "OTHER")
                
                # The closing speed is the convergence score divided by distance.
                closing_speed_ms = threat_score 

                threats.append({
                    'score': threat_score,
                    'id': agent_id,
                    'type': agent_type_str,
                    'dist': distance,
                    'closing_speed': closing_speed_ms
                })

    except (ValueError, IndexError):
        # Return an empty list if any data is malformed
        return []

    # Sort all identified threats by their score in descending order
    sorted_threats = sorted(threats, key=lambda x: x['score'], reverse=True)
    
    # Format the top k threats into clean strings for the prompt
    output_list = []
    for threat in sorted_threats[:k]:
        output_list.append(
            f"{threat['type']} (ID: {threat['id']}): "
            f"dist={threat['dist']:.1f}m, "
            f"closing_speed={threat['closing_speed']:.1f}m/s"
        )
        
    return output_list