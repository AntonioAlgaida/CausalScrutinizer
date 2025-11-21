# File: src/reasoning/prompt_builder_v5.py

import numpy as np
from typing import Dict

def get_av_traffic_light_state_at_ts(scenario_data: Dict, timestep: int) -> str:
    """
    --- FINAL GEOMETRIC VERSION ---
    Finds the traffic light state that is most relevant to the AV using a
    proximity and alignment-based geometric check.
    """
    dynamic_states = scenario_data.get('dynamic_map_states')
    if dynamic_states is None or timestep >= dynamic_states.shape[0]:
        return "NOT APPLICABLE"

    # 1. Get the AV's current state (position and heading)
    try:
        av_state = scenario_data['all_agent_trajectories'][scenario_data['sdc_track_index'], timestep, :]
        av_pos = av_state[:2]
        av_heading_rad = av_state[6]
        av_heading_vec = np.array([np.cos(av_heading_rad), np.sin(av_heading_rad)])
    except IndexError:
        return "NOT APPLICABLE"

    # 2. Iterate through all active traffic lights in the current timestep
    lights_at_ts = dynamic_states[timestep, :, :]
    valid_lights = lights_at_ts[lights_at_ts[:, 0] > 0]
    
    if valid_lights.shape[0] == 0:
        return "NOT VISIBLE OR NOT APPLICABLE"

    closest_relevant_light = {
        'dist': float('inf'),
        'state': "NOT VISIBLE OR NOT APPLICABLE"
    }

    for light in valid_lights:
        state_enum = int(light[1])
        stop_point = light[2:4]
        
        vec_to_light = stop_point - av_pos
        dist_to_light = np.linalg.norm(vec_to_light)
        
        # --- Condition A: Proximity Check ---
        # Ignore lights that are too far away (e.g., > 70 meters)
        if dist_to_light > 70.0:
            continue
            
        # --- Condition B: Alignment Check ---
        # Check if the light is generally in front of the AV.
        # The dot product will be positive if the angle between the vectors is < 90 degrees.
        if np.dot(vec_to_light, av_heading_vec) < 0:
            continue # Light is behind the AV

        # --- This is a candidate light. Check if it's the closest one yet. ---
        if dist_to_light < closest_relevant_light['dist']:
            closest_relevant_light['dist'] = dist_to_light
            if state_enum in {1, 4, 7}: closest_relevant_light['state'] = "RED"
            elif state_enum in {2, 5, 8}: closest_relevant_light['state'] = "YELLOW"
            elif state_enum in {3, 6}: closest_relevant_light['state'] = "GREEN"

    return closest_relevant_light['state']