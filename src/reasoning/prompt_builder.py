# File: src/reasoning/prompt_builder.py

import numpy as np
from typing import Dict, Tuple

from scipy.spatial import cKDTree

def _is_feature_near_route(feature_polyline: np.ndarray, sdc_route: np.ndarray, threshold_m: float = 30.0) -> bool:
    """
    Checks if a map feature is within a certain distance of the SDC's route.

    Args:
        feature_polyline: The (N, 2) points of the map feature.
        sdc_route: The (T, 9) trajectory of the SDC.
        threshold_m: The maximum distance to be considered "near".

    Returns:
        True if the feature is near the route, False otherwise.
    """
    
    if feature_polyline.shape[0] == 0 or sdc_route.shape[0] == 0:
        return False
    
    # Create a k-d tree for the SDC route for fast nearest-neighbor lookups
    route_tree = cKDTree(sdc_route[:, :2])
    
    # Query the tree to find the distance from each point in the feature
    # to the nearest point on the SDC's route.
    distances, _ = route_tree.query(feature_polyline[:, :2], k=1)
    
    # If the minimum distance found is less than our threshold, the feature is "near".
    return np.min(distances) < threshold_m


# --- Helper Function 1: Scene Classifier ---
def _classify_scene_type(map_types: np.ndarray, has_proximal_tl: bool, has_proximal_stop_sign: bool) -> str:
    unique_types = set(map_types)
    
    if has_proximal_tl or has_proximal_stop_sign:
        return "Urban street"

    lane_center_types = [t for t in map_types if t in {1, 2, 3}]
    if not lane_center_types:
        return "Road Segment (Undefined Type)"

    freeway_lane_count = lane_center_types.count(1)
    if freeway_lane_count / len(lane_center_types) > 0.5:
        if 41 not in unique_types: # 41 = Crosswalk
            return "Highway/Freeway"
            
    return "Surface Street"

# --- Helper Function 2: Get AV Goal ---
def _get_av_goal(sdc_route: np.ndarray, initial_heading: float) -> str:
    """Infers the AV's high-level driving goal."""
    if sdc_route.shape[0] < 2:
        return "Unknown"

    # Compare the initial heading to the heading at a future point (e.g., 5 seconds)
    future_index = min(50, sdc_route.shape[0] - 1) # 5 seconds or last point
    future_heading = sdc_route[future_index, 6] # Column 6 is heading

    # Calculate the total change in heading, wrapped to [-pi, pi]
    heading_change = (future_heading - initial_heading + np.pi) % (2 * np.pi) - np.pi

    # Classify based on heading change (in radians)
    if abs(heading_change) < 0.35: # Approx 20 degrees
        return "Proceed Straight"
    elif heading_change > 0.35:
        return "Execute a Left Turn or Lane Change"
    else: # heading_change < -0.35
        return "Execute a Right Turn or Lane Change"

# --- Helper Function 3: Get Initial Traffic Light State ---
def _get_traffic_light_state_at_ts(dynamic_map_states: np.ndarray, timestep: int) -> str:
    TL_STATE_STOP = {1, 4, 7}
    TL_STATE_CAUTION = {2, 5, 8}
    TL_STATE_GO = {3, 6}
    
    if timestep >= dynamic_map_states.shape[0]:
        return "Unknown (OOB)"

    lights_at_ts = dynamic_map_states[timestep, :, :]
    valid_lights = lights_at_ts[lights_at_ts[:, 0] > 0]

    if valid_lights.shape[0] == 0:
        return "None Visible" # Explicitly state that none are visible at this moment

    # Here we assume the first valid light is the most relevant one
    first_light_state = int(valid_lights[0, 1])

    if first_light_state in TL_STATE_STOP: return "RED"
    elif first_light_state in TL_STATE_CAUTION: return "YELLOW"
    elif first_light_state in TL_STATE_GO: return "GREEN"
    else: return "UNKNOWN"

# --- Main V4 Builder Function ---
def build_prompt_v4(scenario_data: Dict[str, np.ndarray], key_timestep: int) -> Tuple[str, str, str]:
    """
    Builds the complete V4 prompt components (System, Ground-Truth Context, User Task).

    Args:
        scenario_data: The dictionary loaded by the waymo_parser.
        key_timestep: The critical timestep for analysis (e.g., from the mining stage).

    Returns:
        A tuple containing (system_prompt, ground_truth_context, user_task_prompt).
    """
    # --- 1. Define the Static Prompt Components ---
    system_prompt = """You are a meticulous and objective Autonomous Vehicle Safety Analyst. Your sole purpose is to audit driving scenarios based on the ground-truth context and visual evidence provided, and nothing more.

CRITICAL CONTEXT: The visualization is a simplified, top-down rendering of the ego-centric reality as perceived by an Autonomous Vehicle (AV) from the Waymo Open Motion Dataset.
- **The colors used are schematic for identification and do not represent real-world colors.** Rely only on the Visual Legend for their meaning.
- Occlusions are real: Objects may be hidden behind others. What you see is all the AV could see.
- Partial Views: You may only see the traffic lights relevant to the AV, not all lights at an intersection.
- Sensor Limitations: Objects can suddenly appear or disappear.

Your Core Directives:
1. Evidence First: Base every statement on either the 'Ground-Truth Context' text or the visual events. Do not invent details.
2. Ego-Centric Reality: Acknowledge the AV's limited perception when analyzing risks.
3. Think in Causal Chains: Connect actions to consequences to identify root causes of risk."""

    user_task_prompt = """
First, perform a detailed, step-by-step analysis of the scene before providing your final answer. Follow this exact process:

**Internal Monologue (Chain-of-Thought):**
1.  **Scene Description:** Describe the static environment in detail. What type of road is it (highway, intersection, etc.)? What map elements are visible (lane lines, crosswalks, traffic lights)?
2.  **Dynamic Analysis (Frame-by-Frame):** For each keyframe provided, describe the actions of the Magenta AV and any other relevant agents (vehicles, pedestrians). Meticulously track their changes in position, speed, and heading.
3.  **Synthesize Key Events:** Based on your frame-by-frame analysis, summarize the most critical event or interaction that defines this scenario.

**Final Answer:**
Now, based on your detailed analysis above, provide your final answer in the following three-step format:

**Step 1: Factual Event Chronology.**
Concisely summarize the sequence of events you identified in your dynamic analysis.

**Step 2: Causal Risk Identification.**
Based on the key events, identify the top 2-3 primary causal risks for the AV. Explain the causal chain for each.

**Step 3: Optimal Action Recommendation.**
Based on your risk analysis, what is the single, objectively safest action for the Magenta AV to take? Justify your recommendation.
"""

    # --- 2. Build the Dynamic Ground-Truth Context String ---
    context_lines = [f"--- Ground-Truth Context ---"]

    sdc_index = scenario_data['sdc_track_index']
    sdc_state_at_key_ts = scenario_data['all_agent_trajectories'][sdc_index, key_timestep, :]
    sdc_route = scenario_data['sdc_route']

    # AV State & Goal
    speed_ms = np.linalg.norm(sdc_state_at_key_ts[7:9])
    goal = _get_av_goal(sdc_route, sdc_state_at_key_ts[6])
    context_lines.append(f"\n[AV State at Key Timestep (t={key_timestep})]")
    context_lines.append(f"- Speed: {speed_ms:.1f} m/s")
    context_lines.append(f"- Goal: {goal}")

    # Scene Composition
    object_types = scenario_data['object_types']
    num_other_vehicles = max(0, np.count_nonzero(object_types == 1) - 1)
    num_peds = np.count_nonzero(object_types == 2)
    num_cycs = np.count_nonzero(object_types == 3)
    context_lines.append("\n[Scene Composition]")
    context_lines.append(f"- Other Vehicles: {num_other_vehicles}")
    context_lines.append(f"- Pedestrians: {num_peds}{'' if num_peds > 0 else ' (NOT PRESENT)'}")
    context_lines.append(f"- Cyclists: {num_cycs}{'' if num_cycs > 0 else ' (NOT PRESENT)'}")

    # Environmental Rules (with proximity checks)
    # map_types = scenario_data['map_polyline_types']
    # map_polylines = scenario_data['map_polylines']
    
    # has_proximal_stop_sign = any(
    #     _is_feature_near_route(map_polylines[i], sdc_route)
    #     for i, p_type in enumerate(map_types) if p_type == 31
    # )

    # dynamic_map = scenario_data.get('dynamic_map_states', np.array([]))
    # has_proximal_tl = False
    # if dynamic_map.size > 0:
    #     tl_stop_points = dynamic_map[:, :, 2:4].reshape(-1, 2)
    #     valid_tl_points = tl_stop_points[np.linalg.norm(tl_stop_points, axis=1) > 1]
    #     if valid_tl_points.shape[0] > 0:
    #         has_proximal_tl = _is_feature_near_route(valid_tl_points, sdc_route)

    # scene_type = _classify_scene_type(map_types, has_proximal_tl, has_proximal_stop_sign)
    # has_crosswalks = any(_is_feature_near_route(map_polylines[i], sdc_route) for i, p_type in enumerate(map_types) if p_type == 41)

    # context_lines.append("\n[Environmental Rules]")
    # # context_lines.append(f"- Scene Type: {scene_type}")
    # context_lines.append(f"- Relevant Traffic Lights: {'PRESENT' if has_proximal_tl else 'NOT PRESENT'}")
    # if has_proximal_tl:
    #     tl_state = _get_traffic_light_state_at_ts(dynamic_map, key_timestep)
    #     context_lines.append(f"- Relevant Traffic Light at t={key_timestep}: {tl_state}")
    
    # context_lines.append(f"- Relevant Stop Signs: {'PRESENT' if has_proximal_stop_sign else 'NOT PRESENT'}")
    # context_lines.append(f"- Relevant Crosswalks: {'PRESENT' if has_crosswalks else 'NOT PRESENT'}")
    
    # context_lines.append("----------------------------")
    # ground_truth_context = "\n".join(context_lines)

    return system_prompt, None, user_task_prompt