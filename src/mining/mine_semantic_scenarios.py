# File: src/mining/mine_causal_complexity.py

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count

# --- Add project root to path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.data_processing.waymo_parser import load_npz_scenario
from src.reasoning.prompt_builder import get_av_traffic_light_state_at_ts

# --- Category 1: VRU (Pedestrian/Cyclist) Crossing Score ---
def calculate_vru_crossing_score(scenario_data: dict) -> float:
    """
    --- V2 (Sanitized) ---
    Score is high if a Pedestrian/Cyclist is in front of the AV 
    and moving across its path. Ignores the initial noisy timesteps.
    """
    score = 0.0
    av_idx = scenario_data['sdc_track_index']
    
    # --- DATA SANITIZATION ---
    start_timestep = 3

    for t in range(start_timestep, min(50, scenario_data['all_agent_trajectories'].shape[1]), 2):
        if not scenario_data['valid_mask'][av_idx, t]: continue
        
        av_state = scenario_data['all_agent_trajectories'][av_idx, t]
        av_pos = av_state[:2]
        av_heading = av_state[6]
        av_dir = np.array([np.cos(av_heading), np.sin(av_heading)])
        av_perp = np.array([-av_dir[1], av_dir[0]])
        
        agents = scenario_data['all_agent_trajectories'][:, t, :]
        valid = scenario_data['valid_mask'][:, t]
        types = scenario_data['object_types']
        
        vru_indices = np.where((valid) & ((types == 2) | (types == 3)))[0]
        
        for idx in vru_indices:
            vru_pos = agents[idx, :2]
            vru_vel = agents[idx, 7:9]
            
            vec_to_vru = vru_pos - av_pos
            long_dist = np.dot(vec_to_vru, av_dir)
            lat_dist = np.dot(vec_to_vru, av_perp)
            
            if 0 < long_dist < 30.0 and abs(lat_dist) < 4.0:
                crossing_speed = abs(np.dot(vru_vel, av_perp))
                if crossing_speed > 0.5:
                    score = max(score, crossing_speed)

    # Normalize: A crossing speed > 2.0 m/s is a very high score.
    return min(score / 2.0, 1.0)

# --- Category 2: Cut-In / Lane Change Conflict Score ---
def calculate_cut_in_score(scenario_data: dict) -> float:
    """
    --- V2 (Sanitized) ---
    Score is high if another vehicle moves laterally INTO the AV's lane closely ahead.
    Ignores the initial noisy timesteps.
    """
    score = 0.0
    av_idx = scenario_data['sdc_track_index']
    
    # --- DATA SANITIZATION ---
    start_timestep = 3
    
    for t in range(start_timestep, min(50, scenario_data['all_agent_trajectories'].shape[1]), 2):
        if not scenario_data['valid_mask'][av_idx, t]: continue
        
        av_state = scenario_data['all_agent_trajectories'][av_idx, t]
        av_pos = av_state[:2]
        av_heading = av_state[6]
        av_dir = np.array([np.cos(av_heading), np.sin(av_heading)])
        av_perp = np.array([-av_dir[1], av_dir[0]])
        
        agents = scenario_data['all_agent_trajectories'][:, t, :]
        valid = scenario_data['valid_mask'][:, t]
        types = scenario_data['object_types']
        
        veh_indices = np.where((valid) & (types == 1))[0]
        
        for idx in veh_indices:
            if idx == av_idx: continue
            
            rel_pos = agents[idx, :2] - av_pos
            long_dist = np.dot(rel_pos, av_dir)
            lat_dist = np.dot(rel_pos, av_perp)
            
            if 5.0 < long_dist < 25.0 and 1.5 < abs(lat_dist) < 5.0:
                rel_vel = agents[idx, 7:9] - av_state[7:9]
                lat_closing_speed = -np.dot(rel_vel, av_perp) * np.sign(lat_dist)
                
                if lat_closing_speed > 0.5:
                    score = max(score, lat_closing_speed)

    # Normalize: A lateral closing speed > 2.0 m/s is a very high score.
    return min(score / 2.0, 1.0)

# --- Category 3: Emergency Braking Score ---
def calculate_braking_score(scenario_data: dict) -> float:
    """
    --- V2 (Sanitized) ---
    Finds the hardest braking event, ignoring the noisy initialization period.
    """
    av_idx = scenario_data['sdc_track_index']
    traj = scenario_data['all_agent_trajectories'][av_idx]
    valid = scenario_data['valid_mask'][av_idx]
    
    # --- DATA SANITIZATION ---
    # Ignore the first 3 timesteps (0.3s) to avoid initialization artifacts.
    start_timestep = 3
    if len(traj) <= start_timestep:
        return 0.0

    sanitized_traj = traj[start_timestep:]
    sanitized_valid = valid[start_timestep:]

    if np.sum(sanitized_valid) < 2:
        return 0.0
    
    speeds = np.linalg.norm(sanitized_traj[:, 7:9], axis=1)
    accels = np.gradient(speeds, 0.1)
    
    # Find the minimum acceleration (max deceleration) only in the valid, sanitized data
    min_accel = np.min(accels[sanitized_valid])
    
    # Score is high for any braking event harder than -4.0 m/s^2
    if min_accel < -4.0:
        # Normalize: -4.0 is a score of 0.0, -8.0 is a score of 1.0
        return min(abs(min_accel + 4.0) / 4.0, 1.0)
        
    return 0.0

# --- Metric 1: Occlusion Score ---
# In src/mining/mine_causal_complexity.py

def calculate_occlusion_score(scenario_data: dict) -> float:
    """
    --- V2 (Sanitized) ---
    Detects if large vehicles are blocking the line of sight to crosswalks.
    Ignores the initial noisy timesteps.
    """
    score = 0.0
    map_layers = scenario_data.get('map_layers', {})
    crosswalks = map_layers.get('crosswalks', [])
    
    if not crosswalks:
        return 0.0

    cw_points = np.vstack([cw[:, :2] for cw in crosswalks if len(cw) > 0])
    if len(cw_points) == 0:
        return 0.0
    cw_tree = cKDTree(cw_points)

    # --- DATA SANITIZATION ---
    start_timestep = 3
    
    for t in range(start_timestep, min(50, scenario_data['all_agent_trajectories'].shape[1]), 5):
        av_idx = scenario_data['sdc_track_index']
        if not scenario_data['valid_mask'][av_idx, t]: continue
        av_pos = scenario_data['all_agent_trajectories'][av_idx, t, :2]
        
        agents = scenario_data['all_agent_trajectories'][:, t, :]
        valid = scenario_data['valid_mask'][:, t]
        types = scenario_data['object_types']
        
        occluder_mask = (types == 1) & valid & (agents[:, 3] > 4.8)
        occluder_indices = np.where(occluder_mask)[0]
        
        for idx in occluder_indices:
            if idx == av_idx: continue
            occ_pos = agents[idx, :2]
            
            dist_to_occ = np.linalg.norm(occ_pos - av_pos)
            if dist_to_occ > 25.0: continue
            
            dist_to_cw, _ = cw_tree.query(occ_pos, k=1)
            
            if dist_to_cw < 10.0:
                score += 1.0

    # Normalize: 4 frames of occlusion (across 2 seconds of checks) = max score
    return min(score / 4.0, 1.0)

GLOBAL_CONFIG = None

def init_worker(config_path):
    """Initializer for each worker process."""
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = load_config(config_path)

def process_scenario_worker(npz_path: str) -> dict | None:
    """
    Worker function that loads one scenario, calculates all semantic scores,
    and returns a dictionary of results.
    """
    try:
        sid = os.path.splitext(os.path.basename(npz_path))[0]
        data = load_npz_scenario(npz_path)
        
        # Calculate all semantic scores for this single scenario
        vru_score = calculate_vru_crossing_score(data)
        cutin_score = calculate_cut_in_score(data)
        brake_score = calculate_braking_score(data)
        occlusion_score = calculate_occlusion_score(data)
        
        # Return a dictionary with all the results
        return {
            'scenario_id': sid,
            'vru_score': vru_score,
            'cutin_score': cutin_score,
            'brake_score': brake_score,
            'occlusion_score': occlusion_score
        }
    except Exception:
        # If any scenario fails to load or process, return None
        return None

# --- NEW: The Main Orchestrator ---

def main():
    """
    Main orchestrator that finds all scenarios, processes them in parallel,
    and then performs the final bucketing and saving.
    """
    print("--- Running Parallelized Causal Sieve Mining ---")
    
    config_path = os.path.join(PROJECT_ROOT, 'configs/main_config.yaml')
    config = load_config(config_path)
    npz_dir = config['data']['processed_npz_dir']
    
    # --- Allow user to select which dataset to mine ---
    dataset_to_mine = input("Which dataset would you like to mine? [training/validation]: ").strip().lower()
    if dataset_to_mine not in ['training', 'validation']:
        print("❌ Invalid selection. Please enter 'training' or 'validation'.")
        return
        
    target_dir = os.path.join(npz_dir, dataset_to_mine)
    all_npz_paths = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.npz')]
    
    if not all_npz_paths:
        print(f"❌ No .npz files found in '{target_dir}'.")
        return

    num_workers = config['data'].get('num_workers', max(1, cpu_count() - 2))
    print(f"Found {len(all_npz_paths)} scenarios in '{dataset_to_mine}' set.")
    print(f"Spawning {num_workers} worker processes...")

    # --- Run the Parallel Processing ---
    results = []
    with Pool(processes=num_workers, initializer=init_worker, initargs=(config_path,)) as pool:
        # Use imap_unordered for efficiency, as order doesn't matter yet
        for result in tqdm(pool.imap_unordered(process_scenario_worker, all_npz_paths), total=len(all_npz_paths)):
            if result is not None: # Only append successful results
                results.append(result)

    if not results:
        print("❌ No scenarios were successfully processed.")
        return

    df = pd.DataFrame(results)
    print(f"\nSuccessfully processed {len(df)} scenarios.")
    
    # --- Perform the Final Bucketing Strategy (unchanged) ---
    print("Applying semantic bucketing to find the Golden Batch...")
    
    df_vru = df.sort_values('vru_score', ascending=False).head(30)
    df_vru['category'] = 'VRU_Crossing'
    
    df_cutin = df.sort_values('cutin_score', ascending=False).head(30)
    df_cutin['category'] = 'Cut_In'
    
    # ... (rest of the bucketing logic is the same)
    df_brake = df.sort_values('brake_score', ascending=False)
    df_brake = df_brake[~df_brake['scenario_id'].isin(df_vru['scenario_id'])]
    df_brake = df_brake[~df_brake['scenario_id'].isin(df_cutin['scenario_id'])].head(20)
    df_brake['category'] = 'Hard_Braking'
    
    df_occ = df.sort_values('occlusion_score', ascending=False)
    df_occ = df_occ[~df_occ['scenario_id'].isin(df_vru['scenario_id'])]
    df_occ = df_occ[~df_occ['scenario_id'].isin(df_cutin['scenario_id'])]
    df_occ = df_occ[~df_occ['scenario_id'].isin(df_brake['scenario_id'])].head(20)
    df_occ['category'] = 'Occlusion'

    final_df = pd.concat([df_vru, df_cutin, df_brake, df_occ])
    
    # --- Save the Final CSV ---
    output_filename = f"golden_batch_semantic_{dataset_to_mine}.csv"
    output_path = os.path.join("data/mined_scenarios", output_filename)
    final_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Semantic Mining Complete for '{dataset_to_mine}' set.")
    print(f"   - VRU Scenarios: {len(df_vru)}")
    print(f"   - Cut-In Scenarios: {len(df_cutin)}")
    print(f"   - Braking Scenarios: {len(df_brake)}")
    print(f"   - Occlusion Scenarios: {len(df_occ)}")
    print(f"   Total Unique Scenarios: {len(final_df)}")
    print(f"   Saved final batch to: {output_path}")

if __name__ == "__main__":
    # NOTE: You will need to copy the full implementation of your
    # heuristic functions (calculate_vru_crossing_score, etc.)
    # into this script for it to be self-contained.
    main()