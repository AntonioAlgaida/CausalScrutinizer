# src/data_processing/find_critical_scenarios.py

import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil
from scipy.spatial import cKDTree

# Add project root to path for clean imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config
# Note: geometry is not used yet, but is ready for future PET implementation
from src.utils import geometry 

# --- Global Configuration for Worker Processes ---
CONFIG = None

def init_worker(config_path: str):
    """Initializer for each worker process to load the config."""
    global CONFIG
    CONFIG = load_config(config_path)

# --- Pillar 1: EGO-CENTRIC DYNAMICS ---

def calculate_kinematic_volatility_scores(sdc_route: np.ndarray) -> np.ndarray:
    """
    --- V2: Comprehensive Volatility Score ---
    Pillar 1: Ego-Centric Volatility
    Calculates a score based on three key kinematic indicators:
    1.  High Longitudinal Deceleration (Hard Braking)
    2.  High Jerk (Reactive/Unstable Plan)
    3.  High Yaw Acceleration (Sudden Swerve)
    """
    # Timestep duration
    dt = 0.1
    
    # --- Calculate Longitudinal Kinematics ---
    velocities = sdc_route[:, 7:9]  # vx, vy
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = np.diff(speeds, prepend=speeds[0]) / dt
    jerks = np.diff(accelerations, prepend=accelerations[0]) / dt
    
    # --- Calculate Rotational Kinematics ---
    yaws = sdc_route[:, 6]
    yaw_rates = np.diff(np.unwrap(yaws), prepend=yaws[0]) / dt
    yaw_accelerations = np.diff(yaw_rates, prepend=yaw_rates[0]) / dt
    
    # --- Normalize each component to a [0, 1] score ---
    # We use a fixed, physically-motivated scale to ensure scores are comparable.
    
    # Metric 1: Hard Braking Score
    # We only care about deceleration, so we clip positive accelerations to 0.
    # A deceleration of -4.0 m/s^2 is a very hard brake.
    hard_braking_score = np.clip(-accelerations / 4.0, 0, 1)
    
    # Metric 2: Jerk Score
    # A jerk > 8.0 m/s^3 is very high (emergency brake application).
    jerk_score = np.clip(np.abs(jerks) / 8.0, 0, 1)
    
    # Metric 3 (Proxy): Yaw Acceleration Score (Sudden Swerve)
    # A yaw acceleration > 3.0 rad/s^2 is a very sudden turn/swerve.
    yaw_accel_score = np.clip(np.abs(yaw_accelerations) / 3.0, 0, 1)
    
    # Metric 4: Longitudinal Acceleration Score (Smoothness)
    # A longitudinal acceleration > 3.0 m/s^2 is a very sudden acceleration
    longitudinal_accel_score = np.clip(np.abs(accelerations) / 3.0, 0, 1)
    
    # --- Combine the scores with a weighted maximum ---
    # We want hard braking to be the most important factor. The other metrics
    # indicate "interestingness," but braking indicates direct safety relevance.
    # We can take the maximum of the "interestingness" scores, and then combine
    # that with the braking score, giving braking a higher weight.

    # First, find the peak "non-braking" volatility score
    other_volatility = np.maximum.reduce([longitudinal_accel_score, jerk_score, yaw_accel_score])
    
    # Combine with braking, giving braking a higher effective weight.
    # For example, we can average them but count braking twice.
    # A simpler and more direct way is to take a weighted max:
    volatility_score = np.maximum(hard_braking_score, 0.75 * other_volatility)
    
    return volatility_score

def calculate_off_road_proximity_scores(sdc_route: np.ndarray, map_polylines: list, map_polyline_types: list) -> np.ndarray:
    """Calculates a score based on the SDC's proximity to road boundaries."""
    num_timesteps = sdc_route.shape[0]
    off_road_scores = np.zeros(num_timesteps, dtype=np.float32)
    boundary_type_ids = {21, 22} # Road edge, Median
    
    boundary_polylines = [p for p, t in zip(map_polylines, map_polyline_types) if t in boundary_type_ids]
    if not boundary_polylines: return off_road_scores

    all_boundary_points = np.vstack([p[:, :2] for p in boundary_polylines if p.shape[0] > 0])
    if all_boundary_points.shape[0] == 0: return off_road_scores

    kdtree = cKDTree(all_boundary_points)
    min_distances, _ = kdtree.query(sdc_route[:, :2], k=1)
    
    proximity_threshold = 1.0  # Score increases sharply within 1 meter
    score_mask = min_distances < proximity_threshold
    off_road_scores[score_mask] = 1.0 - (min_distances[score_mask] / proximity_threshold)
    
    return off_road_scores

def calculate_lane_deviation_scores(sdc_route: np.ndarray, map_polylines: list, map_polyline_types: list) -> np.ndarray:
    """Calculates a score based on the SDC's lateral distance to the nearest lane centerline."""
    num_timesteps = sdc_route.shape[0]
    lane_deviation_scores = np.zeros(num_timesteps, dtype=np.float32)
    lane_polylines = [p for p, t in zip(map_polylines, map_polyline_types) if t in {1, 2, 3}]

    if not lane_polylines: return lane_deviation_scores

    all_lane_points = np.vstack([p[:, :2] for p in lane_polylines if p.shape[0] > 0])
    if all_lane_points.shape[0] == 0: return lane_deviation_scores

    kdtree = cKDTree(all_lane_points)
    min_distances, _ = kdtree.query(sdc_route[:, :2], k=1)
    
    max_dist_for_score = 2.0 # A 2m deviation from any centerline point is high
    return np.clip(min_distances / max_dist_for_score, 0, 1)

# --- Pillar 2: INTERACTION DYNAMICS ---

def calculate_interaction_scores_vectorized(all_trajectories: np.ndarray, valid_mask: np.ndarray, sdc_track_index: int) -> np.ndarray:
    """Calculates a score based on how directly other agents are moving towards the SDC."""
    num_agents, num_timesteps, _ = all_trajectories.shape
    sdc_traj_expanded = np.expand_dims(all_trajectories[sdc_track_index], axis=0)
    
    relative_pos = all_trajectories[:, :, :2] - sdc_traj_expanded[:, :, :2]
    relative_vel = all_trajectories[:, :, 7:9] - sdc_traj_expanded[:, :, 7:9]
    
    dot_products = np.einsum('ati,ati->at', relative_pos, relative_vel)
    convergence_risk = -np.clip(dot_products, None, 0)
    
    valid_other_agent_mask = valid_mask.copy()
    valid_other_agent_mask[sdc_track_index, :] = False
    convergence_risk[~valid_other_agent_mask] = 0
    
    interaction_scores_raw = np.max(convergence_risk, axis=0)
    return np.clip(interaction_scores_raw / 200.0, 0, 1) # Normalize by a high-risk value

def calculate_social_density_scores(valid_mask: np.ndarray) -> np.ndarray:
    """Calculates a simple score based on the number of agents in the scene."""
    num_agents = np.sum(valid_mask, axis=0)
    return np.clip(num_agents / 20.0, 0, 1) # Normalize by 20 agents

# --- Main Worker & Orchestrator ---

def process_scenario(npz_path: str) -> bool:
    """Processes a single scenario file and saves its heuristic scores."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        scenario_id = data['scenario_id'].item()

        # Calculate all heuristic components
        volatility = calculate_kinematic_volatility_scores(data['sdc_route'])
        interaction = calculate_interaction_scores_vectorized(data['all_agent_trajectories'], data['valid_mask'], data['sdc_track_index'])
        density = calculate_social_density_scores(data['valid_mask'])
        off_road = calculate_off_road_proximity_scores(data['sdc_route'], list(data['map_polylines']), list(data['map_polyline_types']))
        lane_deviation = calculate_lane_deviation_scores(data['sdc_route'], list(data['map_polylines']), list(data['map_polyline_types']))
        
        # Save all individual scores for later tuning and analysis
        output_subdir = "validation" # We are only processing the validation set
        output_dir = os.path.join(CONFIG['data']['criticality_scores_dir'], output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{scenario_id}.npz")
        np.savez_compressed(
            output_path,
            volatility=volatility.astype(np.float32),
            interaction=interaction.astype(np.float32),
            off_road=off_road.astype(np.float32),
            lane_deviation=lane_deviation.astype(np.float32),
            density=density.astype(np.float32)
        )
        return True
    except Exception as e:
        print(f"❌ ERROR processing {npz_path}: {e}")
        return False

def main():
    """Finds all validation .npz files, processes them in parallel, and saves scores."""
    config_path = os.path.join(PROJECT_ROOT, 'configs/main_config.yaml')
    global CONFIG
    CONFIG = load_config(config_path)

    npz_base_dir = CONFIG['data']['processed_npz_dir']
    # --- We are ONLY using the validation set, as per our scientific principles ---
    validation_npz_paths = glob(os.path.join(npz_base_dir, 'validation', '*.npz'))

    if not validation_npz_paths:
        print(f"❌ ERROR: No .npz files found in the validation directory: {os.path.join(npz_base_dir, 'validation')}")
        return

    output_dir = CONFIG['data']['criticality_scores_dir']
    if os.path.exists(output_dir):
        response = input(f"❗ WARNING: Output directory '{output_dir}' exists. Delete and restart? [y/N]: ")
        if response.lower() == 'y':
            shutil.rmtree(output_dir)
        else:
            print("Aborting.")
            return

    num_workers = CONFIG['data'].get('num_workers', cpu_count())
    print(f"\nFound {len(validation_npz_paths)} validation scenarios to score.")
    print(f"Using {num_workers} worker processes.")

    with Pool(processes=num_workers, initializer=init_worker, initargs=(config_path,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_scenario, validation_npz_paths), total=len(validation_npz_paths)))

    success_count = sum(results)
    print("\n--- Heuristic Scoring Complete ---")
    print(f"✅ Successfully processed and saved scores for {success_count} scenarios.")
    print(f"❌ Failed scenarios: {len(results) - success_count}")
    print(f"   Detailed per-timestep scores saved to: {output_dir}")

if __name__ == '__main__':
    main()