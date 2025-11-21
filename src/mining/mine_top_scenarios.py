# src/mining/mine_top_scenarios.py

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# Add project root to path for clean imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config

def calculate_peak_score(scores_data: np.lib.npyio.NpzFile, weights: dict) -> float:
    """
    Calculates a single "peak criticality" score for a scenario.

    This function combines the per-timestep heuristic scores using the weights
    from the config file and then finds the moment of maximum criticality.

    Args:
        scores_data: The loaded .npz file containing the per-heuristic scores.
        weights: A dictionary of weights for each heuristic component.

    Returns:
        A single float score representing the peak criticality of the scenario.
    """
    # Combine the individual heuristic scores using the provided weights
    final_scores = (
        weights['weight_volatility'] * scores_data['volatility'] +
        weights['weight_interaction'] * scores_data['interaction'] +
        weights['weight_lane_deviation'] * scores_data['lane_deviation'] +
        weights['weight_off_road'] * scores_data['off_road'] +
        weights['weight_density'] * scores_data['density']
    )

    # --- The Peak Score Calculation ---
    # We define the scenario's overall score as its moment of highest risk.
    # The max() function finds the single most critical timestep.
    # An alternative could be to average the top 5 scores to find moments
    # of sustained criticality, but max() is excellent for finding sharp,
    # long-tail events.
    peak_score = np.max(final_scores)
    
    return peak_score

def main():
    """
    Main function to load all pre-computed scores, rank them, and save the
    top N scenario IDs to a CSV file.
    """
    print("--- Running Top Scenario Miner ---")
    
    config = load_config()
    scores_dir = config['data']['criticality_scores_dir']
    weights = config['scoring']['heuristic']
    num_to_select = config['mining']['num_top_scenarios_to_select']

    # We are only interested in the validation set for our paper
    validation_scores_dir = os.path.join(scores_dir, 'validation')
    score_files = glob(os.path.join(validation_scores_dir, '*.npz'))

    if not score_files:
        print(f"❌ ERROR: No pre-computed score files found in '{validation_scores_dir}'.")
        print("   Please ensure you have run 'score_criticality_heuristic.py' first.")
        return

    print(f"Found {len(score_files)} scored scenarios to analyze.")
    
    results = []
    for score_path in tqdm(score_files, desc="Ranking Scenarios"):
        try:
            scenario_id = os.path.splitext(os.path.basename(score_path))[0]
            scores_data = np.load(score_path)
            
            peak_score = calculate_peak_score(scores_data, weights)
            
            results.append({'scenario_id': scenario_id, 'score': peak_score})
        except Exception as e:
            print(f"Warning: Could not process score file {score_path}. Error: {e}")
            continue
            
    if not results:
        print("❌ ERROR: No scenarios were successfully processed.")
        return

    # Convert to a DataFrame for easy sorting and saving
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='score', ascending=False)
    
    top_scenarios = df_sorted.head(num_to_select)
    
    # Save the final list to a CSV file
    output_dir = "data/mined_scenarios"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "critical_scenario_ids_v1.csv")
    top_scenarios.to_csv(output_path, index=False)
    
    print("\n--- Mining Complete ---")
    print(f"✅ Successfully ranked {len(df)} scenarios.")
    print(f"   Top {num_to_select} scenario IDs saved to: {output_path}")
    print("\n--- Top 5 Most Critical Scenarios ---")
    print(top_scenarios.head())
    print("-------------------------------------")

if __name__ == "__main__":
    main()