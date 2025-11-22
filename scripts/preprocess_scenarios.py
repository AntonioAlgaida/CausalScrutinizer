# File: preprocess_scenarios_v2.py
# (Updated to use the new semantic batch file)

import os
import sys
import pandas as pd
import traceback
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- HEADLESS MODE & PATH SETUP (unchanged) ---
os.environ["SDL_VIDEODRIVER"] = "dummy"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.data_processing.waymo_parser import load_npz_scenario
from src.rendering.scenario_renderer_v6 import ScenarioRendererV6

# --- Global Config & Worker Function (unchanged) ---
GLOBAL_CONFIG = None

def init_worker(config_path):
    """Initialize global config in each worker process."""
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = load_config(config_path)

def process_scenario_worker(scenario_id: str):
    """
    Worker function to render a single scenario.
    (This function's internal logic is robust and does not need to change.)
    """
    try:
        npz_dir = GLOBAL_CONFIG['data']['processed_npz_dir']
        output_base_dir = "outputs/preprocessed_scenarios_v2"
        scenario_output_dir = os.path.join(output_base_dir, scenario_id)
        gif_output_path = os.path.join(scenario_output_dir, "scenario.gif")

        if os.path.exists(gif_output_path):
            return "SKIP"

        scenario_path = os.path.join(npz_dir, 'training', f"{scenario_id}.npz")
        if not os.path.exists(scenario_path):
            return "MISSING"
        
        scenario_data = load_npz_scenario(scenario_path)
        os.makedirs(scenario_output_dir, exist_ok=True)

        renderer = ScenarioRendererV6(scenario_data, GLOBAL_CONFIG['renderer_v6'])
        renderer.render_to_gif(gif_output_path)
        renderer.close()
        
        return "DONE"
    except Exception as e:
        return f"ERROR: {str(e)}"

def main():
    print("============================================================")
    print("   STAGE 1: PREPROCESS SEMANTIC BATCH (V6 RENDERER)     ")
    print("============================================================")

    try:
        # --- 1. Setup ---
        config_path = os.path.join(PROJECT_ROOT, "configs/main_config.yaml")
        temp_config = load_config(config_path)
        
        # --- THE CRITICAL CHANGE IS HERE ---
        # Point to the new, semantically mined CSV file.
        mined_scenarios_csv = "data/mined_scenarios/golden_batch_semantic_training.csv"
        
        if not os.path.exists(mined_scenarios_csv):
            raise FileNotFoundError(f"Mined semantic batch CSV not found at '{mined_scenarios_csv}'. "
                                    "Please run 'mine_causal_complexity.py' first.")
        
        # --- 2. Load Work Queue ---
        df_scenarios = pd.read_csv(mined_scenarios_csv)
        scenario_ids = df_scenarios['scenario_id'].tolist()
        
        num_workers = temp_config['data'].get('num_workers', max(1, cpu_count() - 2))
        print(f"📂 Found semantic batch file. Scenarios to process: {len(scenario_ids)}")
        
        # Optional: Print a summary of the categories we've loaded
        print("\n--- Batch Composition ---")
        print(df_scenarios['category'].value_counts())
        print("-------------------------\n")
        
        print(f"🚀 Spawning {num_workers} worker processes...")

        # --- 3. Run Parallel Processing ---
        results = []
        with Pool(processes=num_workers, initializer=init_worker, initargs=(config_path,)) as pool:
            for result in tqdm(pool.imap_unordered(process_scenario_worker, scenario_ids), total=len(scenario_ids)):
                results.append(result)

        # --- 4. Summary ---
        skipped = sum(1 for r in results if r == "SKIP")
        done = sum(1 for r in results if r == "DONE")
        missing = sum(1 for r in results if r == "MISSING")
        errors = [r for r in results if r.startswith("ERROR")]

        print("\n--- Batch Preprocessing Complete ---")
        print(f"✅ Rendered: {done} new GIFs.")
        print(f"⏭️  Skipped (already exist): {skipped}")
        print(f"❓ Missing NPZ file: {missing}")
        print(f"❌ Errors:   {len(errors)}")
        
        if errors:
            print("\n--- Error Details ---")
            # To avoid spamming, we'll just show the first 5 errors
            for i, err in enumerate(errors[:5]):
                print(f"  {i+1}: {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more.")

    except Exception as e:
        print(f"\n❌ Fatal Script Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()