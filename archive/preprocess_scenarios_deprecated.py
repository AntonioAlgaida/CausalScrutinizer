# File: preprocess_scenarios.py

import os
import sys
import pandas as pd
import traceback
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process
import traceback

# --- HEADLESS MODE ---
# Vital for multiprocessing with Pygame. Prevents opening windows for every process.
os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.data_processing.waymo_parser import load_npz_scenario
from src.rendering.scenario_renderer_v6 import ScenarioRendererV6

# --- Global Config for Workers ---
GLOBAL_CONFIG = None

def init_worker(config_path):
    """Initialize global config in each worker process."""
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = load_config(config_path)

def process_scenario_worker(scenario_id: str):
    """
    Worker function to render a single scenario.
    """
    try:
        # 1. Setup Paths
        npz_dir = GLOBAL_CONFIG['data']['processed_npz_dir']
        output_base_dir = "outputs/preprocessed_scenarios"
        
        scenario_output_dir = os.path.join(output_base_dir, scenario_id)
        gif_output_path = os.path.join(scenario_output_dir, "scenario.gif")

        # 2. Resume Logic: Skip if GIF exists
        if os.path.exists(gif_output_path):
            return f"SKIP: {scenario_id}"

        # 3. Load Data
        scenario_path = os.path.join(npz_dir, 'validation', f"{scenario_id}.npz")
        if not os.path.exists(scenario_path):
            return f"MISSING: {scenario_id}"
        
        scenario_data = load_npz_scenario(scenario_path)

        # 4. Create Directory
        os.makedirs(scenario_output_dir, exist_ok=True)

        # 6. Render GIF using V6 Engine
        # Note: We initialize the renderer INSIDE the worker.
        # Pygame surfaces cannot be pickled, so we can't pass a renderer object.
        renderer = ScenarioRendererV6(scenario_data, GLOBAL_CONFIG['renderer_v6'])
        renderer.render_to_gif(gif_output_path)
        renderer.close()
        
        return f"DONE: {scenario_id}"

    except Exception as e:
        print(traceback.format_exc())
        # Return error message instead of crashing
        return f"ERROR: {scenario_id} - {str(e)}"

def main():
    print("============================================================")
    print("   STAGE 1: PREPROCESS SCENARIOS (V6 RENDERER + MULTIPROC)  ")
    print("============================================================")

    try:
        # 1. Setup
        config_path = os.path.join(PROJECT_ROOT, "configs/main_config.yaml")
        if not os.path.exists(config_path):
             raise FileNotFoundError("Config file not found.")
             
        # Load config locally just to get paths for the CSV
        temp_config = load_config(config_path)
        mined_scenarios_csv = "data/mined_scenarios/critical_scenario_ids_v1.csv"
        
        if not os.path.exists(mined_scenarios_csv):
            raise FileNotFoundError(f"Mined CSV not found at '{mined_scenarios_csv}'.")
        
        # 2. Load Work Queue
        df_scenarios = pd.read_csv(mined_scenarios_csv)
        scenario_ids = df_scenarios['scenario_id'].tolist()
        
        num_workers = max(1, cpu_count() - 2) # Leave 2 cores for OS/System
        print(f"📂 Scenarios to process: {len(scenario_ids)}")
        print(f"🚀 Spawning {num_workers} worker processes...")

        # 3. Run Parallel Processing
        results = []
        with Pool(processes=num_workers, initializer=init_worker, initargs=(config_path,)) as pool:
            # imap_unordered is faster as we don't care about order
            for result in tqdm(pool.imap_unordered(process_scenario_worker, scenario_ids), total=len(scenario_ids)):
                results.append(result)

        # 4. Summary
        skipped = sum(1 for r in results if r.startswith("SKIP"))
        done = sum(1 for r in results if r.startswith("DONE"))
        errors = [r for r in results if r.startswith("ERROR")]
        missing = sum(1 for r in results if r.startswith("MISSING"))

        print("\n--- Batch Processing Complete ---")
        print(f"✅ Rendered: {done}")
        print(f"⏭️  Skipped:  {skipped}")
        print(f"❌ Errors:   {len(errors)}")
        print(f"❓ Missing:  {missing}")
        
        if errors:
            print("\nError Details:")
            for err in errors:
                print(err)

    except Exception as e:
        print(f"\n❌ Fatal Script Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()