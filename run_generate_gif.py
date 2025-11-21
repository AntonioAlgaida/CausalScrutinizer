# File: run_generate_gif.py

import os
from glob import glob
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.data_processing.waymo_parser import load_npz_scenario
from src.rendering.scenario_renderer import ScenarioRenderer

def main():
    """
    Loads a scenario and uses the full V2 ScenarioRenderer to generate a
    high-fidelity animated GIF, including all dynamic elements.
    """
    print("--- Running High-Fidelity GIF Generation Script ---")

    try:
        config = load_config()
        npz_dir = config['data']['processed_npz_dir']
        
        # Find a scenario at an intersection to test with.
        # Scenarios with more map features are more likely to have lights.
        # Let's use the one we've been testing with as it's a good example.
        scenario_id_to_render = "72ed65984ad6f37f"
        scenario_path = os.path.join(npz_dir, 'validation', f"{scenario_id_to_render}.npz")

        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Specified scenario file not found: {scenario_path}")
        
        scenario_data = load_npz_scenario(scenario_path)
        
        # Define where to save the output
        output_dir = "outputs/rendered_gifs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"v2_render_{scenario_id_to_render}.gif")

        # --- Initialize and run the renderer ---
        renderer = ScenarioRenderer(scenario_data)
        renderer.render_to_gif(output_path)
        renderer.close()
        
        print("\n--- GIF Generation Complete ---")
        print(f"Please view the output file: {os.path.abspath(output_path)}")

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"❌ Script Failed: {e}")

if __name__ == "__main__":
    main()