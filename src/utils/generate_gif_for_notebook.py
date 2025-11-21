import os
import sys
import numpy as np

# --- Add project root to path for our src imports ---
# This is the key change. We make the script aware of its own location
# and find the project root from there.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config # Corrected import from your file structure
from src.data_processing.waymo_parser import load_npz_scenario
from src.rendering.scenario_renderer import ScenarioRenderer

def render_gif_for_scenario(scenario_id: str, output_dir: str):
    """
    Finds, loads, and renders a single scenario ID to a GIF using the
    high-fidelity ScenarioRenderer.
    """
    # --- CORRECTED PATH LOGIC ---
    # Now, we build the config path from the known PROJECT_ROOT.
    config_path = os.path.join(PROJECT_ROOT, 'configs/main_config.yaml')
    config = load_config(config_path=config_path)
    
    npz_dir = config['data']['processed_npz_dir']
    
    # We assume we are working with the validation set for this tuning task
    scenario_path = os.path.join(npz_dir, 'validation', f"{scenario_id}.npz")

    if not os.path.exists(scenario_path):
        print(f"❌ ERROR: Raw data file for {scenario_id} not found at {scenario_path}.")
        return None

    print(f"\n--- Rendering GIF for: {scenario_id} ---")
    try:
        scenario_data = load_npz_scenario(scenario_path)
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{scenario_id}.gif")

        # Initialize and run the pygame renderer
        renderer = ScenarioRenderer(scenario_data)
        renderer.render_to_gif(output_path)
        renderer.close()

        print(f"✅ GIF saved to '{output_path}'")
        return output_path
    except Exception as e:
        print(f"❌ An error occurred during rendering for {scenario_id}: {e}")
        return None

if __name__ == '__main__':
    # This allows testing the script directly from the command line
    if len(sys.argv) > 1:
        test_scenario_id = sys.argv[1]
        # When run directly, save to a subfolder of the project's output directory
        cli_output_dir = os.path.join(PROJECT_ROOT, 'outputs/cli_renders')
        render_gif_for_scenario(test_scenario_id, cli_output_dir)
    else:
        print("Usage: python -m src.utils.generate_gif_for_notebook <scenario_id>")