# File: run_renderer_comparison.py
# (Full updated script)

import os
from glob import glob
from PIL import Image
import numpy as np # Needed for pygame saving

import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.data_processing.waymo_parser import load_npz_scenario
from src.rendering.scenario_renderer import ScenarioRenderer
from src.rendering.matplotlib_renderer import render_matplotlib_snapshot # UPDATED import

import pygame

def main():
    print("--- Running Full Scene Snapshot Comparison ---")
    
    # Define the specific timestep we want to visualize
    SNAPSHOT_TIMESTEP = 20 # The 2.0 second mark

    try:
        config = load_config()
        npz_dir = config['data']['processed_npz_dir']
        validation_dir = os.path.join(npz_dir, 'validation')
        sample_files = glob(os.path.join(validation_dir, '*.npz'))
        
        if not sample_files:
            raise FileNotFoundError(f"No .npz files found in '{validation_dir}'.")
        
        scenario_path = sample_files[0]
        scenario_data = load_npz_scenario(scenario_path)
        scenario_id = scenario_data['scenario_id']
        print(f"Loaded scenario: {scenario_id} for snapshot at t={SNAPSHOT_TIMESTEP}")

        # --- Render with Pygame ---
        print("\n--- Rendering with Pygame ---")
        pygame_renderer = ScenarioRenderer(scenario_data)
        pygame_renderer.surface.fill(pygame_renderer.config['bg_color'])
        pygame_renderer._draw_static_map()
        pygame_renderer._draw_agents_at_timestep(SNAPSHOT_TIMESTEP) # NEW: Draw agents
        
        pygame_output_path = f"comparison_snapshot_{scenario_id}_pygame.png"
        
        frame_data = pygame.surfarray.array3d(pygame_renderer.surface)
        frame_data = np.rot90(frame_data, k=3)
        frame_data = np.fliplr(frame_data)
        img = Image.fromarray(frame_data)
        img.save(pygame_output_path)
        pygame_renderer.close()
        print(f"✅ Pygame snapshot saved to: {pygame_output_path}")

        # --- Render with Matplotlib ---
        print("\n--- Rendering with Matplotlib ---")
        matplotlib_output_path = f"comparison_snapshot_{scenario_id}_matplotlib.png"
        render_matplotlib_snapshot(scenario_data, SNAPSHOT_TIMESTEP, matplotlib_output_path) # UPDATED call
        
        print("\n--- Comparison Complete ---")
        print(f"Please open and compare the following two snapshot files:")
        print(f"1. {os.path.abspath(pygame_output_path)}")
        print(f"2. {os.path.abspath(matplotlib_output_path)}")

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"❌ Script Failed: {e}")

if __name__ == "__main__":
    main()