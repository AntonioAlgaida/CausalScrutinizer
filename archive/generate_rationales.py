# File: generate_rationales.py

import os
import sys
from PIL import Image
import base64
from io import BytesIO
import pandas as pd # <-- ADDED for reading the CSV
from tqdm import tqdm # <-- ADDED for a nice progress bar

# --- Add project root to path for our src imports ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from src.utils.config_loader import load_config
from src.data_processing.waymo_parser import load_npz_scenario
from src.reasoning.prompt_builder import build_prompt_v4

# --- Helper function (unchanged) ---
def pil_image_to_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def main():
    """
    --- UPDATED FOR BATCH PROCESSING ---
    Main script for Stage 2: Rationale Generation.
    This script iterates through all preprocessed scenario IDs, loads the assets
    for each, builds the final V4 prompt, and runs VLM inference to save the rationale.
    """
    # --- 1. CONFIGURATION AND SETUP ---
    print("--- Running Stage 2: Batch Rationale Generation ---")
    
    config = load_config()
    preprocessed_dir = "outputs/preprocessed_scenarios"    
    rationales_output_dir = "outputs/rationales_other" # "outputs/rationales_gemma-3-12b-it-UD-Q8_K_XL", "outputs/rationales_gemma3-12b-Q4"
    mined_scenarios_csv = "data/mined_scenarios/critical_scenario_ids_v1.csv"
    
    # --- MODIFIED: Load the list of scenario IDs to process ---
    if not os.path.exists(mined_scenarios_csv):
        raise FileNotFoundError(f"Mined scenarios file not found: '{mined_scenarios_csv}'. Please run the mining script first.")
    
    df_scenarios = pd.read_csv(mined_scenarios_csv)
    scenario_ids_to_process = df_scenarios['scenario_id'].tolist()
    print(f"Found {len(scenario_ids_to_process)} scenarios to generate rationales for.")

    # --- 2. EFFICIENT VLM LOADING ---
    # Load the model ONCE outside the loop for maximum efficiency.
    print("\n--- Initializing VLM (this may take a moment)... ---")
    repo_id = "lmstudio-community/Qwen3-VL-8B-Instruct-GGUF"   #"unsloth/gemma-3-12b-it-GGUF", "google/gemma-3-12b-it-qat-q4_0-gguf"
    llm_filename = "Qwen3-VL-8B-Instruct-Q6_K.gguf"        #"gemma-3-12b-it-UD-Q8_K_XL.gguf", "gemma-3-12b-it-q4_0.gguf" 
    projector_filename = "mmproj-Qwen3-VL-8B-Instruct-F16.gguf"  #"mmproj-F16.gguf", "mmproj-model-f16-12B.gguf"
    
    chat_handler = Llava15ChatHandler.from_pretrained(repo_id=repo_id, filename=projector_filename)
    llm = Llama.from_pretrained(
        repo_id=repo_id, filename=llm_filename,
        chat_handler=chat_handler, n_gpu_layers=-1, n_ctx=16384, verbose=False
    )
    print("--- VLM loaded successfully. Starting generation loop. ---")
    
    # --- 3. MAIN GENERATION LOOP ---
    for scenario_id in tqdm(scenario_ids_to_process, desc="Generating Rationales"):
        try:
            # --- Check if rationale already exists to make script resumable ---
            output_dir = os.path.join(rationales_output_dir, scenario_id)
            output_path = os.path.join(output_dir, "rationale.txt")
            if os.path.exists(output_path):
                tqdm.write(f"Skipping {scenario_id}, rationale already exists.")
                continue

            # Define the critical timestep for analysis (from our mining stage)
            KEY_TIMESTEP = 45 # Using placeholder for now

            scenario_assets_dir = os.path.join(preprocessed_dir, scenario_id)
            
            # --- Load preprocessed assets for the current scenario ---
            context_path = os.path.join(scenario_assets_dir, "context.txt")
            with open(context_path, 'r') as f:
                ground_truth_context = f.read()

            gif_path = os.path.join(scenario_assets_dir, "scenario.gif")
            gif_image = Image.open(gif_path)
            total_frames = gif_image.n_frames
            key_frame_indices = [int(total_frames * p) for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]]
            key_frames = [gif_image.seek(i) or gif_image.convert("RGB").copy() for i in key_frame_indices]

            # --- Build the V4 prompt ---
            npz_path = os.path.join(config['data']['processed_npz_dir'], 'validation', f"{scenario_id}.npz")
            scenario_data = load_npz_scenario(npz_path)
            system_prompt, _, user_task_prompt = build_prompt_v4(scenario_data, KEY_TIMESTEP)

            visual_legend = """The following keyframes are based on this visual legend:

            MAP ELEMENTS:
            - Road Surface: The solid, drivable area, rendered in medium gray.
            - Non-Drivable Area: The black background.
            - Lane Markings: Thin white and yellow lines on the road surface.
            - Crosswalks: Semi-transparent white rectangular areas. *This element may not be present in all scenes.*
            - Traffic Lights: Colored circles (Red, Yellow, or Green) rendered on the road at the stop line. *This element may not be present in all scenes.*
            - Stop Signs: Solid red circles near an intersection. *This element may not be present in all scenes.*

            AGENTS:
            - The AV: The single **Magenta vehicle shape** (polygon with a tapered front).
            - The AV's Recent Path: A **dashed Magenta line** trailing the AV, indicating its trajectory over the last 2 seconds.
            - Other Vehicles: **Blue vehicle shapes** (polygons with a tapered front).
            - Pedestrians: Orange Circles.
            - Cyclists: **Cyan vehicle shapes** (smaller polygons with a tapered front)."""
            # full_user_prompt = f"{ground_truth_context}\n\n{visual_legend}\n\n{user_task_prompt}"

            # --- Prepare the VLM call (V2 - Interleaved with Frame Indices) ---
            user_content = []
            
            # First, add the text that comes before the images
            user_content.append({"type": "text", "text": f"{ground_truth_context}\n\n{visual_legend}\n\n{user_task_prompt}\n\n--- Scenario Keyframes ---"})
            
            # Now, interleave each keyframe with its index label
            for i, frame in enumerate(key_frames):
                frame_idx = key_frame_indices[i]
                
                # Add the text label for the frame
                user_content.append({"type": "text", "text": f"\n**Frame {i+1} (Original Timestep: {frame_idx})**"})
                
                # Add the image itself
                user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(frame)}})

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            # --- Run Inference ---
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=800,
                temperature=0.05
            )
            assistant_response = response['choices'][0]['message']['content']

            # --- Save the output ---
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(assistant_response)
        
        except Exception as e:
            tqdm.write(f"❌ Failed to process scenario {scenario_id}. Error: {e}")
            continue # Continue to the next scenario

    print("\n\n--- Batch Rationale Generation Complete ---")
    print(f"Results saved in: {os.path.abspath(rationales_output_dir)}")

if __name__ == "__main__":
    main()