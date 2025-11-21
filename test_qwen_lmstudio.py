# File: test_qwen_lmstudio.py
import os
import sys
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI # <-- We use the openai library to talk to the server

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.reasoning.prompt_builder import build_prompt_v4
from src.data_processing.waymo_parser import load_npz_scenario
from src.utils.config_loader import load_config

# --- Helper function (from generate_rationales.py) ---
def pil_image_to_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def main():
    print("--- Testing Qwen-8B via LM Studio Server ---")

    # Point the client to the local server
    base_url = "http://192.168.1.67:1234/v1"
    client = OpenAI(base_url=base_url, api_key="not-needed")

    # --- Load Scenario Assets (same as before) ---
    config = load_config()
    scenario_id_to_test = "84a1c7968294e32d" # The RLV scenario
    preprocessed_dir = "outputs/preprocessed_scenarios"
    legend_assets_dir = "outputs/legend_assets"
    
    # --- NEW: Load the single legend image ---
    legend_image_path = os.path.join(legend_assets_dir, "visual_legend.png")
    if not os.path.exists(legend_image_path):
        raise FileNotFoundError(f"Visual legend image not found at '{legend_image_path}'. Please run 'create_legend_image.py' first.")
    legend_image = Image.open(legend_image_path).convert("RGB")
    print(f"✅ Successfully loaded visual legend image.")
    
    # Load scenario GIF and select keyframes (same as before)
    gif_path = os.path.join(preprocessed_dir, scenario_id_to_test, "scenario.gif")
    gif_image = Image.open(gif_path)
    key_frame_indices = [int(gif_image.n_frames * p) for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]]
    key_frames = [gif_image.seek(i) or gif_image.convert("RGB").copy() for i in key_frame_indices]
    print(f"✅ Successfully loaded {len(key_frames)} keyframes for scenario {scenario_id_to_test}.")
    
    # --- Build the Prompt (same as before) ---
    npz_path = os.path.join(config['data']['processed_npz_dir'], 'validation', f"{scenario_id_to_test}.npz")
    scenario_data = load_npz_scenario(npz_path)
    system_prompt, _, user_task_prompt = build_prompt_v4(scenario_data, 45)
    visual_legend = """The following keyframes are based on this visual legend:

    MAP ELEMENTS:
    - Road Surface: The solid, drivable area, rendered in medium gray.
    - Non-Drivable Area: The black background.
    - Lane Markings: Thin white and yellow lines on the road surface.
    - Crosswalks: White rectangular striped areas. *This element may not be present in all scenes.*
    - Traffic Lights: Colored circles (Red, Yellow, or Green) rendered on the road at the stop line. *This element may not be present in all scenes.*
    - Stop Signs: Solid **red pentagons**. *This element may not be present in all scenes.*

    AGENTS:
    - The AV: The single **Magenta vehicle shape** (polygon with a tapered front).
    - The AV's Recent Path: A **dashed Magenta line** trailing the AV, indicating its trajectory over the last 2 seconds.
    - Other Vehicles: **Blue vehicle shapes** (polygons with a tapered front).
    - Pedestrians: Orange Circles.
    - Cyclists: **Cyan vehicle shapes** (smaller polygons with a tapered front)."""

    
    # --- Construct the OpenAI-compatible messages list ---
    user_content = []
    # Add text parts
    full_user_prompt = f"{visual_legend}\n\n{user_task_prompt}"
    user_content.append({"type": "text", "text": full_user_prompt})
    
    # Part 1: The Legend Image and its introduction
    user_content.append({"type": "text", "text": "You can also use this visual legend to understand the objects in the upcoming scenario:"})
    user_content.append({
        "type": "image_url",
        "image_url": {"url": pil_image_to_data_uri(legend_image)}
    })
    
    # Part 2: The main user task prompt
    user_content.append({"type": "text", "text": f"\n\nNow, analyze the following scenario keyframes.\n{user_task_prompt}\n\n--- Scenario Keyframes ---"})
    
    # Part 3: The scenario keyframes, interleaved with their labels
    for i, frame in enumerate(key_frames):
        frame_idx = key_frame_indices[i]
        user_content.append({"type": "text", "text": f"\n**Frame {i+1} (Original Timestep: {frame_idx})**"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": pil_image_to_data_uri(frame)}
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    print(f'Prompt constructed with {len(messages)} messages, ready for LM Studio inference.')
    
    # --- Run Inference via API Call ---
    print("\n--- Sending request to LM Studio server... ---")
    try:
        response = client.chat.completions.create(
            model="local-model", # The model name doesn't matter for LM Studio
            messages=messages,
            max_tokens=2048,
            temperature=0.1
        )
        output_text = response.choices[0].message.content
        print("\n--- Qwen3-VL-8B LM Studio Output ---")
        print(output_text)
    except Exception as e:
        print(f"\n❌ ERROR: Could not connect to the LM Studio server.")
        print(f"   Please ensure the server is running on {base_url}")
        print(f"   Error details: {e}")

    print("\n--- END OF EXPERIMENT ---")

if __name__ == "__main__":
    main()