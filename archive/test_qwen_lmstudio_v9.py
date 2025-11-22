# File: test_qwen_lmstudio_v9.py
# Purpose: The definitive test harness for the Causal Scrutinizer.
# This script implements the "Ultimate RAG" pipeline, injecting ground-truth
# from our deterministic geometry engine directly into the VLM prompt.

import os
import sys
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import json # NEW import for structured logging
import datetime # NEW import for unique filenames

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.data_processing.waymo_parser import load_npz_scenario
from src.reasoning.prompt_builder import get_av_traffic_light_state_at_ts
from src.utils.geometry import check_for_path_conflict, get_top_k_threats

# --- Helper function ---
def pil_image_to_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"


# --- NEW HELPER FUNCTION: THE DEBUG DOSSIER ---
def save_debug_dossier(scenario_id: str, messages: list, model_output: str = None):
    """
    Saves the complete input payload and (optionally) the model's output
    to a timestamped text file for easy debugging.

    NOTE: Any image data (image_url) will be removed/redacted before saving.
    """
    print("\n--- Saving Debug Dossier ---")
    
    # 1. Create a unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{scenario_id}_dossier.txt"
    
    # 2. Define the output directory
    output_dir = "outputs/debug_dossiers"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Helper: recursively sanitize the messages structure, removing image data
    def _sanitize(obj):
        if isinstance(obj, dict):
            sanitized = {}
            for k, v in obj.items():
                if k == "image_url":
                    # Remove or replace large image data; keep placeholder so structure is clear
                    sanitized[k] = {"url": "<IMAGE_REMOVED>"}
                else:
                    sanitized[k] = _sanitize(v)
            return sanitized
        elif isinstance(obj, list):
            return [_sanitize(i) for i in obj]
        else:
            return obj

    sanitized_messages = _sanitize(messages)

    # 3. Write the content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"SCENARIO ID: {scenario_id}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write("="*80 + "\n\n")
        
        f.write("--- INPUT PAYLOAD (MESSAGES) ---\n\n")
        f.write(json.dumps(sanitized_messages, indent=2))
        
        if model_output:
            f.write("\n\n" + "="*80 + "\n\n")
            f.write("--- MODEL OUTPUT ---\n\n")
            f.write(model_output)
            
        f.write("\n\n" + "="*80 + "\n")

    print(f"✅ Debug dossier saved to: {output_path}")
    
# --- Prompt Definitions ---
SYSTEM_PROMPT = """You are a meticulous, expert Autonomous Vehicle Safety Analyst.

--- CRITICAL CONTEXT: HOW TO INTERPRET THE VISUALIZATION & DATA ---
The images are a custom, **ego-centric schematic** visualization. You MUST interpret everything according to these rules:

1.  **Fixed Perspective:** The Magenta AV's initial travel is always "UP."
2.  **Symbolic Scaling:** Pedestrians (Orange) and Cyclists (Cyan) are intentionally enlarged for visibility.
3.  **Ground-Truth is Law:** For each frame, you are provided with **Ground-Truth Data** computed by a deterministic geometry engine. This data (Traffic Light state, Path Conflict, and Top Threats) is **100% accurate**. Your primary task is to use this data to build a causal explanation for the events you see. Do not contradict the ground truth.

Your Core Directives:
1.  **Synthesize, Do Not Calculate:** Use the provided data; do not perform your own distance or conflict calculations.
2.  **Acknowledge Perception Limits:** Consider what might be occluded from the AV's view.
3.  **Think in Causal Chains:** Connect the ground-truth facts to the visual events to explain the *why*.
"""

USER_TASK_GENERATION = """Analyze the provided visual legend and the sequence of keyframes. For each keyframe, you have been given **Ground-Truth Data**.

First, perform a detailed, step-by-step analysis. Follow this exact process:

**Internal Monologue (Chain-of-Thought):**
1.  **Scene Description:** Describe **in detail** (rich, detailed) the static environment (road type, elements).
2.  **Dynamic Analysis (Frame-by-Frame):** For each frame, describe the actions of the AV and other key agents. You **MUST** reference the provided `AV Traffic Light`, `AV Path Conflict`, and `Top 3 Dynamic Threats` ground-truth data in your description for that frame to explain the situation.
3.  **Synthesize Key Events:** Based on the ground-truth data across all frames, what is the single most critical event or conflict that defines this scenario?

**Final Answer:**
Now, based on your analysis, provide your final answer in this format:

**Step 1: Detailed Scene Elaboration.**
In a descriptive paragraph, "paint a picture" of the overall scene and the AV's situation. Describe the environment, the AV's position and intended path, and the general configuration of other agents. Make the scene feel dynamic and alive.

**Step 2: Factual Event Chronology.**
Summarize the key sequence of events, incorporating the ground-truth conflict information.

**Step 3: Causal Risk Identification.**
Identify the primary causal risk, using the ground-truth `AV Path Conflict` and `Top 3 Dynamic Threats` as the central evidence.

**Step 4: Optimal Action at the Critical Moment.**
What is the single, safest action the Magenta AV should have taken at the moment the conflict was first detected? Justify it.
"""

def main():
    print("==========================================================")
    print("   V9: ULTIMATE RAG TEST HARNESS (with Debug Logging)     ")
    print("==========================================================")

    # --- 1. SETUP ---
    config = load_config()
    base_url = "http://192.168.1.67:1234/v1" 
    client = OpenAI(base_url=base_url, api_key="not-needed")
    
    scenario_id_to_test = sys.argv[1] if len(sys.argv) > 1 else "8807e9963f411c48"
    print(f"--- Testing scenario: {scenario_id_to_test} ---")

    # --- 2. Load All Assets (Raw Data, Legend, GIF) ---
    try:
        npz_dir = config['data']['processed_npz_dir']
        npz_path = os.path.join(npz_dir, 'validation', f"{scenario_id_to_test}.npz")
        scenario_data = load_npz_scenario(npz_path)
        
        legend_image = Image.open(os.path.join("outputs/legend_assets", "visual_legend.png")).convert("RGB")
        gif_path = os.path.join("outputs/preprocessed_scenarios", scenario_id_to_test, "scenario.gif")
        gif_image = Image.open(gif_path)
        
        key_frame_indices = [int(gif_image.n_frames * p) for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]]
        key_frames = [gif_image.seek(i) or gif_image.convert("RGB").copy() for i in key_frame_indices]
        print(f"✅ Loaded all assets for scenario {scenario_id_to_test}.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load assets. {e}")
        return

    # --- 3. Construct the "Ultimate RAG" Messages List ---
    user_content = []
    
    # Part 1: Initial instructions and legend
    user_content.append({"type": "text", "text": "Use this visual legend to identify all objects (only these objects can appear):"})
    user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(legend_image)}})
    user_content.append({"type": "text", "text": f"\n\n{USER_TASK_GENERATION}\n\n--- Scenario Keyframes & Ground-Truth Data ---"})
    
    # Part 2: Interleave keyframes with a full ground-truth label for each
    print("\n--- Injecting Full Ground-Truth Context for Each Frame ---")
    for i, frame in enumerate(key_frames):
        frame_idx = key_frame_indices[i]
        
        # Call ALL our deterministic geometry engine functions
        av_tl_state = get_av_traffic_light_state_at_ts(scenario_data, frame_idx)
        has_conflict, conflict_type, conflict_id = check_for_path_conflict(scenario_data, frame_idx)
        top_threats = get_top_k_threats(scenario_data, frame_idx, k=3)

        # Format the results into clean strings
        conflict_str = f"YES ({conflict_type}, ID: {conflict_id})" if has_conflict else "NO"
        threats_str = "\n".join([f"    - {t}" for t in top_threats]) if top_threats else "    - None"
        
        # Create the final, rich, ground-truth label for this frame
        frame_label = (
            f"\n**Frame {i+1} (Timestep: {frame_idx})**\n"
            f"**--- Ground Truth ---**\n"
            f"  - **AV Traffic Light:** {av_tl_state}\n"
            f"  - **AV Path Conflict:** {conflict_str}\n"
            f"  - **Top 3 Dynamic Threats (sorted by risk):**\n{threats_str}"
        )
        print(frame_label) # Print for debugging
        
        user_content.append({"type": "text", "text": frame_label})
        user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(frame)}})

    # This part is just to complete the user_content list for the dossier
    initial_prompt_text = "Use this visual legend to identify all objects:"
    task_prompt_text = f"\n\n{USER_TASK_GENERATION}\n\n--- Scenario Keyframes & Ground-Truth Data ---"
    
    # We rebuild the user_content to be exactly what's sent to the model
    final_user_content = [
        {"type": "text", "text": initial_prompt_text},
        {"type": "image_url", "image_url": {"url": pil_image_to_data_uri(legend_image)}},
        {"type": "text", "text": task_prompt_text},
    ] + user_content # This user_content already contains the interleaved text and images

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": final_user_content}]

    # --- 4. Run Inference ---
    print("\n--- Sending final V9 prompt to LM Studio server... ---")
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=messages,
            max_tokens=2048,
            temperature=0.1
        )
        output_text = response.choices[0].message.content
        
        print("\n" + "="*20 + " FINAL CAUSAL AUDIT " + "="*20)
        print(output_text)
        print("="*62)
        
        save_debug_dossier(scenario_id_to_test, messages, output_text)

    except Exception as e:
        print(f"\n❌ ERROR: Could not connect or generate. Details: {e}")
        save_debug_dossier(scenario_id_to_test, messages, model_output=f"SCRIPT FAILED:\n{e}")


    print("\n--- END OF EXPERIMENT ---")

if __name__ == "__main__":
    main()