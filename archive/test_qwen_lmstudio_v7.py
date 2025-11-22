# File: test_qwen_lmstudio_v7.py
# Purpose: The definitive test harness, implementing a "Continuous RAG" prompt
#          that provides the traffic light state for every keyframe.

import os
import sys
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.data_processing.waymo_parser import load_npz_scenario
from src.reasoning.prompt_builder import get_av_traffic_light_state_at_ts

# --- Helper function ---
def pil_image_to_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def main():
    print("--- Testing Qwen with Final V15 Prompt (Continuous RAG) ---")

    # --- 1. SETUP ---
    config = load_config()
    base_url = "http://192.168.1.67:1234/v1" 
    client = OpenAI(base_url=base_url, api_key="not-needed")
    
    # Allow passing scenario ID via command line
    scenario_id_to_test = sys.argv[1] if len(sys.argv) > 1 else "8807e9963f411c48"
    print(f"--- Testing scenario: {scenario_id_to_test} ---")

    # --- 2. Load All Assets ---
    try:
        npz_dir = config['data']['processed_npz_dir']
        npz_path = os.path.join(npz_dir, 'validation', f"{scenario_id_to_test}.npz")
        scenario_data = load_npz_scenario(npz_path)
        
        preprocessed_dir = "outputs/preprocessed_scenarios"
        legend_image = Image.open(os.path.join("outputs/legend_assets", "visual_legend.png")).convert("RGB")
        
        gif_path = os.path.join(preprocessed_dir, scenario_id_to_test, "scenario.gif")
        gif_image = Image.open(gif_path)
        key_frame_indices = [int(gif_image.n_frames * p) for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]]
        key_frames = [gif_image.seek(i) or gif_image.convert("RGB").copy() for i in key_frame_indices]
        
        print(f"✅ Loaded all assets for scenario {scenario_id_to_test}.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load assets. {e}")
        return

    # --- 3. Define the Final Prompt Components ---
    # We will use the final V12 system prompt and V13 user task prompt
    system_prompt = """You are a meticulous, expert Autonomous Vehicle Safety Analyst.

--- CRITICAL CONTEXT: HOW TO INTERPRET THE VISUALIZATION ---
The images are a custom, **ego-centric schematic** visualization from a simulator, not a photograph. You MUST interpret the visuals according to these rules:

1.  **Fixed Perspective:** The entire scene has been **rotated once** so that the Magenta AV's initial direction of travel is oriented **towards the top of the screen**. This perspective is then locked and remains fixed for the entire GIF. As the AV turns, its orientation on the screen will change relative to this fixed "up."
2.  **Symbolic Scaling for Salience:** The size of **Pedestrians (Blue Icons)** and **Cyclists (Cyan Triangles)** has been **intentionally exaggerated** for maximum visual salience. Do not interpret their large size literally; treat them as high-visibility symbolic markers.
3.  **Schematic Shapes:** All agents are represented by SUMO-style sprites, as defined in the visual legend.
4.  **Occlusions are Real:** Objects can and will disappear if they are hidden from the AV's line of sight.

Your Core Directives:
1.  **Trust the Visuals:** Base your analysis on the events as they are depicted, following the rules above.
2.  **Acknowledge Perception Limits:** When identifying risks, explicitly consider the AV's limited perception (e.g., "The risk is high because the truck is occluding a potential pedestrian").
3.  **Think in Causal Chains:** Connect actions to consequences to identify the root causes of risk.
"""

    # Component B: The User Task Prompt (Our proven CoT Hybrid)
    user_task_prompt = """First, perform a detailed, step-by-step analysis of the scene before providing your final answer. Follow this exact process:

**Internal Monologue (Chain-of-Thought):**
1.  **Scene Description:** Describe the static environment. What type of road is it? What map elements are visible?
2.  **Dynamic Analysis (Frame-by-Frame):** Describe the actions of the Magenta AV and any other relevant agents, noting their movement relative to the "Ego-Up" perspective. For any Pedestrians or Cyclists, you must state their approximate location relative to the AV and the nearest crosswalk.
3.  **Synthesize Key Events:** Summarize the most critical event or traffic law violation.

**Final Answer:**
Now, based on your analysis, provide your final answer in this format:

**Step 1: Factual Event Chronology.**
Summarize the sequence of events.

**Step 2: Causal Risk Identification.**
Identify the primary causal risk. In your explanation, you must **explicitly state whether the AV's path and the pedestrian's path actually intersect.** Analyze the risk based on their spatial relationship.

**Step 3: Optimal Action at the Critical Moment.**
The critical moment is in **Frame 1 and Frame 2**. What is the single, safest action the Magenta AV **should have taken**? Justify your recommendation based on the *actual* positions of the agents.
"""
    
    # --- 4. Construct the Final "Continuous RAG" Messages List ---
    user_content = []
    
    # Part 1: Visual Legend
    user_content.append({"type": "text", "text": "Use this visual legend to identify all objects:"})
    user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(legend_image)}})
    
    # Part 2: The Main Task
    user_content.append({"type": "text", "text": f"\n\n{user_task_prompt}\n\n--- Scenario Keyframes ---"})
    
    # --- THIS IS THE CRITICAL NEW LOGIC ---
    # Part 3: The keyframes, interleaved with labels that now include the traffic light state
    print("\n--- Injecting Ground-Truth Traffic Light States ---")
    for i, frame in enumerate(key_frames):
        frame_idx = key_frame_indices[i]
        
        # Call our helper to get the TL state for THIS specific frame
        av_tl_state = get_av_traffic_light_state_at_ts(scenario_data, frame_idx)
        
        # Create the new, richer label
        frame_label = f"\n**Frame {i+1} (Timestep: {frame_idx}) | AV Traffic Light: {av_tl_state}**"
        print(f"  - {frame_label.strip()}")
        
        user_content.append({"type": "text", "text": frame_label})
        user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(frame)}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # --- 5. Run Inference ---
    print("\n--- Sending final V15 prompt to LM Studio server... ---")
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=messages,
            max_tokens=2048,
            temperature=0.1
        )
        output_text = response.choices[0].message.content
        print(f"\n--- Rationale for Scenario: {scenario_id_to_test} ---")
        print(output_text)
    except Exception as e:
        print(f"\n❌ ERROR: Could not connect or generate. Details: {e}")

    print("\n--- END OF EXPERIMENT ---")

if __name__ == "__main__":
    main()