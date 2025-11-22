# File: test_qwen_lmstudio_v4.py
# Purpose: The final test harness for our V6 renderer. This script uses our
#          definitive "V11" prompt to test any given scenario.

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

def pil_image_to_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def main():
    # --- 1. SETUP: Select Scenario and Connect to Server ---
    
    # --- FLEXIBLE SCENARIO SELECTION ---
    # You can provide a scenario ID as a command-line argument,
    # or it will default to the one below.
    if len(sys.argv) > 1:
        scenario_id_to_test = sys.argv[1]
        print(f"--- Testing custom scenario: {scenario_id_to_test} ---")
    else:
        scenario_id_to_test = "8807e9963f411c48" # Default to our RLV scenario
        print(f"--- Testing default scenario: {scenario_id_to_test} ---")

    config = load_config()
    base_url = "http://192.168.1.67:1234/v1" 
    client = OpenAI(base_url=base_url, api_key="not-needed")
    
    # --- NEW: Load the GBNF Grammar file ---
    grammar_path = os.path.join(PROJECT_ROOT, 'prompts/grammars/causal_scrutinizer_v1.gbnf')
    try:
        with open(grammar_path, 'r') as f:
            gbnf_grammar = f.read()
        print(f"✅ Successfully loaded GBNF grammar from: {grammar_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: Grammar file not found at '{grammar_path}'.")
        gbnf_grammar = None # Proceed without grammar if not found

    # --- 2. Load Visual Assets (Legend + Scenario) ---
    try:
        preprocessed_dir = "outputs/preprocessed_scenarios"
        legend_assets_dir = "outputs/legend_assets"
        
        legend_image_path = os.path.join(legend_assets_dir, "visual_legend.png")
        legend_image = Image.open(legend_image_path).convert("RGB")
        
        gif_path = os.path.join(preprocessed_dir, scenario_id_to_test, "scenario.gif")
        
        print(f"Scenario GIF Path: {os.path.abspath(gif_path)}")

        gif_image = Image.open(gif_path)
        key_frame_indices = [int(gif_image.n_frames * p) for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]]
        key_frames = [gif_image.seek(i) or gif_image.convert("RGB").copy() for i in key_frame_indices]
        
        print(f"✅ Loaded visual assets for scenario {scenario_id_to_test}.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find asset file. {e}")
        print("   Ensure you have run 'create_legend_image.py' and 'preprocess_scenarios.py' for this scenario.")
        return

    # --- 3. Define the Final "V11 - Ego-Centric" Prompt ---
    
    # Component A: The New System Prompt (Explains the new rules)

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
    # --- 4. Construct the Final Messages List ---
    user_content = []
    user_content.append({"type": "text", "text": "Use this visual legend to identify all objects:"})
    user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(legend_image)}})
    user_content.append({"type": "text", "text": f"\n\n{user_task_prompt}\n\n--- Scenario Keyframes ---"})
    
    for i, frame in enumerate(key_frames):
        frame_idx = key_frame_indices[i]
        user_content.append({"type": "text", "text": f"\n**Frame {i+1} (Original Timestep: {frame_idx})**"})
        user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(frame)}})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # --- 5. Run Inference with Grammar ---
    print("\n--- Sending final prompt with grammar to LM Studio server... ---")
    try:
        # --- THE CRITICAL CHANGE IS HERE ---
        # We add the "grammar" to the 'extra_body' of the request.
        # This is how we pass non-standard parameters to the underlying llama.cpp engine.
        response = client.chat.completions.create(
            model="local-model",
            messages=messages,
            max_tokens=2048,
            temperature=0.0, # Set to 0.0 for maximum determinism with grammar
            extra_body={
                "grammar": gbnf_grammar
            }
        )
        output_text = response.choices[0].message.content
        print(f"\n--- Rationale for Scenario: {scenario_id_to_test} (Constrained by Grammar) ---")
        print(output_text)
    except Exception as e:
        print(f"\n❌ ERROR: Could not connect or generate. Details: {e}")

    print("\n--- END OF EXPERIMENT ---")

if __name__ == "__main__":
    main()