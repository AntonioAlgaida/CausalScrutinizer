# File: v8_reflexion.py
# Purpose: The definitive, state-of-the-art script implementing a two-turn,
#          "Self-Refinement" or "Reflexion" agent for maximum reasoning robustness.

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

# --- Prompt Definitions (Final Versions) ---

SYSTEM_PROMPT = """You are a meticulous, expert Autonomous Vehicle Safety Analyst.

--- CRITICAL CONTEXT: HOW TO INTERPRET THE VISUALIZATION ---
The images are a custom, **ego-centric schematic** visualization from a simulator, not a photograph. You MUST interpret the visuals according to these rules:

1.  **Fixed Perspective:** The entire scene has been **rotated once** so that the Magenta AV's initial direction of travel is oriented **towards the top of the screen**. This perspective is then locked and remains fixed for the entire GIF.
2.  **Symbolic Scaling for Salience:** The size of **Pedestrians (Orange Icons)** and **Cyclists (Cyan Triangles)** has been **intentionally exaggerated** for maximum visual salience. Do not interpret their large size literally; treat them as high-visibility symbolic markers.
3.  **Schematic Shapes:** All agents are represented by SUMO-style sprites, as defined in the visual legend you will be provided.
4.  **Occlusions are Real:** Objects can and will disappear if they are hidden from the AV's line of sight.
5.  **Qualitative Sense of Scale:** To help you judge distance, use the lane markings as a reference. A typical dashed white lane marking is approximately **3 meters (or 10 feet) long**. You can use this as a "mental ruler" to estimate the approximate distances between agents.

Your Core Directives:
1.  **Trust the Visuals and Provided Data:** Base your analysis ONLY on the events as they are depicted and the ground-truth data provided in the prompt.
2.  **Acknowledge Perception Limits:** When identifying risks, explicitly consider the AV's limited perception (e.g., "The risk is high because the truck is occluding a potential pedestrian").
3.  **Think in Causal Chains:** Connect actions to consequences to identify the root causes of risk.
"""

# The user task for the first "Generator" turn
USER_TASK_GENERATION = """Analyze the provided visual legend and scenario keyframes. For each keyframe, a ground-truth label specifies the AV's relevant traffic light state.

First, perform a detailed, step-by-step analysis of the scene. Follow this exact process:

**Internal Monologue (Chain-of-Thought):**
1.  **Scene Description:** Describe the static environment.
2.  **Dynamic Analysis (Frame-by-Frame):** Describe the actions of all agents.
3.  **Conflict Scan (NEW STEP):**
    *   **AV vs. Blue Vehicle 1:** Is there a risk of collision? Yes/No. Why?
    *   **AV vs. Blue Vehicle 2:** Is there a risk of collision? Yes/No. Why?
    *   ... (repeat for all agents)
    *   **AV vs. Orange Pedestrian 1:** Is there a risk of collision? Yes/No. Why?
4.  **Synthesize Key Events:** Based on the conflict scan, what is the single most immediate and high-risk event?

**Final Answer:**
Now, based on your analysis, provide your final answer in this format:

**Step 1: Factual Event Chronology.**
Summarize the sequence of events using the correct agent colors.

**Step 2: Causal Risk Identification.**
Identify the primary causal risk for the AV.

**Step 3: Optimal Action at the Critical Moment.**
What is the single, safest action the Magenta AV should have taken? Justify it.
"""

# The new prompt for the second "Refinement" turn
USER_TASK_REFINEMENT = """You have provided the following initial analysis:

--- PREVIOUS ANALYSIS ---
{previous_analysis}
-------------------------

Now, perform a self-critique. Your task is to verify two things:
1.  **Traffic Light Analysis:** Reread the ground-truth `AV Traffic Light` label for **each keyframe**. Was your original analysis of the traffic light state and any violations correct?
2.  **Agent Identification:** Review the visuals again. Did you correctly identify all agents according to the legend (e.g., Orange Icons are Pedestrians, Blue Rectangles are Vehicles)?

Provide a final, refined answer. If your original analysis was correct, state that and provide the final answer again. If it was incorrect, correct the specific error in your chronology and causal risk sections, and then provide the updated, corrected final answer.
"""

def main():
    print("==========================================================")
    print("   V8 REFLEXION AGENT: GENERATOR-CRITIC CAUSAL AUDIT    ")
    print("==========================================================")

    # --- 1. SETUP ---
    config = load_config()
    base_url = "http://192.168.1.67:1234/v1" 
    client = OpenAI(base_url=base_url, api_key="not-needed")
    
    scenario_id_to_test = sys.argv[1] if len(sys.argv) > 1 else "8807e9963f411c48"
    print(f"--- Testing scenario: {scenario_id_to_test} ---")

    # --- 2. Load All Assets ---
    try:
        npz_dir = config['data']['processed_npz_dir']
        npz_path = os.path.join(npz_dir, 'validation', f"{scenario_id_to_test}.npz")
        scenario_data = load_npz_scenario(npz_path)
        
        legend_image = Image.open(os.path.join("outputs/legend_assets", "visual_legend.png")).convert("RGB")
        gif_path = os.path.join("outputs/preprocessed_scenarios", scenario_id_to_test, "scenario.gif")
        gif_image = Image.open(gif_path)
        
        print(f"Scenario GIF Path: {os.path.abspath(gif_path)}")
        
        key_frame_indices = [int(gif_image.n_frames * p) for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]]
        key_frames = [gif_image.seek(i) or gif_image.convert("RGB").copy() for i in key_frame_indices]
        print(f"✅ Loaded all assets for scenario {scenario_id_to_test}.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load assets. {e}")
        return

    # --- 3. TURN 1: GENERATION ---
    print("\n--- TURN 1: Generating Initial Analysis ---")

    # Build the "Continuous RAG" content for the first turn
    generation_content = []
    generation_content.append({"type": "text", "text": "Use this visual legend to identify all objects:"})
    generation_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(legend_image)}})
    generation_content.append({"type": "text", "text": f"\n\n{USER_TASK_GENERATION}\n\n--- Scenario Keyframes ---"})
    
    for i, frame in enumerate(key_frames):
        frame_idx = key_frame_indices[i]
        av_tl_state = get_av_traffic_light_state_at_ts(scenario_data, frame_idx)
        frame_label = f"\n**Frame {i+1} (Timestep: {frame_idx}) | AV Traffic Light: {av_tl_state}**"
        generation_content.append({"type": "text", "text": frame_label})
        generation_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(frame)}})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": generation_content}]

    try:
        response_turn_1 = client.chat.completions.create(model="local-model", messages=messages, max_tokens=1024, temperature=0.1)
        initial_analysis = response_turn_1.choices[0].message.content
        print("\n--- Initial Analysis (Turn 1 Output) ---")
        print(initial_analysis)
    except Exception as e:
        print(f"\n❌ ERROR during Turn 1: {e}")
        return

    # --- 4. TURN 2: REFINEMENT (SELF-CRITIQUE) ---
    print("\n\n--- TURN 2: Performing Self-Refinement ---")

    # The conversation history now includes the model's first answer
    messages.append({"role": "assistant", "content": initial_analysis})
    
    # Add the new user task for refinement
    refinement_task = USER_TASK_REFINEMENT.format(previous_analysis=initial_analysis)
    messages.append({"role": "user", "content": refinement_task})

    try:
        response_turn_2 = client.chat.completions.create(model="local-model", messages=messages, max_tokens=1024, temperature=0.1)
        refined_analysis = response_turn_2.choices[0].message.content
        
        print("\n" + "="*20 + " FINAL REFINED ANALYSIS (Turn 2 Output) " + "="*20)
        print(refined_analysis)
        print("="*80)
    except Exception as e:
        print(f"\n❌ ERROR during Turn 2: {e}")
        return
        
    print("\n--- END OF REFLEXION AGENT ---")

if __name__ == "__main__":
    main()