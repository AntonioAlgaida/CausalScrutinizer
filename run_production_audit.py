# File: run_production_audit.py
# Purpose: The final, production-grade script to generate the "Causal Rationale Dataset".
# It iterates through the mined scenarios and runs the definitive Kimi-VL audit on each.

import os
import sys
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import json
import datetime
import pandas as pd
from tqdm import tqdm
import time
import traceback

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Import all our final, definitive modules ---
from src.utils.config_loader import load_config
from src.data_processing.waymo_parser import load_npz_scenario
from src.reasoning.prompt_builder_v5 import get_av_traffic_light_state_at_ts
from src.utils.geometry import check_for_path_conflict, get_top_k_threats

# --- Helper Functions ---

def pil_image_to_data_uri(image: Image.Image) -> str:
    """Encodes a PIL image into a base64 data URI."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def extract_thinking_and_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> tuple[str, str]:
    """Parses the Kimi-VL model's output to separate the CoT from the final answer."""
    if bot in text and eot in text:
        start_idx = text.find(bot) + len(bot)
        end_idx = text.find(eot)
        if start_idx < end_idx:
            thinking = text[start_idx:end_idx].strip()
            summary = text[end_idx + len(eot):].strip()
            return thinking, summary
    return "Could not parse 'thinking' block.", text

# --- Final Prompt Definitions ---

SYSTEM_PROMPT = """You are "Scrutinizer-AI," a world-class, expert-level Autonomous Vehicle Safety Analyst. Your entire purpose is to perform a deep, objective, and evidence-based causal audit of complex driving scenarios. You are renowned for your meticulous attention to detail and your ability to think in clear, logical, causal chains.

--- YOUR CORE OPERATING PRINCIPLES (NON-NEGOTIABLE) ---

1.  **Ground-Truth is Law:** You will be provided with **Ground-Truth Data** for each frame (Traffic Light state, Path Conflict status, and a list of Dynamic Agents). This data is **100% accurate** and has been computed by a deterministic geometry engine. It is your absolute source of truth. Your entire analysis MUST be built upon and consistent with this data. **Never contradict the ground truth.**

2.  **Synthesize, Do Not Calculate:** Your role is to be a **Causal Reasoner**, not a geometry engine. Do NOT perform your own geometric calculations, estimate speeds from pixels, or guess at distances. Your task is to synthesize the provided visuals with the ground-truth data to explain the *causal implications* of the scene.

3.  **Interpret Visuals According to Strict Rules:** The images are a custom, **ego-centric schematic** visualization from a simulator. They are a symbolic representation of reality, not a photograph. You MUST interpret them *only* according to these rules:
    *   **Fixed "Up" Perspective:** The Magenta AV's initial direction of travel is always towards the top of the screen.
    *   **Symbolic, Not Literal:** Pedestrians (Orange Icons) and Cyclists (Cyan Triangles) are intentionally enlarged for high visibility. Their size is symbolic of their importance, not their physical scale.
    *   **Trust the Legend:** All objects are defined by the provided visual legend. If you see something that looks like a "bus," but it is colored blue, you MUST refer to it as a "Blue Vehicle" as per the legend. Do not add details not present in the schematics.

4.  **Prioritize Direct Conflicts:** A ground-truth `AV Path Conflict: YES` is the most severe and immediate type of risk. This is a definitive signal of a safety-critical event and **must** be the primary focus of your causal analysis when present. The `Top 3 Dynamic Agents` list provides crucial context about other actors in the scene but does not imply a collision course unless also flagged by the `Path Conflict` status.
"""

USER_TASK_KIMI = """Your task is to produce a **Comprehensive Causal Audit** of the provided scenario by synthesizing the visual evidence and the associated **Ground-Truth Data**.

--- DEEP REASONING DIRECTIVE ---
Before you begin writing the final, structured output, you **MUST** engage in a deep, internal, step-by-step reasoning process. Use your native `◁think▷` capability to build a detailed mental model of the entire scenario from start to finish. In your thinking process, you should:
1.  Methodically analyze each frame, correlating the visual evidence with every piece of ground-truth data provided.
2.  Formulate a clear hypothesis about the primary causal event and any contributing factors.
3.  Consider the full chain of cause and effect.

Only after you have completed this internal monologue should you generate the final, clean output in the format specified below.

--- FINAL OUTPUT FORMAT ---

**Step 1: Detailed Scene Elaboration.**
In a descriptive paragraph, "paint a picture" of the overall scene and the AV's situation. Describe the environment, the AV's position and intended path, and the general configuration of other agents.

**Step 2: Factual Event Chronology.**
Summarize the key sequence of events. You **MUST** incorporate the provided ground-truth data for each frame, including the `AV Traffic Light` state, the `AV Path Conflict` status, and the list of `Top 3 Dynamic Agents`.

**Step 3: Comprehensive Causal Risk Identification.**
-   **Primary Causal Risk:** Based on all the evidence, identify the single most immediate and severe risk. The **`AV Path Conflict: YES`** flag is the most direct indicator of a critical risk. Explain the direct causal chain (Cause -> Effect -> Consequence).
-   **Secondary / Contributing Risks:** Describe any other risk factors (e.g., occlusions, environmental conditions, the behavior of other non-conflicting agents) that contribute to the complexity of the situation.

**Step 4: Optimal Action and Counterfactual.**
-   **Optimal Action:** What is the single, safest action the Magenta AV should have taken at the **critical moment** (the first timestep where a conflict or high-risk situation is identified)?
-   **Justification:** Justify your recommendation by directly referencing the ground-truth data.
-   **Counterfactual Analysis:** Briefly describe what would have likely happened if the AV had *not* taken the optimal action.
"""
# --- Main Worker Function ---

def process_single_scenario(scenario_id: str, client: OpenAI, config: dict, legend_image: Image.Image) -> bool:
    """
    The main workhorse function. Processes a single scenario from ID to saved JSON.
    """
    # --- 1. Setup Paths and Check for Existing Output ---
    output_dir = "outputs/causal_rationale_dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scenario_id}.json")

    if os.path.exists(output_path):
        return True # Success, as it's already processed

    # --- 2. Load Scenario-Specific Assets (Data & GIF) ---
    try:
        npz_dir = config['data']['processed_npz_dir']
        npz_path = os.path.join(npz_dir, 'validation', f"{scenario_id}.npz")
        scenario_data = load_npz_scenario(npz_path)
        
        gif_path = os.path.join("outputs/preprocessed_scenarios", scenario_id, "scenario.gif")
        gif_image = Image.open(gif_path)
        
        key_frame_indices = [int(gif_image.n_frames * p) for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]]
        key_frames = [gif_image.seek(i) or gif_image.convert("RGB").copy() for i in key_frame_indices]
    except FileNotFoundError as e:
        print(f"\n- WARNING: Asset not found for {scenario_id}. Skipping. Error: {e}")
        return False
    except Exception as e:
        print(f"\n- ERROR: Failed to load assets for {scenario_id}. Error: {e}")
        return False

    # --- 3. Construct the Prompt Payload ---
    user_content = []
    user_content.append({"type": "text", "text": "Use this visual legend to identify all objects:"})
    user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(legend_image)}})
    user_content.append({"type": "text", "text": f"\n\n{USER_TASK_KIMI}"})
    user_content.append({"type": "text", "text": "\n\n--- Scenario Keyframes & Ground-Truth Data ---"})
    
    for i, frame in enumerate(key_frames):
        frame_idx = key_frame_indices[i]
        
        av_tl_state = get_av_traffic_light_state_at_ts(scenario_data, frame_idx)
        has_conflict, conflict_type, conflict_id = check_for_path_conflict(scenario_data, frame_idx)
        top_agents = get_top_k_threats(scenario_data, frame_idx, k=3)

        conflict_str = f"YES ({conflict_type}, ID: {conflict_id})" if has_conflict else "NO"
        agents_str = "\n".join([f"    - {t}" for t in top_agents]) if top_agents else "    - None"
        
        frame_label = (
            f"\n**Frame {i+1} (Timestep: {frame_idx})**\n"
            f"**--- Ground Truth ---**\n"
            f"  - **AV Traffic Light:** {av_tl_state}\n"
            f"  - **AV Path Conflict:** {conflict_str}\n"
            f"  - **Top 3 Dynamic Agents (sorted by interaction score):**\n{agents_str}"
        )
        user_content.append({"type": "text", "text": frame_label})
        user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(frame)}})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]

    # --- 4. Run Inference ---
    response = client.chat.completions.create(
        model="local-model", messages=messages, max_tokens=4096, temperature=0.1
    )
    output_text = response.choices[0].message.content
    
    # --- 5. Parse and Save the Output ---
    thinking_part, final_answer_part = extract_thinking_and_summary(output_text)
    
    # We will save everything in a structured JSON
    output_data = {
        "scenario_id": scenario_id,
        "generation_timestamp": datetime.datetime.now().isoformat(),
        "model_used": "Kimi-VL-A3B-Thinking-2506-Q8_0",
        "prompt_version": "V22 - Deep Reflexion",
        "internal_monologue": thinking_part,
        "final_causal_audit": final_answer_part,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    return True

# --- Main Orchestrator ---

def main():
    print("==========================================================")
    print("   FINAL PRODUCTION RUN: Generating Causal Rationale Dataset   ")
    print("==========================================================")
    
    try:
        # --- 1. Load Config, Client, and Shared Assets ---
        config = load_config()
        client = OpenAI(base_url="http://192.168.1.67:1234/v1", api_key="not-needed")
        
        # Load the legend ONCE to pass to all workers
        legend_image = Image.open(os.path.join("outputs/legend_assets", "visual_legend.png")).convert("RGB")
        
        # --- 2. Load the Work Queue ---
        mined_scenarios_csv = "data/mined_scenarios/critical_scenario_ids_v1.csv"
        df_scenarios = pd.read_csv(mined_scenarios_csv)
        scenario_ids = df_scenarios['scenario_id'].tolist()
        
        print(f"Loaded {len(scenario_ids)} critical scenarios to audit.")
        
        # --- 3. Process the Batch with a Progress Bar ---
        success_count = 0
        skipped_count = 0
        error_list = []
        
        # Note: We are running this sequentially for stability with the API.
        # To parallelize, we would need to wrap this in a multiprocessing pool.
        for scenario_id in tqdm(scenario_ids, desc="Auditing Scenarios"):
            output_path = os.path.join("outputs/causal_rationale_dataset", f"{scenario_id}.json")
            if os.path.exists(output_path):
                skipped_count += 1
                continue

            try:
                if process_single_scenario(scenario_id, client, config, legend_image):
                    success_count += 1
                else:
                    error_list.append(scenario_id)
            except Exception as e:
                print(f"\n❌ A fatal error occurred on scenario {scenario_id}: {e}")
                traceback.print_exc()
                error_list.append(scenario_id)
            
            # Add a small delay to be kind to the LM Studio server
            time.sleep(1)

        # --- 4. Final Summary ---
        print("\n--- Batch Processing Complete ---")
        print(f"✅ Successfully generated: {success_count} new rationales.")
        print(f"⏭️  Skipped (already exist): {skipped_count}")
        print(f"❌ Failed scenarios: {len(error_list)}")
        if error_list:
            print("   Failed IDs:", error_list)
        print(f"   Final dataset saved in: {os.path.abspath('outputs/causal_rationale_dataset')}")

    except FileNotFoundError as e:
        print(f"\n❌ FATAL ERROR: A required file was not found. {e}")
        print("   Please ensure you have run the mining and preprocessing scripts first.")
    except Exception as e:
        print(f"\n❌ An unexpected fatal error occurred in the main orchestrator: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()