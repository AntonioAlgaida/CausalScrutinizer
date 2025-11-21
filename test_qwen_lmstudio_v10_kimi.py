# File: run_single_scenario_audit.py
# Purpose: The definitive, single-scenario auditing script.
# This script implements the "Kimi-Native" Ultimate RAG pipeline for any given scenario.

import os
import sys
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import json
import datetime

# --- Add project root to path ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config
from src.data_processing.waymo_parser import load_npz_scenario
from src.reasoning.prompt_builder_v5 import get_av_traffic_light_state_at_ts
from src.utils.geometry import check_for_path_conflict, get_top_k_threats

# --- Helper functions ---
def pil_image_to_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def save_debug_dossier(scenario_id: str, messages: list, model_output: str = None):
    """Saves the complete input/output to a timestamped text file for debugging."""
    print("\n--- Saving Debug Dossier ---")
    output_dir = "outputs/debug_dossiers"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{scenario_id}_dossier.txt"
    output_path = os.path.join(output_dir, filename)

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: "<IMAGE_REDACTED>" if k == "image_url" else _sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_sanitize(i) for i in obj]
        return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"SCENARIO ID: {scenario_id}\nTIMESTAMP: {timestamp}\n\n")
        f.write("--- INPUT PAYLOAD (MESSAGES - IMAGES REDACTED) ---\n")
        f.write(json.dumps(_sanitize(messages), indent=2))
        if model_output:
            f.write("\n\n--- MODEL OUTPUT ---\n")
            f.write(model_output)
    print(f"✅ Debug dossier saved to: {output_path}")

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

def main():
    print("==========================================================")
    print("   FINAL AUDIT SCRIPT: Kimi-Native Ultimate RAG         ")
    print("==========================================================")

    # --- 1. SETUP ---
    config = load_config()
    client = OpenAI(base_url="http://192.168.1.67:1234/v1", api_key="not-needed")
    scenario_id_to_test = sys.argv[1] if len(sys.argv) > 1 else "8807e9963f411c48"
    print(f"--- Testing scenario: {scenario_id_to_test} ---")

    # --- 2. Load All Assets ---
    try:
        npz_dir = config['data']['processed_npz_dir']
        npz_path = os.path.join(npz_dir, 'validation', f"{scenario_id_to_test}.npz")
        scenario_data = load_npz_scenario(npz_path)
        
        legend_image = Image.open("outputs/legend_assets/visual_legend.png").convert("RGB")
        gif_path = os.path.join("outputs/preprocessed_scenarios", scenario_id_to_test, "scenario.gif")
        gif_image = Image.open(gif_path)
        
        key_frame_indices = [int(gif_image.n_frames * p) for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]]
        key_frames = [gif_image.seek(i) or gif_image.convert("RGB").copy() for i in key_frame_indices]
        print(f"✅ Loaded all assets for {scenario_id_to_test}.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load assets. {e}")
        return

    # --- 3. CONSTRUCT THE MESSAGES LIST (THE CORRECT WAY) ---
    print("\n--- Building final prompt payload ---")
    
    # The user_content list will hold ALL parts of the user's turn
    user_content = []

    # Part 1: The Visual Legend
    user_content.append({"type": "text", "text": "Use this visual legend to identify all objects:"})
    user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(legend_image)}})

    # Part 2: The Main Task Instruction
    user_content.append({"type": "text", "text": f"\n\n{USER_TASK_KIMI}"})
    
    # Part 3: The Interleaved Keyframes and Ground-Truth Data
    user_content.append({"type": "text", "text": "\n\n--- Scenario Keyframes & Ground-Truth Data ---"})
    
    for i, frame in enumerate(key_frames):
        frame_idx = key_frame_indices[i]
        
        av_tl_state = get_av_traffic_light_state_at_ts(scenario_data, frame_idx)
        has_conflict, conflict_type, conflict_id = check_for_path_conflict(scenario_data, frame_idx)
        top_threats = get_top_k_threats(scenario_data, frame_idx, k=3)

        conflict_str = f"YES ({conflict_type}, ID: {conflict_id})" if has_conflict else "NO"
        threats_str = "\n".join([f"    - {t}" for t in top_threats]) if top_threats else "    - None"
        
        frame_label = (
            f"\n**Frame {i+1} (Timestep: {frame_idx})**\n"
            f"**--- Ground Truth ---**\n"
            f"  - **AV Traffic Light:** {av_tl_state}\n"
            f"  - **AV Path Conflict:** {conflict_str}\n"
            f"  - **Top 3 Dynamic Agents (sorted by interaction score):**\n{threats_str}"
        )
        
        user_content.append({"type": "text", "text": frame_label})
        user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(frame)}})

    # The final messages list is now clean and correct
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    # --- 4. Run Inference & Parse Output ---
    print("\n--- Sending prompt to LM Studio server... ---")
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=messages,
            max_tokens=4096,
            temperature=0.1
        )
        output_text = response.choices[0].message.content
        
        thinking_part, final_answer_part = extract_thinking_and_summary(output_text)

        print("\n" + "="*20 + " MODEL'S INTERNAL THOUGHT PROCESS " + "="*20)
        print(thinking_part)

        print("\n" + "="*20 + " FINAL CAUSAL AUDIT " + "="*20)
        print(final_answer_part)
        
        save_debug_dossier(scenario_id_to_test, messages, output_text)

    except Exception as e:
        print(f"❌ ERROR: Could not connect or generate. Details: {e}")
        save_debug_dossier(scenario_id_to_test, messages, model_output=f"SCRIPT FAILED:\n{e}")

    print("\n--- END OF SCRIPT ---")

if __name__ == "__main__":
    main()