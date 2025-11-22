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
from src.reasoning.prompt_builder import get_av_traffic_light_state_at_ts
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

4.  **Prioritize Conflict Inception:** The ground-truth `AV Path Conflict Status` is your key signal.
    *   A status of **`CONFLICT_INCEPTION`** marks the beginning of a new, safety-critical event. This is the **critical moment** for your analysis.
    *   A status of **`CONFLICT_SUSTAINED`** indicates a stable, ongoing situation (like car-following) that is less urgent.
    *   A status of **`NO_CONFLICT`** indicates the path is clear.
"""

USER_TASK_KIMI = """Your task is to produce a **Comprehensive Causal Audit** by **deeply synthesizing** the visual evidence with the provided Ground-Truth Data.

--- EXAMPLE OF A PERFECT CAUSAL AUDIT ---

Here is an example of a perfect, expert-level audit for a fictional scenario. You MUST follow this format and match this level of detail, tone, and reasoning quality.

**Fictional Scenario Context:**
A Magenta AV is approaching a large, signalized intersection. Its traffic light is GREEN. An Orange Pedestrian, initially on the sidewalk, suddenly steps into the crosswalk directly in the AV's path.

**Example Ground-Truth Data (Abbreviated):**
> **Frame 1 (Timestep 10):** AV Traffic Light: GREEN, AV Path Conflict: NO
> **Frame 2 (Timestep 20):** AV Traffic Light: GREEN, AV Path Conflict: NO
> **Frame 3 (Timestep 30):** AV Traffic Light: GREEN, AV Path Conflict: YES (PEDESTRIAN, ID: 42)
> **Frame 4 (Timestep 40):** AV Traffic Light: GREEN, AV Path Conflict: YES (PEDESTRIAN, ID: 42)

**Example of the Perfect Output You Should Generate:**

**Step 1: Detailed Scene Elaboration.**
The scene is a large, signalized urban intersection with three lanes of travel in each direction. The environment is complex, featuring multiple striped crosswalks, a dark red bike lane on the right, and several gray driveways. The Magenta AV is positioned in the center lane, and its initial "ego-up" orientation indicates a clear intention to proceed straight. The scene is populated with several other Blue Vehicles and multiple Orange Pedestrians on the sidewalks. The presence of SUMO-style traffic light bars confirms that this is a signal-controlled junction.

**Step 2: Factual Event Chronology (Detailed Frame-by-Frame Analysis).**
Summarize the key sequence of events. For each event, you **MUST** briefly describe the visual evidence that corresponds to the ground-truth data.
*   **Frame 1 (Timestep 10):** The Magenta AV is proceeding towards the intersection at a steady speed. The ground truth confirms its traffic light is **GREEN** and there is **NO** path conflict. Visually, an Orange Pedestrian (ID: 42) is visible on the right-hand sidewalk, stationary and not indicating intent to cross. The situation is nominal.
*   **Frame 2 (Timestep 20):** The AV continues its approach. The ground truth remains nominal (`AV Traffic Light: GREEN`, `AV Path Conflict: NO`). Visually, the pedestrian (ID: 42) appears to have turned towards the street, but is still on the sidewalk. This is a subtle visual cue of a potential change in intent.
*   **Frame 3 (Timestep 30) - THE CRITICAL MOMENT:** The AV is now close to the crosswalk. The ground truth flags a critical state change: **`AV Path Conflict: YES (PEDESTRIAN, ID: 42)`**. This aligns perfectly with the visual evidence, which now shows the pedestrian has stepped off the curb and is entering the AV's lane. The ground truth confirms this is the primary dynamic agent, closing at 4.5 m/s.
*   **Frame 4 (Timestep 40):** The AV has begun to decelerate (inferred from the shrinking distance covered between frames). The ground truth confirms the **Path Conflict** is still active. Visually, the pedestrian is now fully in the center of the AV's lane.

**Step 3: Comprehensive Causal Risk Identification.**
-   **Primary Causal Risk:** The primary risk is an imminent collision with a vulnerable road user (Pedestrian 42) who has entered the AV's path unexpectedly.
    -   *Cause:* The pedestrian made an unpredictable decision to step into the crosswalk against the AV's green light.
    -   *Effect:* The ground-truth system correctly detected a path conflict at Timestep 30.
    -   *Consequence:* If the AV had maintained its course and speed based only on its green light, a high-severity impact would have been unavoidable.
-   **Secondary / Contributing Risks:** The AV's own forward momentum while proceeding through the "stale green" light is a contributing risk. While legally permissible, it reduces the available time to react to unexpected "expectation violation" events like this one.

**Step 4: Optimal Action and Counterfactual.**
-   **Optimal Action:** The single, safest action the AV must take at the critical moment (Timestep 30) is to initiate immediate, maximum-effort emergency braking.
-   **Justification:** The ground-truth `AV Path Conflict: YES` is a non-negotiable, safety-critical signal. It instantly overrides the green light. The AV's safety protocols must prioritize avoiding a collision with a pedestrian over maintaining its right-of-way.
-   **Counterfactual Analysis:** If the AV's system had a delay or failed to react to the conflict flag at Timestep 30, it would have struck the pedestrian. This would represent a catastrophic failure to handle a classic "jaywalking" or "sudden actor" scenario.

--- END OF EXAMPLE ---

Now, apply this exact same methodology and level of detail to the new scenario provided below.


--- DEEP REASONING DIRECTIVE ---
Before your final answer, you MUST engage in a deep `◁think▷` process. In your thinking, for **each and every frame**, you must perform this two-step analysis:
1.  **State the Ground Truth:** First, state the `AV Path Conflict Status` provided for that frame.
2.  **Find the Visual Evidence:** Then, you MUST describe what you see in the image that **supports or explains** that ground-truth fact. *Example: "The ground truth says Path Conflict is YES with Vehicle 78. Visually, I can see that the Magenta AV's dashed trail is pointed directly at the Blue Vehicle (ID 78), and they are only about one lane-dash apart."*

Only after completing this frame-by-frame visual grounding should you generate the final output.

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
    scenario_id_to_test = sys.argv[1] if len(sys.argv) > 1 else "fffec9df0f0059d1"
    print(f"--- Testing scenario: {scenario_id_to_test} ---")

    # --- 2. Load All Assets ---
    try:
        npz_dir = config['data']['processed_npz_dir']
        npz_path = os.path.join(npz_dir, 'training', f"{scenario_id_to_test}.npz")
        scenario_data = load_npz_scenario(npz_path)
        
        legend_image = Image.open("outputs/legend_assets/visual_legend.png").convert("RGB")
        gif_path = os.path.join("outputs/preprocessed_scenarios_v2", scenario_id_to_test, "scenario.gif")
        gif_image = Image.open(gif_path)
        
        key_frame_indices = [int(gif_image.n_frames * p) for p in [0.1, 0.2, 0.3 ,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
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
        has_conflict_raw, conflict_type, conflict_id = check_for_path_conflict(scenario_data, frame_idx)
        top_threats = get_top_k_threats(scenario_data, frame_idx, k=3)

        # conflict_str = f"YES ({conflict_type}, ID: {conflict_id})" if has_conflict else "NO"
        threats_str = "\n".join([f"    - {t}" for t in top_threats]) if top_threats else "    - None"
        
        # --- 2. THE NEW STATEFUL LOGIC ---
        conflict_status_str = "NO_CONFLICT"
        if has_conflict_raw:
            if conflict_id != last_conflict_id:
                # A new conflict has just started! This is the critical event.
                conflict_status_str = f"CONFLICT_INCEPTION (with {conflict_type} ID: {conflict_id})"
            else:
                # This is just a continuation of the previous conflict (car-following).
                conflict_status_str = f"CONFLICT_SUSTAINED (with {conflict_type} ID: {conflict_id})"
        
        # Update the state for the next iteration
        last_conflict_id = conflict_id if has_conflict_raw else None
    
        frame_label = (
            f"\n**Frame {i+1} (Timestep: {frame_idx})**\n"
            f"**--- Ground Truth ---**\n"
            f"  - **AV Traffic Light:** {av_tl_state}\n"
            f"  - **AV Path Conflict Status:** {conflict_status_str}\n" # New, more descriptive label
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
            temperature=0.2
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