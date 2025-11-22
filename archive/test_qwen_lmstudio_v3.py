import os
import sys
import random
import base64
import pandas as pd
import datetime
from io import BytesIO
from PIL import Image
from openai import OpenAI

# --- Add project root ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_config

# --- CONFIGURATION ---
LM_STUDIO_URL = "http://192.168.1.67:1234/v1"
MODEL_ID = "local-model"
LOG_FILE = "outputs/prompt_engineering_log.txt"

# --- PROMPT V16: THE UNIVERSAL TRAFFIC AUDITOR ---

# --- PROMPT V17: THE SIGNAL MAPPER ---

SYSTEM_PROMPT_V18 = """You are a Senior AV Safety Auditor.
You are analyzing a **Schematic Sensor Visualization** from the Waymo Open Motion Dataset, similar to SUMO traffic simulator.

--- EPISTEMIC CONTEXT ---
1. **Sensor Data:** This is a Lidar/Map visualization.
2. **Trust the Pixels:** Do not use your "common sense" about how intersections usually work. Only report what is explicitly drawn.
3. **No Hallucinations:** - If you see a **Continuous Green Bar**, do NOT imagine a Red segment just because it's a left turn lane.
   - **Cyan Triangle** = Cyclist.
   - **Blue Rectangle** = Vehicle.
"""

VISUAL_RULES_V18 = """
--- VISUAL DIALECT (STRICT SIGNAL RULES) ---

1. **The Signal Mapping Protocol (CRITICAL):**
   - **The "Spanning Bar" Rule:** Often, a single bar spans across MULTIPLE lanes.
     - If a Green Bar touches the Left Lane AND the Straight Lane, it is **GREEN for BOTH**.
     - Do NOT split a continuous bar into imaginary segments (e.g., "Red Left, Green Straight") unless you distinctly see different colors.
   - **The "Pixel-Spot" Check:** Look specifically at the pixels *directly in front* of the Magenta AV's nose. What color is the bar *at that exact point*?

2. **The Agents:**
   - **Ego AV:** MAGENTA Polygon + Dashed Trail.
   - **Vehicles:** BLUE Rectangles.
   - **Cyclists:** CYAN TRIANGLES.

"""

USER_TASK_V17 = """
--- YOUR AUDIT TASK ---
Perform a step-by-step "Causal Audit". Follow the Signal Mapping Protocol strictly.

**Stage 1: Perception & Signal Mapping (Chain-of-Thought)**
* **Step 1.1: Scenario Identification**:
    - Describe the road type (Intersection, Highway, etc.) and visible map elements.
* **Step 1.2: Lane Identification:**
    - Look at the **Dashed Magenta Trail**. Which specific lane is the AV in?.
* **Step 1.3: The Signal Filter:**
    - Identify **ALL** visible Traffic Light Bars.
    - **FILTER:** For each bar, ask: "Does this bar physically block the AV's current lane?" Do this only in the first frame.
    - **CONCLUSION:** What color is the *Relevant* signal? (If no bar blocks the AV's specific lane, state "None/Unprotected").
* **Step 1.4: Conflict Scan**

**Stage 2: Scenario Synthesis**
"The AV is executing a [Maneuver]. The relevant signal is [Color/None]. The primary conflict is [Agent Type]."

**Stage 3: The Causal Audit**
* **Safety Score:** 1 (Dangerous) to 5 (Perfect).
* **Root Cause:** Did the AV run a *relevant* Red Light? Did it fail to yield to *oncoming* traffic?
* **Recommendation:** Proceed, Yield, or Stop?
"""

# --- HELPER FUNCTIONS ---

def pil_image_to_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def get_random_scenario_id() -> str:
    csv_path = "data/mined_scenarios/critical_scenario_ids_v1.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Mined scenarios CSV not found.")
    df = pd.read_csv(csv_path)
    return random.choice(df['scenario_id'].tolist())

def log_result(scenario_id, prompt_version, output_text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*40}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"SCENARIO: {scenario_id}\n")
        f.write(f"PROMPT: {prompt_version}\n")
        f.write(f"{'-'*40}\n")
        f.write(output_text)
        f.write(f"\n{'='*40}\n")

def main():
    print("--- Spot-Checking Qwen3 (Prompt V11) ---")
    
    # 1. Pick Scenario
    override_id = "8807e9963f411c48" # Set this to test a specific ID
    # override_id = None
    
    scenario_id = override_id if override_id else get_random_scenario_id()
    print(f"🎯 Selected Scenario: {scenario_id}")
    
    # 2. Load Assets
    preprocessed_dir = "outputs/preprocessed_scenarios"
    gif_path = os.path.join(preprocessed_dir, scenario_id, "scenario.gif")
    legend_path = "outputs/legend_assets/visual_legend.png"
    
    # Print here the url of the scenario gif to visuallize the scenario from the console with a click
    print(f"Scenario GIF Path: {os.path.abspath(gif_path)}")


    if not os.path.exists(gif_path):
        print(f"❌ GIF not found for {scenario_id}. Run preprocess_scenarios.py first.")
        return

    # 3. Prepare Images
    legend_uri = pil_image_to_data_uri(Image.open(legend_path).convert("RGB"))
    
    gif_img = Image.open(gif_path)
    indices = [int(gif_img.n_frames * p) for p in [0.0, 0.3, 0.4, 0.6, 0.8, 0.9]]
    key_frames = []
    for i in indices:
        gif_img.seek(i)
        key_frames.append(gif_img.convert("RGB").copy())

    # 4. Build Payload
    content = []
    content.append({"type": "text", "text": "Reference Visual Legend:"})
    content.append({"type": "image_url", "image_url": {"url": legend_uri}})
    content.append({"type": "text", "text": VISUAL_RULES_V18})
    content.append({"type": "text", "text": USER_TASK_V17})
    content.append({"type": "text", "text": "\n--- SCENARIO KEYFRAMES ---"})
    
    for i, frame in enumerate(key_frames):
        content.append({"type": "text", "text": f"\n**Frame {i+1} (Timestep: {indices[i]})**"})
        content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(frame)}})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V18},
        {"role": "user", "content": content}
    ]

    # 5. Run Inference
    print("⏳ Sending request to LM Studio...")
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")
    
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=2048,
            temperature=0.1
        )
        output_text = response.choices[0].message.content
        
        print("\n" + "*"*20 + " QWEN RATIONALE " + "*"*20)
        print(output_text)
        print("*"*56)
        
        log_result(scenario_id, "V11", output_text)
        print(f"✅ Result saved to {LOG_FILE}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()