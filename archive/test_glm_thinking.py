import os
import sys
import base64
import re
from io import BytesIO
from PIL import Image
from openai import OpenAI

# --- CONFIGURATION ---
LM_STUDIO_URL = "http://192.168.1.67:1234/v1"
MODEL_ID = "local-model" # Ensure GLM-4.1V is loaded in LM Studio

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCENARIO_ID = "84a1c7968294e32d" # The Slip Lane Scenario
GIF_PATH = f"outputs/preprocessed_scenarios/{SCENARIO_ID}/scenario.gif"
LEGEND_PATH = "outputs/legend_assets/visual_legend.png"

# --- GLM-SPECIFIC PROMPT ---
# GLM doesn't need the "Step 1, Step 2" forced structure. 
# It needs a clear goal. We strip the CoT instructions and let it think.
SYSTEM_PROMPT = """You are an AV Safety Auditor. Analyze the sensor visualization.
CRITICAL RULES:
1. **Cyan Triangles** are Cyclists.
2. **Red Bars** are Traffic Lights.
3. **Dashed Magenta Line** is the AV's past path.
"""

USER_PROMPT = """
Analyze this scenario. 
1. Is the AV in a Slip Lane? 
2. Does the Red Light apply to the AV? 
3. Is there a Cyclist?
Finally, give a Safety Score (1-5).
"""

def pil_image_to_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def extract_thinking(text: str):
    """Parses GLM's <think> tags."""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return thinking, answer
    return None, text

def main():
    print(f"--- Testing GLM-4.1V-9B-Thinking on {SCENARIO_ID} ---")
    
    # 1. Load Images
    legend_uri = pil_image_to_data_uri(Image.open(LEGEND_PATH).convert("RGB"))
    gif_img = Image.open(GIF_PATH)
    # GLM handles context well, lets give it 4 key frames
    indices = [int(gif_img.n_frames * p) for p in [0.1, 0.4, 0.7, 0.9]]
    
    content = []
    content.append({"type": "text", "text": "Visual Legend:"})
    content.append({"type": "image_url", "image_url": {"url": legend_uri}})
    content.append({"type": "text", "text": USER_PROMPT})
    
    for i, idx in enumerate(indices):
        gif_img.seek(idx)
        frame = gif_img.convert("RGB")
        content.append({"type": "text", "text": f"\nFrame {i+1}:"})
        content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(frame)}})

    # 2. Inference
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")
    
    try:
        print("⏳ Sending request (GLM Thinking)...")
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            max_tokens=2048,
            temperature=0.6 # Thinking models need slight temp to be creative
        )
        
        raw_output = response.choices[0].message.content
        thinking, answer = extract_thinking(raw_output)
        
        print("\n" + "="*20 + " 🧠 INTERNAL THOUGHT PROCESS " + "="*20)
        print(thinking if thinking else "No <think> tags found (Model might not be supporting them in API)")
        
        print("\n" + "="*20 + " 🎯 FINAL VERDICT " + "="*20)
        print(answer)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()