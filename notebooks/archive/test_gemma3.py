# %%
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image
import base64
from io import BytesIO
import os

# --- 1. SETUP THE MODEL (Same as before) ---
# Ensure you have downloaded these files to your project directory
repo_id = "google/gemma-3-12b-it-qat-q4_0-gguf"
llm_filename = "gemma-3-12b-it-q4_0.gguf"
# The crucial multimodal projector file
projector_filename = "mmproj-model-f16-12B.gguf"

print("--- Initializing Chat Handler with Vision Projector... ---")
# This handler tells llama.cpp how to process the image
chat_handler = Llava15ChatHandler.from_pretrained(
    repo_id=repo_id,
    filename=projector_filename
)
print("--- Chat Handler loaded. Initializing GGUF model... ---")

llm = Llama.from_pretrained(
    repo_id=repo_id,
    filename=llm_filename,
    chat_handler=chat_handler, # <-- We pass the vision-enabled handler here
    n_gpu_layers=-1,
    n_ctx=8096,
    verbose=False
)
print("--- Multimodal model loaded successfully. ---")


# --- 2. PREPARE MULTI-IMAGE INPUTS ---

# --- Helper function to convert PIL Image to a Data URI ---
def pil_image_to_data_uri(image):
    """Converts a PIL Image object to a base64-encoded data URI."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

# --- Load the GIF and select keyframes ---
gif_path = "outputs/rendered_gifs/v2_render_72ed65984ad6f37f.gif"
print(f"\n--- Loading GIF and selecting keyframes from '{gif_path}'... ---")
gif_image = Image.open(gif_path)

print(f"Total frames in GIF: {gif_image.n_frames}")

# Our "storytelling" keyframe selection strategy
# The GIF has 30 frames, collision is at frame 15.
total_frames = gif_image.n_frames
key_frame_indices = [
    int(total_frames * 0.1),  # Early state
    int(total_frames * 0.3),  # Mid-approach
    int(total_frames * 0.5),  # Point of conflict/interaction
    int(total_frames * 0.9)   # Resolution/final state
]
print(f"Selected keyframe indices: {key_frame_indices}")

key_frames = []
for i in key_frame_indices:
    gif_image.seek(i)
    # We must copy the frame, as seek() loads them lazily
    key_frames.append(gif_image.convert("RGB").copy())

# --- Construct the messages payload ---
# The prompt is now much more sophisticated, asking for sequential analysis
system_prompt_v2 = """
You are a world-class expert in autonomous vehicle safety and causal reasoning, with a specific focus on analyzing sensor data. You will be shown a series of keyframes from a scenario taken from the Waymo Open Motion Dataset.

CRITICAL CONTEXT: The visualization is a simplified, top-down rendering of the ego-centric reality as perceived by an Autonomous Vehicle (AV). This is not a god's-eye view. This means:
- Occlusions are real: Objects may be hidden behind others. What you see is all the AV could see.
- Partial Views: You may only see the traffic lights relevant to the AV, not all lights at an intersection.
- Sensor Limitations: Objects can suddenly appear or disappear at the edges of the sensor range or due to detection failures.

Your task is to analyze these potentially incomplete scenes with the mindset of a safety engineer auditing the AV's behavior.
"""

user_prompt_v2 = """
The following keyframes are based on this visual legend:

MAP ELEMENTS:
- Road Surface: The solid, drivable area, rendered in medium gray.
- Non-Drivable Area: The black background.
- Lane Markings: Thin white and **yellow lines** on the road surface.
- Crosswalks: Semi-transparent white rectangular areas. *This element may not be present in all scenes.*
- Traffic Lights: Colored circles (Red, **Yellow**, or Green) rendered on the road at the stop line. A yellow circle in this context indicates a CAUTION state. *This element may not be present in all scenes.*
AGENTS:
- The AV: The single **Magenta Rectangle**. Your analysis should focus on its actions.
- Other Vehicles: **Blue Rectangles**.
- Pedestrians: **Orange Circles**. *This element may not be present in all scenes.*
- Cyclists: **Cyan Rectangles**. *This element may not be present in all scenes.*

---
QUESTIONS:
1. **Factual Scene Description:** Based *only on the visual evidence and the legend provided*, describe the sequence of events. Note any significant interactions between the AV (Green) and other agents.
2. **Causal Risk Analysis:** What are the primary causal risks for the AV in this scene? Specifically consider risks arising from **partial information, potential occlusions, and the unpredictable behavior of pedestrians and other vehicles.**
3. **Safest Course of Action:** What is the single safest maneuver for the AV to execute in the next few seconds to mitigate these risks? Be specific (e.g., 'yield and wait for the blue car to pass,' 'inch forward to improve visibility,' 'cover the brake').
"""

# Build the content list for the user message.
# We will add all images first, followed by the text prompt.
user_content = []
for frame in key_frames:
    user_content.append({"type": "image_url", "image_url": {"url": pil_image_to_data_uri(frame)}})
user_content.append({"type": "text", "text": user_prompt_v2})

messages = [
    {"role": "system", "content": system_prompt_v2},
    {"role": "user", "content": user_content}
]


# --- 3. RUN INFERENCE ---
print("\n--- Generating response for multi-image input... ---")
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=700, # Increased max_tokens for a more detailed response
    temperature=0.1 # Keep it factual and deterministic
)

# --- 4. PRINT AND ANALYZE ---
assistant_response = response['choices'][0]['message']['content']

print("\n--- MODEL OUTPUT (Gemma 3 12B - Multi-Frame Causal Analysis) ---")
print(assistant_response)
print("\n--- END OF EXPERIMENT ---")
# %%