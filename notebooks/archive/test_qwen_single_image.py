# File: test_qwen_single_image.py (v2 - with automatic download)

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image
import base64
from io import BytesIO
import os

# --- 1. SETUP THE QWEN MODEL (with automatic download) ---

# Define the Hugging Face repository and the specific files we want
QWEN_REPO_ID = "unsloth/Qwen3-VL-8B-Instruct-GGUF"
QWEN_LLM_FILENAME = "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
QWEN_PROJECTOR_FILENAME = "mmproj-F16.gguf"

print(f"--- Loading patched model from repo: {QWEN_REPO_ID} ---")
print(f"--- Main Model: {QWEN_LLM_FILENAME} ---")
print(f"--- Projector: {QWEN_PROJECTOR_FILENAME} ---")

print("\n--- Initializing Chat Handler with Vision Projector... ---")
chat_handler = Llava15ChatHandler.from_pretrained(
    repo_id=QWEN_REPO_ID,
    filename=QWEN_PROJECTOR_FILENAME
)


# Initialize the Llama model. This will download the main GGUF file if needed.
print("--- Chat Handler loaded. Initializing GGUF model... ---")
llm = Llama.from_pretrained(
    repo_id=QWEN_REPO_ID,
    filename=QWEN_LLM_FILENAME,
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=8096,
    verbose=False
)
print("--- Patched Qwen multimodal model loaded successfully. ---")

# --- 2. PREPARE THE INPUT IMAGE AND PROMPT (Unchanged) ---

# We will use the same successful schematic image for a direct comparison
IMAGE_PATH = "./notebooks/archive/assets/pygame_test.jpg"

if not os.path.exists(IMAGE_PATH):
    print(f"ERROR: Test image not found at {IMAGE_PATH}")
    # We still need a local check for the image, as it's our own asset.
    exit()

print(f"\n--- Loading test image: {IMAGE_PATH} ---")
test_image = Image.open(IMAGE_PATH).convert("RGB")

# Helper function to convert PIL Image to a Data URI
def pil_image_to_data_uri(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

# The prompt remains the same
system_prompt = "You are an expert in driving safety and risk analysis. Analyze the provided image to determine if an accident has occurred."
user_prompt_text = "This is a simplified, top-down rendering of a traffic scenario. The gray shapes are roads, and the colored rectangles are vehicles. Describe this scene in detail. Is there a collision or accident?"

# Construct the messages payload
messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": pil_image_to_data_uri(test_image)}},
            {"type": "text", "text": user_prompt_text}
        ]
    }
]

# --- 3. RUN INFERENCE (Unchanged) ---
print("\n--- Generating response from Qwen... ---")
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=500,
    temperature=0.1
)

# --- 4. PRINT AND ANALYZE THE OUTPUT (Unchanged) ---
assistant_response = response['choices'][0]['message']['content']

print("\n--- MODEL OUTPUT (Qwen3-VL-8B-Instruct-GGUF) ---")
print(assistant_response)
print("\n--- END OF EXPERIMENT ---")