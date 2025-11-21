# %%
from llama_cpp import Llama
# We need this handler to correctly load the vision model
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image
import base64
from io import BytesIO

# --- 1. SETUP THE MODEL with the Vision Projector ---
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

# --- 2. PREPARE YOUR INPUTS (Unchanged) ---
crash_image = Image.open("pygame_test.jpg").convert("RGB")
prompt_text = "This is a simplified, top-down pygame rendering of a traffic scenario. The gray shapes are roads, and the colored rectangles are vehicles. Describe this scene in detail. Is there a collision or accident?"

def image_to_data_uri(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

image_uri = image_to_data_uri(crash_image)

messages = [
    {
        "role": "system",
        "content": "You are an expert in driving safety and risk analysis. Analyze the provided image to determine if an accident has occurred."
    },
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": prompt_text}
        ]
    }
]

# --- 3. RUN INFERENCE ---
print("\n--- Generating response... ---")
response = llm.create_chat_completion(
    messages=messages,
    max_tokens=500,
    temperature=0.1
)

# --- 4. PRINT AND ANALYZE ---
assistant_response = response['choices'][0]['message']['content']

print("--- MODEL OUTPUT (Gemma 3 12B QAT GGUF - Full Multimodal) ---")
print(assistant_response)
print("\n--- END OF EXPERIMENT ---")
# %%