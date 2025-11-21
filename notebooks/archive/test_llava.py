#%%
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import requests # To load a sample image if you don't have one

# --- 1. SETUP THE MODEL ---
# Use the same 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Define the LLaVA-NeXT model ID from the Hugging Face Hub
# We'll use the Vicuna-7B version, which is a strong, well-known open model
model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"

# Load the LLaVA model with our quantization config
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, 
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16 # It's good practice to specify the dtype
)

# Load the processor, which is specific to LLaVA-NeXT
processor = AutoProcessor.from_pretrained(model_id)
#%%
# --- 2. PREPARE YOUR INPUTS (LLaVA has a specific chat format) ---
gif_path = "video_1.gif"
gif_image = Image.open(gif_path)

# Your smart frame selection strategy
key_frame_indices = [4, 8, 16, 32] # Using 6 frames to keep the prompt cleaner

if max(key_frame_indices) >= gif_image.n_frames:
    print(f"Warning: GIF has only {gif_image.n_frames} frames. Adjusting indices.")
    key_frame_indices = [i for i in key_frame_indices if i < gif_image.n_frames]

key_frames = []
for i in key_frame_indices:
    gif_image.seek(i)
    key_frames.append(gif_image.convert("RGB"))

# The images should be in a simple list
images_list = key_frames

# Create the text part of the prompt.
# We will insert one <image> token for each image in our list.
text_content = "".join(["<image>"] * len(images_list)) + "\n"
text_content += "You are an expert driving safety instructor. Look at the sequence of frames from a driving scenario. What are the key causal risks in this scene? What is the single safest course of action for the ego vehicle (the car at the center)?"

# Now, build the chat structure that the processor's template expects.
# It's a list of dictionaries.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": text_content},
        ]
    }
]
#%%
# The processor will now correctly combine the text and images.
# add_generation_prompt=True adds the "ASSISTANT:" token for us.
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# --- 3. RUN INFERENCE ---
# The processor now takes the final prompt string and the list of images
inputs = processor(text=prompt, images=images_list, return_tensors="pt").to("cuda")

# Generate the text (this part is unchanged)
generated_ids = model.generate(**inputs, max_new_tokens=500)

# Decode and print the output (this part is unchanged)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
# %%

image_test = Image.open("test1.jpg").convert("RGB")
text_test = "Describe in detail the image <image> in detail. Is there any accident?"
inputs_test = processor(text=text_test, images=image_test, return_tensors="pt").to("cuda")
generated_ids_test = model.generate(**inputs_test, max_new_tokens=200)
generated_text_test = processor.batch_decode(generated_ids_test, skip_special_tokens=True)[0]
print(generated_text_test)  
# %%






# %%
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

# --- 1. SETUP THE MODEL & PROCESSOR ---

# Use the same 4-bit quantization config for consistency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# --- CHOOSE WHICH MODEL TO TEST FOR EXPERIMENT 2 ---
# Option A: A more advanced 7B parameter model
model_id = "llava-hf/llama3-llava-next-8b-hf"

# Option B: A much larger 34B parameter model. 
# This will be slower and use more VRAM, but may offer better reasoning.
# model_id = "llava-hf/llava-v1.6-34b-hf" 
# ----------------------------------------------------

print(f"--- Starting Experiment 2: Testing model: {model_id} ---")

# Load the selected LLaVA model with our quantization config
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, 
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load the corresponding processor
processor = AutoProcessor.from_pretrained(model_id)

# %%
# --- 2. PREPARE THE INPUTS (IMAGE AND PROMPT) ---

# The URL of the accident image we are using as our benchmark
image = Image.open("test2.jpg").convert("RGB")

# Our simple, direct prompt for this test
prompt_text = "<image>\nDescribe this scene in detail. Is there any accident? If so, explain what happened and why."

# Build the chat structure that the processor expects
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
        ]
    }
]

# The processor correctly applies the chat template and adds the generation prompt
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# %%
# --- 3. RUN INFERENCE ---

# The processor combines the prompt text and the image(s)
# Note: LLaVA-NeXT models handle a single image in a list
inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

print("Model inputs prepared. Starting generation...")

# Generate the text response
# We add a temperature parameter to reduce randomness and get the model's "best" answer
generated_ids = model.generate(**inputs, max_new_tokens=500, temperature=0.1)

# Decode and print the output
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n--- MODEL OUTPUT ---")
# We process the output slightly to remove the prompt part for cleaner reading
# The model's actual response comes after the "ASSISTANT: " token
assistant_response = generated_text.split("ASSISTANT: ")[-1].strip()
print(assistant_response)
print("--- END OF EXPERIMENT ---")

#%%
# %%
