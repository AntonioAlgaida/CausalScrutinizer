# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
import requests
from io import BytesIO

# --- 1. SETUP THE MODEL (Same Quantization, New Model) ---

# We absolutely need 4-bit quantization for a 12B model on a 3090
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 # float16 is fine for our setup
)

# Define the Gemma 3 model ID
model_id = "google/gemma-3-12b-it"

# IMPORTANT: Use the correct model class for Gemma 3
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, 
    quantization_config=quantization_config,
    device_map="auto"
    # Note: No need to specify torch_dtype here, the quantization config handles it
)

# Load the processor, which is specific to the model
processor = AutoProcessor.from_pretrained(model_id)

# %%
# --- 2. PREPARE YOUR INPUTS (Crucial: Gemma 3 Chat Format) ---

# Load our test image (the car crash)
# Replace with the actual local path if you have it downloaded
crash_image = Image.open("test2.jpg").convert("RGB")


# This is the "general" prompt we are using for our test.
# We want to see if the model can identify the risk without being told to.
prompt_text = "Describe this scene in detail. Is there any risk?"

# Create the chat structure that Gemma 3 expects.
# The user "content" is a LIST of dictionaries.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": crash_image}, # Pass the PIL image object directly
            {"type": "text", "text": prompt_text},
        ]
    }
]

# The processor will now correctly combine the text and image based on this template.
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,  # <--- THIS LINE IS THE FIX
    return_tensors="pt"
).to("cuda")

# %%
# --- 3. RUN INFERENCE ---

# Generate the text
# Using a slightly higher token limit to allow for a detailed description
generated_ids = model.generate(**inputs, max_new_tokens=500)

# The Gemma processor needs the full output to decode correctly, not just the new tokens
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# --- 4. PRINT AND ANALYZE ---
# We need to manually "clean" the output to only see the assistant's response
# The output will contain our full prompt.
# Let's find the start of the assistant's part. A common marker is "<start_of_turn>model"
# or simply find the end of our own prompt in the output.

user_prompt_part_end = prompt_text
if user_prompt_part_end in generated_text:
    assistant_response = generated_text.split(user_prompt_part_end, 1)[1].strip()
else:
    # Fallback if the exact text isn't found (might have special tokens)
    assistant_response = "Could not automatically isolate assistant response. Full output:\n" + generated_text

print("--- MODEL OUTPUT (Gemma 3 12B) ---")
print(assistant_response)
print("\n--- END OF EXPERIMENT ---")
# %%