import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

# --- 1. SETUP THE MODEL ---
# This is the key to making it run on your GPU.
# We are telling the library to load the model in 4-bit precision.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Define the model ID from the Hugging Face Hub
model_id = "HuggingFaceM4/idefics2-8b"

# Load the processor (handles text and image formatting)
processor = AutoProcessor.from_pretrained(model_id)

# Load the model with our quantization config
# device_map="auto" tells it to automatically use the GPU
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

# --- 2. PREPARE YOUR INPUTS ---
# Let's say you have your scenario GIF saved
gif_path = "video_1.gif"
gif_image = Image.open(gif_path)

# A GIF is a sequence of frames. We need to extract them.
# The VLM can't watch a long video, so we will select a few key frames.
# For example, let's pick the start, middle, and end frames.
total_frames = gif_image.n_frames
key_frame_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # Adjust based on your GIF length
key_frames = []
for i in key_frame_indices:
    gif_image.seek(i)
    # We need to convert the frame to RGB format
    key_frames.append(gif_image.convert("RGB"))

# A more dynamic way to build the prompt to handle any number of images
prompt_parts = [
    "You are an expert driving safety instructor. Look at the following driving scenario shown in this sequence of frames.",
]
# Add an <image> token for each frame
for _ in key_frames:
    prompt_parts.append("<image>")

# Add the final question
prompt_parts.append(
    "\nQuestion: What are the key causal risks in this scene? What is the single safest course of action for the ego vehicle (the car at the center)? Answer:"
)

# Join all the parts together with newlines
text_prompt = "\n".join(prompt_parts)

# The list of image objects remains the same
images_list = key_frames
# --- 3. RUN INFERENCE (The new, more robust way) ---
# We now pass the text and images as separate arguments.
# We wrap them in lists to create a "batch of one", which is what the processor expects.
inputs = processor(text=[text_prompt], images=[images_list], return_tensors="pt").to("cuda")

# Generate the text (this part is unchanged)
generated_ids = model.generate(**inputs, max_new_tokens=500)

# Decode and print the output (this part is unchanged)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)