import os
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

# --- Add project root to path for our src imports ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.reasoning.prompt_builder import build_prompt_v4
from src.data_processing.waymo_parser import load_npz_scenario
from src.utils.config_loader import load_config

def main():
    """
    Tests the Qwen3-VL-8B-Instruct model using the official Hugging Face
    transformers library with 4-bit quantization and Flash Attention 2.
    """
    # --- 1. CONFIGURATION AND MODEL SETUP ---
    print("--- Initializing Qwen3-VL-8B with Transformers ---")
    
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    
    # Define the 4-bit quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for modern GPUs
    )

    # Load the model with quantization and Flash Attention
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        device_map="auto" # Automatically use the GPU
    )

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_id)
    print("--- Model and Processor loaded successfully ---")

    # --- 2. LOAD SCENARIO ASSETS ---
    config = load_config()
    
    # This is the scenario ID where Gemma-8B failed the perception task
    scenario_id_to_test = "72ed65984ad6f37f" # Use the specific RLV scenario ID
    preprocessed_dir = "outputs/preprocessed_scenarios"
    scenario_assets_dir = os.path.join(preprocessed_dir, scenario_id_to_test)
    
    print(f"\n--- Loading assets for scenario: {scenario_id_to_test} ---")
    
    # Load the ground-truth context text (with the "Motion Status" fix)
    context_path = os.path.join(scenario_assets_dir, "context.txt")
    with open(context_path, 'r') as f:
        ground_truth_context = f.read()
        
    # Load the GIF (with the dashed trail) and select keyframes
    gif_path = os.path.join(scenario_assets_dir, "scenario.gif")
    gif_image = Image.open(gif_path)
    total_frames = gif_image.n_frames
    key_frame_indices = [int(total_frames * p) for p in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]]
    key_frames = [gif_image.seek(i) or gif_image.convert("RGB").copy() for i in key_frame_indices]
    
    # --- 3. BUILD THE PROMPT ---
    npz_path = os.path.join(config['data']['processed_npz_dir'], 'validation', f"{scenario_id_to_test}.npz")
    scenario_data = load_npz_scenario(npz_path)
    system_prompt, _, user_task_prompt = build_prompt_v4(scenario_data, 45) # Using V4, not the "Sledgehammer"

    visual_legend = """The following keyframes are based on this visual legend:

    MAP ELEMENTS:
    - Road Surface: The solid, drivable area, rendered in medium gray.
    - Non-Drivable Area: The black background.
    - Lane Markings: Thin white and yellow lines on the road surface.
    - Crosswalks: Semi-transparent white rectangular areas. *This element may not be present in all scenes.*
    - Traffic Lights: Colored circles (Red, Yellow, or Green) rendered on the road at the stop line. *This element may not be present in all scenes.*
    - Stop Signs: Solid red circles near an intersection. *This element may not be present in all scenes.*

    AGENTS:
    - The AV: The single **Magenta vehicle shape** (polygon with a tapered front).
    - The AV's Recent Path: A **dashed Magenta line** trailing the AV, indicating its trajectory over the last 2 seconds.
    - Other Vehicles: **Blue vehicle shapes** (polygons with a tapered front).
    - Pedestrians: Orange Circles.
    - Cyclists: **Cyan vehicle shapes** (smaller polygons with a tapered front)."""

    # --- Construct the messages list in the format Transformers expects ---
    content = []
    
    # Add the text prompt part
    full_user_prompt = f"{ground_truth_context}\n\n{visual_legend}\n\n{user_task_prompt}"
    content.append({"type": "text", "text": full_user_prompt})
    
    # Add the images
    for frame in key_frames:
        content.append({"type": "image", "image": frame})

    messages = [{"role": "user", "content": content}]

    # --- 4. RUN INFERENCE ---
    print("\n--- Running inference with Transformers... ---")
    
    # The processor handles the entire prompt templating and tokenization
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate the output
    with torch.no_grad(): # Use no_grad for inference to save memory
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False # Use deterministic decoding
        )
    
    # Decode the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # --- 5. DISPLAY RESULTS ---
    print("\n--- Qwen3-VL-8B Transformers Output ---")
    print(output_text)
    print("\n--- END OF EXPERIMENT ---")

if __name__ == "__main__":
    main()