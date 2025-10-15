from diffusers import StableDiffusionPipeline
import torch

# ----------------- Load Model -----------------
# Use a lightweight model: CompVis/stable-diffusion-v1-4 (or a smaller variant if needed)
model_id = "runwayml/stable-diffusion-v1-5"  # lightweight, small weights
device = "cpu"  # CPU mode

print("✅ Loading Stable Diffusion model (lightweight)...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

# ----------------- Generate Image -----------------
def generate_image(prompt: str, filename: str = "generated_image.png"):
    """
    Generate an image from a text prompt and save it locally.
    """
    print(f"🎨 Generating image for prompt: '{prompt}' ...")
    image = pipe(prompt, num_inference_steps=20).images[0]  # fewer steps = faster
    image.save(filename)
    print(f"✅ Image saved as {filename}")

# ----------------- Test Scenario -----------------
if __name__ == "__main__":
    prompt = input("Enter your image prompt: ")
    generate_image(prompt)
