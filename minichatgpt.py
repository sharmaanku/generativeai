#!/usr/bin/env python3
"""
mini_chatgpt_updated.py
Windows-safe, CPU-only, user-prompt text generation
Uses gpt2-medium for better output
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import os

# --- CPU safe settings ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable GPU
torch.set_num_threads(1)                # prevent thread issues on Windows

# --- Get user prompt ---
prompt = input("💬 Enter your prompt: ").strip()
if not prompt:
    prompt = "Once upon a time, a curious developer built a talking AI"

print("\n🔹 Loading model (gpt2-medium, CPU)...")
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# --- Create text generation pipeline ---
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU only
)

set_seed(42)

print("🔹 Generating text... This may take a few seconds on CPU.")

# --- Generate text with top-k / top-p sampling to reduce repetition ---
out = generator(
    prompt,
    max_length=200,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
generated_text = out[0]["generated_text"]

# --- Save output to file ---
out_file = "generated_text.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(generated_text)

# --- Display output ---
print("\n✅ --- Generated Text ---\n")
print(generated_text)
print(f"\n📄 Saved to: {os.path.abspath(out_file)}")

