from transformers import LlamaForCausalLM, GPT2TokenizerFast
import torch

# Load model and tokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="./models/Llama-10M",
    help="Path to model directory",
)
args = parser.parse_args()

model_path = args.model_path
model = LlamaForCausalLM.from_pretrained(model_path)
tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

# Test generation
prompts = ["The cat", "A student", "The scientist"]

print(f"Model loaded from: {model_path}")
print(f"Model parameters: {model.num_parameters():,}")
print("\nGenerating text:\n")

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)
    assert len(generated_text) > 0, "Generated text is empty!"
