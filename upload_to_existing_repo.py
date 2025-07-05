#!/usr/bin/env python3
"""
Simple script to upload a model to an existing Hugging Face repository
"""

import argparse
import json
import yaml
from pathlib import Path
from huggingface_hub import upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer


def upload_to_existing_repo(model_path: str, repo_name: str, token: str):
    """Upload model to an existing Hugging Face repository"""

    model_path = Path(model_path)

    if not model_path.exists():
        raise ValueError(f"Model path {model_path} does not exist")

    # Test model loading
    print("Testing model loading...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✓ Model loaded successfully ({model.num_parameters():,} parameters)")
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    # Load training config and metrics if available
    training_config = {}
    training_metrics = {}

    config_file = model_path / "training_config.yaml"
    if config_file.exists():
        with open(config_file, "r") as f:
            training_config = yaml.safe_load(f)

    metrics_file = model_path / "training_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            training_metrics = json.load(f)

    # Create a simple README if one doesn't exist
    readme_path = model_path / "README.md"
    if not readme_path.exists():
        model_name = training_config.get("model", {}).get("name", "BabyLlama-10M")
        num_params = training_metrics.get("num_parameters", "~10M")
        train_samples = training_metrics.get("train_samples", "N/A")

        readme_content = f"""# {model_name}

A small LLaMA model trained using the BabyLlama framework.

## Model Details
- **Parameters**: {num_params:,}
- **Training Samples**: {train_samples:,}
- **Architecture**: LLaMA

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

inputs = tokenizer("Hello", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training

This model was trained using the [BabyLlama](https://github.com/pgryko/BabyLlama) framework.
"""

        with open(readme_path, "w") as f:
            f.write(readme_content)
        print("✓ Created README.md")

    # Upload model
    print(f"Uploading model to {repo_name}...")
    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            token=token,
            commit_message=f"Upload {training_config.get('model', {}).get('name', 'BabyLlama')} model",
        )

        print(f"✓ Model successfully uploaded to: https://huggingface.co/{repo_name}")
        return f"https://huggingface.co/{repo_name}"

    except Exception as upload_error:
        print(f"❌ Upload failed: {upload_error}")
        raise upload_error


def main():
    parser = argparse.ArgumentParser(
        description="Upload model to existing Hugging Face repository"
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the trained model directory"
    )
    parser.add_argument(
        "repo_name", type=str, help="Existing Hugging Face repository name"
    )
    parser.add_argument("--token", type=str, help="Hugging Face API token")

    args = parser.parse_args()

    try:
        url = upload_to_existing_repo(
            model_path=args.model_path, repo_name=args.repo_name, token=args.token
        )
        print(f"\n🎉 Success! Your model is now available at: {url}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
