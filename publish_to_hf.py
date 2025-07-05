#!/usr/bin/env python3
"""
Script to publish trained BabyLlama models to Hugging Face Hub
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_model_card(
    model_path: Path,
    repo_name: str,
    training_config: dict,
    training_metrics: dict,
    description: Optional[str] = None,
) -> str:
    """Create a comprehensive model card for the Hugging Face Hub"""

    # Extract key information
    model_name = training_config.get("model", {}).get("name", "BabyLlama-10M")
    num_params = training_metrics.get("num_parameters", "~10M")
    train_loss = training_metrics.get("train_loss", "N/A")
    eval_loss = training_metrics.get("eval_loss", "N/A")
    eval_perplexity = training_metrics.get("eval_perplexity", "N/A")
    train_samples = training_metrics.get("train_samples", "N/A")

    model_card = f"""---
language: en
license: mit
tags:
- text-generation
- pytorch
- causal-lm
- babylm
- small-language-model
datasets:
- synthetic
metrics:
- perplexity
model_index:
- name: {model_name}
  results:
  - task:
      type: text-generation
      name: Text Generation
    metrics:
    - type: perplexity
      value: {eval_perplexity}
      name: Perplexity
---

# {model_name}

{description or "A small 10M parameter LLaMA model trained on synthetic data as part of the BabyLM challenge."}

## Model Details

- **Model Type**: Causal Language Model (LLaMA architecture)
- **Parameters**: {num_params:,} parameters
- **Training Data**: Synthetic text data ({train_samples:,} samples)
- **Architecture**: 
  - Hidden Size: {training_config.get("model", {}).get("hidden_size", 192)}
  - Layers: {training_config.get("model", {}).get("n_layer", 6)}
  - Attention Heads: {training_config.get("model", {}).get("n_head", 6)}
  - Sequence Length: {training_config.get("data", {}).get("seq_length", 128)}

## Training Details

- **Training Loss**: {train_loss}
- **Evaluation Loss**: {eval_loss}
- **Perplexity**: {eval_perplexity}
- **Learning Rate**: {training_config.get("training", {}).get("lr", "3e-4")}
- **Batch Size**: {training_config.get("training", {}).get("batch_size", 32)}
- **Epochs**: {training_config.get("training", {}).get("num_epochs", 2)}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Generate text
inputs = tokenizer("The quick brown fox", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, temperature=0.8, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Training Framework

This model was trained using the [BabyLlama](https://github.com/pgryko/BabyLlama) framework, which provides:

- Modern training pipeline with HuggingFace Transformers
- Efficient data processing and tokenization
- Comprehensive evaluation metrics
- Support for multiple architectures (LLaMA, GPT-2, GPT-J)

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{babyllama2024,
  title={{BabyLlama: Training Small Language Models from Scratch}},
  author={{BabyLlama Team}},
  year={{2024}},
  url={{https://github.com/pgryko/BabyLlama}}
}}
```

## License

This model is released under the MIT License.
"""

    return model_card


def publish_model(
    model_path: str,
    repo_name: str,
    token: Optional[str] = None,
    private: bool = False,
    description: Optional[str] = None,
    commit_message: Optional[str] = None,
):
    """Publish a trained model to Hugging Face Hub"""

    model_path = Path(model_path)

    if not model_path.exists():
        raise ValueError(f"Model path {model_path} does not exist")

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

    # Test model loading
    print("Testing model loading...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✓ Model loaded successfully ({model.num_parameters():,} parameters)")
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")

    # Initialize Hugging Face API
    api = HfApi(token=token)

    # Create repository
    print(f"Creating repository: {repo_name}")
    try:
        # Try to create the repository
        repo_url = create_repo(
            repo_id=repo_name, token=token, private=private, exist_ok=True
        )
        print("✓ Repository created/verified")
    except Exception as e:
        print(f"Warning: Repository creation failed: {e}")

        # Check if repository already exists
        try:
            from huggingface_hub import repo_exists

            if repo_exists(repo_name, token=token):
                print("✓ Repository already exists, proceeding with upload")
            else:
                print("❌ Repository does not exist and cannot be created")
                print("Please check:")
                print("1. Your token has write permissions")
                print("2. The repository name is available")
                print("3. You have permission to create repos in this namespace")
                return None
        except Exception as check_error:
            print(f"Could not verify repository existence: {check_error}")
            print("Proceeding with upload attempt...")

    # Create model card
    print("Creating model card...")
    model_card_content = create_model_card(
        model_path, repo_name, training_config, training_metrics, description
    )

    # Save model card
    readme_path = model_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card_content)

    # Upload model
    print(f"Uploading model to {repo_name}...")
    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            token=token,
            commit_message=commit_message
            or f"Upload {training_config.get('model', {}).get('name', 'BabyLlama')} model",
        )

        print(f"✓ Model successfully published to: https://huggingface.co/{repo_name}")
        return f"https://huggingface.co/{repo_name}"

    except Exception as upload_error:
        print(f"❌ Upload failed: {upload_error}")
        print("\nTroubleshooting steps:")
        print("1. Check your token has write permissions")
        print("2. Try creating the repository manually on huggingface.co first")
        print("3. Verify the repository name is correct")
        print("4. Check if you have sufficient storage quota")
        raise upload_error


def main():
    parser = argparse.ArgumentParser(
        description="Publish BabyLlama model to Hugging Face Hub"
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the trained model directory"
    )
    parser.add_argument(
        "repo_name",
        type=str,
        help="Hugging Face repository name (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--token", type=str, help="Hugging Face API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private", action="store_true", help="Make repository private"
    )
    parser.add_argument("--description", type=str, help="Custom model description")
    parser.add_argument("--commit-message", type=str, help="Custom commit message")

    args = parser.parse_args()

    try:
        url = publish_model(
            model_path=args.model_path,
            repo_name=args.repo_name,
            token=args.token,
            private=args.private,
            description=args.description,
            commit_message=args.commit_message,
        )
        print(f"\n🎉 Success! Your model is now available at: {url}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
