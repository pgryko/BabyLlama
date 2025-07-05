#!/usr/bin/env python3
"""
Complete pipeline to train a 10M parameter LLaMA model and publish to Hugging Face
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    end_time = time.time()

    if result.returncode != 0:
        print(f"❌ Error: Command failed with return code {result.returncode}")
        sys.exit(1)

    print(f"✅ Completed in {end_time - start_time:.1f} seconds")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train and publish BabyLlama model")
    parser.add_argument(
        "--dataset-size",
        choices=["500k", "1M", "100M", "1B"],
        default="1B",
        help="Size of synthetic dataset to generate",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Hugging Face repository name (e.g., 'username/babyllama-10m')",
    )
    parser.add_argument(
        "--hf-token", type=str, help="Hugging Face API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private", action="store_true", help="Make Hugging Face repository private"
    )
    parser.add_argument(
        "--skip-data-generation", action="store_true", help="Skip data generation step"
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip training step"
    )
    parser.add_argument(
        "--skip-publishing", action="store_true", help="Skip publishing step"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )

    args = parser.parse_args()

    print("🦙 BabyLlama Training and Publishing Pipeline")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Repository: {args.repo_name}")
    print(f"Private repo: {args.private}")

    # Step 1: Generate synthetic data
    if not args.skip_data_generation:
        print("\n" + "=" * 60)
        print("📊 STEP 1: Generating Synthetic Data")
        print("=" * 60)

        data_cmd = ["uv", "run", "python", "create_synthetic_data.py"]

        if args.dataset_size == "1B":
            data_cmd.append("--large-scale")
        else:
            # Convert size to tokens
            size_map = {"500k": 500_000, "1M": 1_000_000, "100M": 100_000_000}
            data_cmd.extend(["--num-tokens", str(size_map[args.dataset_size])])

        run_command(data_cmd, "Generating synthetic training data")

        # Train tokenizer if needed
        tokenizer_path = Path("models/gpt-clean-16000.json")
        if not tokenizer_path.exists():
            run_command(
                ["uv", "run", "python", "train_tokenizer.py"], "Training tokenizer"
            )

    # Step 2: Train the model
    if not args.skip_training:
        print("\n" + "=" * 60)
        print("🏋️ STEP 2: Training Model")
        print("=" * 60)

        # Choose config based on dataset size
        if args.dataset_size in ["1B", "100M"]:
            config_file = "config/llama-10M-large.yaml"
        else:
            config_file = "config/llama-10M.yaml"

        train_cmd = ["uv", "run", "python", "train.py", "--config", config_file]

        # Update model name based on dataset size
        model_name = f"Llama-10M-{args.dataset_size}"
        train_cmd.extend(["--model_name", model_name])

        run_command(train_cmd, f"Training {model_name} model")

        # Store model path for publishing
        model_path = Path("models") / model_name

    # Step 3: Evaluate the model
    if not args.skip_training:
        print("\n" + "=" * 60)
        print("📈 STEP 3: Evaluating Model")
        print("=" * 60)

        eval_cmd = ["uv", "run", "python", "evaluate.py", str(model_path)]
        run_command(eval_cmd, "Evaluating trained model")

    # Step 4: Publish to Hugging Face
    if not args.skip_publishing:
        print("\n" + "=" * 60)
        print("🚀 STEP 4: Publishing to Hugging Face")
        print("=" * 60)

        if args.skip_training:
            # Find the most recent model
            models_dir = Path("models")
            model_dirs = [
                d
                for d in models_dir.iterdir()
                if d.is_dir() and d.name.startswith("Llama-10M")
            ]
            if not model_dirs:
                print("❌ No trained models found. Please train a model first.")
                sys.exit(1)
            model_path = max(model_dirs, key=lambda x: x.stat().st_mtime)
            print(f"Using model: {model_path}")

        publish_cmd = [
            "uv",
            "run",
            "python",
            "publish_to_hf.py",
            str(model_path),
            args.repo_name,
        ]

        if args.hf_token:
            publish_cmd.extend(["--token", args.hf_token])

        if args.private:
            publish_cmd.append("--private")

        # Add description
        description = f"A 10M parameter LLaMA model trained on {args.dataset_size} synthetic tokens using the BabyLlama framework."
        publish_cmd.extend(["--description", description])

        run_command(publish_cmd, "Publishing model to Hugging Face Hub")

    # Final summary
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 60)

    if not args.skip_training:
        print(f"✅ Model trained: {model_path}")

    if not args.skip_publishing:
        print(f"✅ Model published: https://huggingface.co/{args.repo_name}")

    print("\n📋 Next steps:")
    print("1. Test your model with the Hugging Face interface")
    print("2. Share your model with the community")
    print("3. Consider fine-tuning on domain-specific data")
    print("4. Experiment with different architectures and sizes")

    print(f"\n🔗 Your model: https://huggingface.co/{args.repo_name}")


if __name__ == "__main__":
    main()
