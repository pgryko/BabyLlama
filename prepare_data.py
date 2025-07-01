"""
Data preparation script for BabyLlama
Replaces the cleaning_and_tokenization.ipynb notebook
"""

import argparse
from pathlib import Path
from data_utils import create_cleaner_registry


def clean_babylm_data(input_dir: Path, output_dir: Path):
    """Clean BabyLM dataset files using domain-specific cleaners"""

    output_dir.mkdir(exist_ok=True, parents=True)
    cleaner_registry = create_cleaner_registry()

    # Map file prefixes to cleaner functions
    file_cleaners = {
        "aochildes": cleaner_registry.get("dialogue", lambda x: x),
        "bnc_spoken": cleaner_registry.get("dialogue", lambda x: x),
        "cbt": cleaner_registry.get("default", lambda x: x),
        "children_stories": cleaner_registry.get("default", lambda x: x),
        "gutenberg": cleaner_registry.get("default", lambda x: x),
        "open_subtitles": cleaner_registry.get("subtitles", lambda x: x),
        "qed": cleaner_registry.get("default", lambda x: x),
        "simple_wikipedia": cleaner_registry.get("wikipedia", lambda x: x),
        "switchboard": cleaner_registry.get("dialogue", lambda x: x),
        "wikipedia": cleaner_registry.get("wikipedia", lambda x: x),
    }

    # Process all train and dev files
    for file_path in input_dir.glob("*.train"):
        process_file(file_path, output_dir, file_cleaners)

    for file_path in input_dir.glob("*.dev"):
        process_file(file_path, output_dir, file_cleaners)


def process_file(file_path: Path, output_dir: Path, file_cleaners: dict):
    """Process a single file with appropriate cleaner"""

    # Get the appropriate cleaner
    file_prefix = file_path.stem.split(".")[0]
    cleaner = file_cleaners.get(file_prefix, lambda x: x)

    # Read and clean text
    text = file_path.read_text(encoding="utf-8")
    cleaned_text = cleaner(text)

    # Save cleaned text
    output_path = output_dir / file_path.name
    output_path.write_text(cleaned_text, encoding="utf-8")

    print(f"🧹 Cleaned '{file_path.name}' (size {len(text)} -> {len(cleaned_text)})")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for BabyLlama training")
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Base data directory"
    )
    parser.add_argument(
        "--babylm-10m", type=str, default=None, help="Path to raw BabyLM 10M dataset"
    )
    parser.add_argument(
        "--babylm-dev", type=str, default=None, help="Path to raw BabyLM dev dataset"
    )
    parser.add_argument(
        "--tokenizer-vocab", type=int, default=16000, help="Tokenizer vocabulary size"
    )
    parser.add_argument(
        "--tokenizer-output-path",
        type=str,
        default="./models/gpt-clean-16000.json",
        help="Path to save trained tokenizer",
    )
    parser.add_argument(
        "--skip-cleaning", action="store_true", help="Skip data cleaning step"
    )
    parser.add_argument(
        "--skip-tokenizer", action="store_true", help="Skip tokenizer training"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    # Step 1: Copy/clean BabyLM data if provided
    if args.babylm_10m and not args.skip_cleaning:
        print("\n=== Cleaning BabyLM 10M Dataset ===")
        input_dir = Path(args.babylm_10m)
        output_dir = data_dir / "babylm_10M_clean"
        clean_babylm_data(input_dir, output_dir)

    if args.babylm_dev and not args.skip_cleaning:
        print("\n=== Cleaning BabyLM Dev Dataset ===")
        input_dir = Path(args.babylm_dev)
        output_dir = data_dir / "babylm_dev_clean"
        clean_babylm_data(input_dir, output_dir)

    # Step 2: Train tokenizer
    if not args.skip_tokenizer:
        print("\n=== Training BPE Tokenizer ===")

        # Find training data
        train_dir = data_dir / "babylm_10M_clean"
        if not train_dir.exists():
            print("No BabyLM data found. Using synthetic data for tokenizer training.")
            # Generate synthetic data if needed
            from create_synthetic_data import create_synthetic_text

            train_text = create_synthetic_text(500_000)
            (train_dir / "synthetic.train").write_text(train_text)
            dev_text = create_synthetic_text(50_000)
            (data_dir / "babylm_dev_clean" / "synthetic.dev").write_text(dev_text)

        # Train tokenizer
        train_paths = list(train_dir.glob("*.train"))
        if train_paths:
            from train_tokenizer import train_bpe_tokenizer

            train_bpe_tokenizer(
                train_paths,
                vocab_size=args.tokenizer_vocab,
                output_path=args.tokenizer_output_path,
            )
            print("✅ Tokenizer training complete!")
        else:
            print("❌ No training files found for tokenizer training")

    print("\n=== Data Preparation Complete ===")
    print("You can now train a model with:")
    print("  python train.py --config ./config/llama-10M.yaml")


if __name__ == "__main__":
    main()
