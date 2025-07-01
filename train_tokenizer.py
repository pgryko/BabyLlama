from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFKC
from typing import List, Union
import argparse

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>"]


def train_bpe_tokenizer(
    paths: List[Union[str, Path]],
    vocab_size: int = 16000,
    output_path: str = "models/gpt-clean-16000.json",
) -> Tokenizer:
    """Train a BPE tokenizer on the given files"""

    paths = [str(p) for p in paths]

    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS
    )

    print(f"Training tokenizer on {len(paths)} files...")
    tokenizer.train(paths, trainer)

    tokenizer_path = Path(output_path)
    tokenizer_path.parent.mkdir(exist_ok=True)
    tokenizer.save(str(tokenizer_path), pretty=True)
    print(f"Tokenizer saved to: {tokenizer_path}")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/babylm_10M_clean/",
        help="Path to the directory containing training data",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=16000, help="Tokenizer vocabulary size"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/gpt-clean-16000.json",
        help="Path to save the trained tokenizer",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    paths = [str(f) for f in data_dir.glob("*.train") if f.is_file()]

    print(f"Found {len(paths)} training files")
    for p in paths:
        print(f"  - {p}")

    assert len(paths) > 0, "No data files found"

    tokenizer = train_bpe_tokenizer(
        paths, vocab_size=args.vocab_size, output_path=args.output_path
    )

    print("\nTesting tokenizer:")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Hello, how are you today?",
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"\nOriginal: {text}")
        print(f"Tokens: {encoded.tokens}")
        print(f"Decoded: {decoded}")
        print(f"Num tokens: {len(encoded.tokens)}")


if __name__ == "__main__":
    main()
