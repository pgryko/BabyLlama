import random
from pathlib import Path


def create_synthetic_text(num_tokens=1_000_000):
    """Create simple synthetic text data for testing"""

    subjects = [
        "The cat",
        "A dog",
        "The bird",
        "A child",
        "The teacher",
        "A student",
        "The scientist",
        "An artist",
        "The engineer",
        "A writer",
    ]
    verbs = [
        "runs",
        "jumps",
        "walks",
        "thinks",
        "writes",
        "reads",
        "creates",
        "builds",
        "discovers",
        "learns",
        "teaches",
        "explores",
        "imagines",
        "designs",
    ]
    objects = [
        "quickly",
        "slowly",
        "carefully",
        "happily",
        "sadly",
        "eagerly",
        "thoughtfully",
        "creatively",
        "brilliantly",
        "patiently",
    ]
    endings = [
        "in the park",
        "at home",
        "in the school",
        "at the library",
        "in the lab",
        "on the street",
        "in the garden",
        "at the office",
        "in the classroom",
        "outside",
    ]
    connectors = [
        "and",
        "but",
        "however",
        "therefore",
        "moreover",
        "furthermore",
        "additionally",
        "consequently",
        "meanwhile",
        "afterwards",
    ]

    text_parts = []
    current_tokens = 0

    while current_tokens < num_tokens:
        paragraph = []
        num_sentences = random.randint(3, 8)

        for _ in range(num_sentences):
            sentence = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)} {random.choice(endings)}."
            if random.random() > 0.5:
                connector = random.choice(connectors)
                sentence = f"{sentence[:-1]}, {connector} {random.choice(subjects).lower()} {random.choice(verbs)} {random.choice(objects)}."
            paragraph.append(sentence)

        paragraph_text = " ".join(paragraph) + "\n\n"
        text_parts.append(paragraph_text)

        # Rough estimate of tokens (assuming ~1.3 tokens per word)
        words = paragraph_text.split()
        current_tokens += int(len(words) * 1.3)

    return "".join(text_parts)


def generate_and_save_data(output_dir: Path, filename: str, num_tokens: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    text = create_synthetic_text(num_tokens)
    with open(output_dir / filename, "w") as f:
        f.write(text)
    print(f"Generated {filename}: {len(text)} characters")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=500_000,
        help="Number of tokens to generate for training data",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Base output directory"
    )
    parser.add_argument(
        "--large-scale",
        action="store_true",
        help="Generate large-scale dataset (1B tokens)",
    )

    args = parser.parse_args()

    base_data_dir = Path(args.output_dir)

    # Determine token counts
    if args.large_scale:
        train_tokens = 1_000_000_000  # 1B tokens
        dev_tokens = 10_000_000  # 10M tokens for dev
        output_suffix = "_1B"
    else:
        train_tokens = args.num_tokens
        dev_tokens = max(
            50_000, train_tokens // 10
        )  # 10% of training data or 50k minimum
        output_suffix = (
            f"_{train_tokens//1000}k"
            if train_tokens < 1_000_000
            else f"_{train_tokens//1_000_000}M"
        )

    print(f"Generating synthetic training data ({train_tokens:,} tokens)...")
    generate_and_save_data(
        base_data_dir / f"babylm_10M_clean{output_suffix}",
        "synthetic.train",
        train_tokens,
    )

    print(f"Generating synthetic dev data ({dev_tokens:,} tokens)...")
    generate_and_save_data(
        base_data_dir / f"babylm_dev_clean{output_suffix}", "synthetic.dev", dev_tokens
    )

    print("\nData generation complete!")
    print(f"Training data: {train_tokens:,} tokens")
    print(f"Dev data: {dev_tokens:,} tokens")
    print("Output directories:")
    print(f"  - {base_data_dir / f'babylm_10M_clean{output_suffix}'}")
    print(f"  - {base_data_dir / f'babylm_dev_clean{output_suffix}'}")


if __name__ == "__main__":
    main()
