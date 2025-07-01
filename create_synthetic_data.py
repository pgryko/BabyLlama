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
    base_data_dir = Path("data")

    print("Generating synthetic training data...")
    generate_and_save_data(
        base_data_dir / "babylm_10M_clean", "synthetic.train", 500_000
    )

    print("Generating synthetic dev data...")
    generate_and_save_data(base_data_dir / "babylm_dev_clean", "synthetic.dev", 50_000)

    print("\nData generation complete!")


if __name__ == "__main__":
    main()
