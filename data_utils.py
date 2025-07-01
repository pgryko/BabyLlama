"""
Modern data utilities for BabyLlama using HuggingFace datasets
"""

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import re
from pathlib import Path
from typing import Dict, List, Callable


class DataProcessor:
    """Modern data processor using HuggingFace datasets"""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def load_text_files(self, data_dir: str, split: str = "train") -> Dataset:
        """Load text files from directory into HF Dataset"""
        data_path = Path(data_dir)
        texts = []

        # Load all text files (sorted for consistent ordering)
        for file_path in sorted(data_path.glob(f"*.{split}")):
            text = file_path.read_text(encoding="utf-8")
            texts.append({"text": text, "source": file_path.stem})

        return Dataset.from_list(texts)

    def clean_text(self, example: Dict[str, str]) -> Dict[str, str]:
        """Basic text cleaning"""
        text = example["text"]

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Remove space before certain punctuation
        text = re.sub(r"\s+([.,;])", r"\1", text)  # Always remove before .,;

        # Handle ! and ? specially
        text = re.sub(r"\s+([!?])$", r"\1", text)  # Remove before !? if at end
        text = re.sub(r"([!?])\s+([!?])", r"\1\2", text)  # Remove space between ! and ?

        # Ensure space after punctuation (when followed by letters)
        text = re.sub(r"([.,;!?])([A-Za-z])", r"\1 \2", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        example["text"] = text
        return example

    def tokenize_and_chunk(
        self, example: Dict[str, str], max_length: int = 128
    ) -> Dict[str, List[int]]:
        """Tokenize and chunk text into sequences"""
        # Tokenize
        tokens = self.tokenizer(
            example["text"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]

        # Chunk into sequences
        chunks = []
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i : i + max_length]
            if len(chunk) == max_length:  # Only keep full chunks
                chunks.append(chunk)

        return {"input_ids": chunks}

    def prepare_dataset(
        self,
        train_data_dir: str,
        eval_data_dir: str,
        max_length: int = 128,
        clean: bool = True,
        num_proc: int = 4,
    ) -> DatasetDict:
        """Prepare dataset for training"""

        # Load train and validation data
        train_dataset = self.load_text_files(train_data_dir, "train")
        val_dataset = self.load_text_files(eval_data_dir, "dev")

        # Clean if requested
        if clean:
            train_dataset = train_dataset.map(self.clean_text, num_proc=num_proc)
            val_dataset = val_dataset.map(self.clean_text, num_proc=num_proc)

        # Tokenize and chunk
        def tokenize_and_expand(examples):
            """Tokenize and expand chunks into separate examples"""
            all_chunks = []
            for example in examples:
                result = self.tokenize_and_chunk(example, max_length)
                chunks = result["input_ids"]
                for chunk in chunks:
                    all_chunks.append({"input_ids": chunk})
            return all_chunks

        # Process train dataset
        train_examples = []
        for example in train_dataset:
            result = self.tokenize_and_chunk(example, max_length)
            chunks = result["input_ids"]
            for chunk in chunks:
                train_examples.append({"input_ids": chunk})

        # Process validation dataset
        val_examples = []
        for example in val_dataset:
            result = self.tokenize_and_chunk(example, max_length)
            chunks = result["input_ids"]
            for chunk in chunks:
                val_examples.append({"input_ids": chunk})

        # Create new datasets from the flattened examples
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)

        return DatasetDict({"train": train_dataset, "validation": val_dataset})


# Domain-specific cleaners (optional, for backwards compatibility)
class DomainCleaners:
    """Collection of domain-specific text cleaners"""

    @staticmethod
    def wikipedia(text: str) -> str:
        """Clean Wikipedia text"""
        # Remove headers
        text = re.sub(r"={2,}(.+?)={2,}", r"\1", text)
        # Remove citations
        text = re.sub(r"\[\d+\]", "", text)
        # Remove extra whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def subtitles(text: str) -> str:
        """Clean subtitle text"""
        # Remove subtitle credits (only lines that start with "Subtitles by")
        text = re.sub(r"^Subtitles by.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
        # Remove timing markers
        text = re.sub(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", text)
        # Remove subtitle numbers (standalone numbers on their own line)
        text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
        # Remove empty lines and clean up whitespace
        text = re.sub(r"\n\s*\n", "\n", text)
        text = re.sub(r"^\s*\n", "", text)
        return text.strip()

    @staticmethod
    def dialogue(text: str) -> str:
        """Clean dialogue/conversation text"""
        # Remove stage directions first
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)
        # Remove speaker labels (at beginning of lines or after whitespace)
        text = re.sub(r"(^|\s+)[A-Z][A-Z0-9]*:\s*", r"\1", text, flags=re.MULTILINE)
        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        return text.strip()


def create_cleaner_registry() -> Callable:
    """Create a registry of cleaners by domain"""
    return {
        "wikipedia": DomainCleaners.wikipedia,
        "simple_wikipedia": DomainCleaners.wikipedia,
        "subtitles": DomainCleaners.subtitles,
        "open_subtitles": DomainCleaners.subtitles,
        "dialogue": DomainCleaners.dialogue,
        "switchboard": DomainCleaners.dialogue,
        "default": lambda x: x,  # No cleaning
    }


# Example usage
if __name__ == "__main__":
    from transformers import GPT2TokenizerFast

    # Load tokenizer
    tokenizer = GPT2TokenizerFast(tokenizer_file="./models/gpt-clean-16000.json")
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"

    # Create processor
    processor = DataProcessor(tokenizer)

    # Prepare dataset
    dataset = processor.prepare_dataset(
        train_data_dir="./data/babylm_10M_clean",
        eval_data_dir="./data/babylm_dev_clean",
        max_length=128,
        clean=True,
    )

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Sample: {dataset['train'][0]}")
