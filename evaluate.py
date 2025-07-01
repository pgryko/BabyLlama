"""
Model evaluation utilities for BabyLlama
Provides comprehensive metrics for model quality assessment
"""

import torch
import numpy as np
from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    GPT2LMHeadModel,
    GPTJForCausalLM,
)
from tqdm import tqdm
from pathlib import Path
import json
import argparse
from typing import Dict, List
from collections import defaultdict
import matplotlib.pyplot as plt

# Import our data utilities
from data_utils import DataProcessor


class ModelEvaluator:
    """Comprehensive model evaluation class"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device

        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self):
        """Load model based on config"""
        config_path = self.model_path / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        model_type = config.get("model_type", "llama")

        if model_type == "llama":
            return LlamaForCausalLM.from_pretrained(str(self.model_path))
        elif model_type == "gpt2":
            return GPT2LMHeadModel.from_pretrained(str(self.model_path))
        elif model_type == "gptj":
            return GPTJForCausalLM.from_pretrained(str(self.model_path))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def calculate_perplexity(
        self, texts: List[str], batch_size: int = 8
    ) -> Dict[str, float]:
        """Calculate perplexity on a list of texts"""
        total_loss = 0
        total_tokens = 0
        losses = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), batch_size), desc="Calculating perplexity"
            ):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                # Move to device (handle both dict and BatchEncoding)
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self.device)
                else:
                    inputs = {
                        k: v.to(self.device) if hasattr(v, "to") else v
                        for k, v in inputs.items()
                    }

                # Forward pass
                outputs = self.model(**inputs, labels=inputs["input_ids"])

                # Calculate loss for each sequence
                actual_batch_size = inputs["input_ids"].shape[0]
                for j in range(actual_batch_size):
                    seq_len = (
                        (inputs["attention_mask"][j] == 1).sum().item()
                    )  # Convert to Python int
                    seq_loss = outputs.loss.item()
                    losses.append(seq_loss)
                    total_loss += seq_loss * seq_len
                    total_tokens += seq_len

        # Calculate metrics (protect against division by zero)
        if total_tokens == 0:
            return {
                "perplexity": float("inf"),
                "average_loss": float("inf"),
                "std_loss": 0.0,
                "min_loss": 0.0,
                "max_loss": 0.0,
                "num_sequences": len(texts),
                "total_tokens": 0,
            }

        perplexity = np.exp(total_loss / total_tokens)
        avg_loss = total_loss / total_tokens

        return {
            "perplexity": perplexity,
            "average_loss": avg_loss,
            "std_loss": np.std(losses),
            "min_loss": np.min(losses),
            "max_loss": np.max(losses),
            "num_sequences": len(texts),
            "total_tokens": int(total_tokens),
        }

    def evaluate_generation_quality(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_return_sequences: int = 3,
    ) -> Dict[str, any]:
        """Evaluate generation quality metrics"""

        generations = []
        diversity_scores = []
        repetition_scores = []

        for prompt in tqdm(prompts, desc="Generating samples"):
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # Move to device (handle both dict and BatchEncoding)
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            else:
                inputs = {
                    k: v.to(self.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

            # Generate multiple sequences
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode generations
            batch_generations = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remove prompt from generation
                text = text[len(prompt) :].strip()
                batch_generations.append(text)
                generations.append(text)

            # Calculate diversity (unique n-grams)
            diversity = self._calculate_diversity(batch_generations)
            diversity_scores.append(diversity)

            # Calculate repetition
            repetition = self._calculate_repetition(batch_generations)
            repetition_scores.append(repetition)

        return {
            "avg_diversity_score": np.mean(diversity_scores),
            "avg_repetition_score": np.mean(repetition_scores),
            "generation_samples": generations[:10],  # Store first 10 samples
            "diversity_scores": diversity_scores,
            "repetition_scores": repetition_scores,
        }

    def _calculate_diversity(self, texts: List[str], n: int = 3) -> float:
        """Calculate diversity as ratio of unique n-grams"""
        all_ngrams = []

        for text in texts:
            tokens = text.split()
            ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)

        if not all_ngrams:
            return 0.0

        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)

        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0

    def _calculate_repetition(self, texts: List[str]) -> float:
        """Calculate repetition score based on repeated words (lower is better)"""
        repetition_scores = []

        for text in texts:
            # Convert to lowercase for case-insensitive comparison
            tokens = text.lower().split()
            if len(tokens) < 2:
                continue

            # Count word frequencies
            word_counts = defaultdict(int)
            for token in tokens:
                word_counts[token] += 1

            # Calculate repetition ratio - count all instances of repeated words
            total_repeated_instances = sum(
                count for count in word_counts.values() if count > 1
            )
            repetition_ratio = total_repeated_instances / len(tokens) if tokens else 0
            repetition_scores.append(repetition_ratio)

        return np.mean(repetition_scores) if repetition_scores else 0.0

    def evaluate_token_probabilities(self, texts: List[str]) -> Dict[str, float]:
        """Evaluate token probability statistics"""
        all_probs = []
        all_entropies = []

        with torch.no_grad():
            for text in tqdm(texts, desc="Analyzing token probabilities"):
                inputs = self.tokenizer(text, return_tensors="pt")
                # Move to device (handle both dict and BatchEncoding)
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self.device)
                else:
                    inputs = {
                        k: v.to(self.device) if hasattr(v, "to") else v
                        for k, v in inputs.items()
                    }
                outputs = self.model(**inputs)

                # Get probabilities
                probs = torch.softmax(outputs.logits[0], dim=-1)

                # Calculate entropy for each position
                entropies = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

                # Get top token probabilities
                top_probs, _ = torch.max(probs, dim=-1)

                all_probs.extend(top_probs.cpu().numpy().tolist())
                all_entropies.extend(entropies.cpu().numpy().tolist())

        return {
            "avg_top_token_prob": np.mean(all_probs),
            "std_top_token_prob": np.std(all_probs),
            "avg_entropy": np.mean(all_entropies),
            "std_entropy": np.std(all_entropies),
            "low_confidence_ratio": np.mean([p < 0.5 for p in all_probs]),
        }

    def plot_metrics(
        self, metrics: Dict[str, any], save_path: str = "evaluation_plots.png"
    ):
        """Create visualization of evaluation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Model Evaluation: {self.model_path.name}", fontsize=16)

        # Plot 1: Loss distribution
        if "diversity_scores" in metrics:
            axes[0, 0].hist(
                metrics["diversity_scores"], bins=20, alpha=0.7, color="blue"
            )
            axes[0, 0].set_title("Generation Diversity Distribution")
            axes[0, 0].set_xlabel("Diversity Score")
            axes[0, 0].set_ylabel("Frequency")

        # Plot 2: Repetition scores
        if "repetition_scores" in metrics:
            axes[0, 1].hist(
                metrics["repetition_scores"], bins=20, alpha=0.7, color="red"
            )
            axes[0, 1].set_title("Repetition Score Distribution")
            axes[0, 1].set_xlabel("Repetition Score")
            axes[0, 1].set_ylabel("Frequency")

        # Plot 3: Summary metrics
        summary_metrics = {
            "Perplexity": metrics.get("perplexity", 0),
            "Avg Loss": metrics.get("average_loss", 0),
            "Diversity": metrics.get("avg_diversity_score", 0),
            "Repetition": metrics.get("avg_repetition_score", 0),
        }

        axes[1, 0].bar(
            summary_metrics.keys(),
            summary_metrics.values(),
            color=["green", "orange", "blue", "red"],
        )
        axes[1, 0].set_title("Summary Metrics")
        axes[1, 0].set_ylabel("Score")

        # Plot 4: Sample generations
        if "generation_samples" in metrics and metrics["generation_samples"]:
            sample_text = "\n\n".join(metrics["generation_samples"][:3])
            axes[1, 1].text(
                0.1,
                0.5,
                sample_text[:200] + "...",
                fontsize=10,
                wrap=True,
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Sample Generations")
            axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Evaluation plots saved to: {save_path}")


def evaluate_model(
    model_path: str,
    eval_data_path: str = None,
    num_samples: int = 100,
    output_path: str = None,
) -> Dict[str, any]:
    """Main evaluation function"""

    evaluator = ModelEvaluator(model_path)
    results = {}

    # Load evaluation data
    if eval_data_path:
        with open(eval_data_path) as f:
            eval_texts = [line.strip() for line in f][:num_samples]
    else:
        print("No evaluation data provided. Using BabyLM dev set.")
        data_processor = DataProcessor(evaluator.tokenizer)
        dev_dataset = data_processor.load_text_files("./data/babylm_dev_clean", "dev")
        eval_texts = [item["text"] for item in dev_dataset.select(range(num_samples))]

    print("\n=== Model Evaluation Report ===")
    print(f"Model: {model_path}")
    print(f"Device: {evaluator.device}")
    print(
        f"Number of parameters: {sum(p.numel() for p in evaluator.model.parameters()):,}"
    )

    # 1. Calculate perplexity
    print("\n1. Calculating Perplexity...")
    perplexity_metrics = evaluator.calculate_perplexity(eval_texts)
    results.update(perplexity_metrics)
    print(f"   Perplexity: {perplexity_metrics['perplexity']:.2f}")
    print(f"   Average Loss: {perplexity_metrics['average_loss']:.4f}")

    # 2. Evaluate generation quality
    print("\n2. Evaluating Generation Quality...")
    generation_prompts = eval_texts[:20]  # Use subset for generation
    generation_metrics = evaluator.evaluate_generation_quality(generation_prompts)
    results.update(generation_metrics)
    print(f"   Average Diversity: {generation_metrics['avg_diversity_score']:.3f}")
    print(f"   Average Repetition: {generation_metrics['avg_repetition_score']:.3f}")

    # 3. Analyze token probabilities
    print("\n3. Analyzing Token Probabilities...")
    prob_metrics = evaluator.evaluate_token_probabilities(eval_texts[:50])
    results.update(prob_metrics)
    print(f"   Average Top Token Probability: {prob_metrics['avg_top_token_prob']:.3f}")
    print(f"   Average Entropy: {prob_metrics['avg_entropy']:.3f}")

    # 4. Create visualizations
    print("\n4. Creating Visualizations...")
    plot_path = Path(model_path) / "evaluation_plots.png"
    evaluator.plot_metrics(results, str(plot_path))

    # 5. Save detailed results
    if output_path:
        output_file = Path(output_path)
    else:
        output_file = Path(model_path) / "evaluation_results.json"

    # Prepare results for JSON serialization
    json_results = {
        k: v
        for k, v in results.items()
        if not isinstance(v, (list, np.ndarray)) or k == "generation_samples"
    }

    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Average Loss: {results['average_loss']:.4f}")
    print(f"Diversity Score: {results['avg_diversity_score']:.3f}")
    print(f"Repetition Score: {results['avg_repetition_score']:.3f} (lower is better)")
    print(f"Top Token Confidence: {results['avg_top_token_prob']:.3f}")
    print(f"Average Entropy: {results['avg_entropy']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate BabyLlama models")
    parser.add_argument("model_path", type=str, help="Path to model directory")
    parser.add_argument(
        "--eval-data", type=str, default=None, help="Path to evaluation data file"
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output path for results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluate_model(args.model_path, args.eval_data, args.num_samples, args.output)


if __name__ == "__main__":
    main()
