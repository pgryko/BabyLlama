"""
Benchmark evaluation script for BabyLlama models
Provides standardized benchmarks for model comparison
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from tqdm import tqdm
from evaluate import ModelEvaluator


class BenchmarkSuite:
    """Collection of benchmark tasks for language models"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.evaluator = ModelEvaluator(model_path, device)
        self.results = {}
        with open("benchmark_data.json", "r") as f:
            self.benchmark_data = json.load(f)

    def run_completion_benchmark(self) -> Dict[str, float]:
        """Test model's ability to complete common phrases"""

        test_prompts = self.benchmark_data["completion_prompts"]
        correct = 0
        total = len(test_prompts)

        for item in tqdm(test_prompts, desc="Completion benchmark"):
            prompt = item["prompt"]
            expected_words = item["expected"]
            inputs = self.evaluator.tokenizer(prompt, return_tensors="pt").to(
                self.evaluator.device
            )

            with torch.no_grad():
                outputs = self.evaluator.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.evaluator.tokenizer.pad_token_id,
                )

            completion = self.evaluator.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            completion = completion[len(prompt) :].strip().lower()

            if any(word.lower() in completion for word in expected_words):
                correct += 1

        accuracy = correct / total
        return {
            "completion_accuracy": accuracy,
            "completion_correct": correct,
            "completion_total": total,
        }

    def run_consistency_benchmark(self) -> Dict[str, float]:
        """Test model's consistency across similar prompts"""

        consistency_groups = self.benchmark_data["consistency_groups"]
        consistency_scores = []

        for group in tqdm(consistency_groups, desc="Consistency benchmark"):
            group_outputs = []

            for prompt in group:
                inputs = self.evaluator.tokenizer(prompt, return_tensors="pt").to(
                    self.evaluator.device
                )

                with torch.no_grad():
                    outputs = self.evaluator.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.5,
                        do_sample=True,
                        pad_token_id=self.evaluator.tokenizer.pad_token_id,
                    )

                completion = self.evaluator.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                completion = completion[len(prompt) :].strip()
                group_outputs.append(completion)

            all_words = [set(output.split()) for output in group_outputs]
            if all_words:
                common_words = set.intersection(*all_words)
                total_unique_words = len(set.union(*all_words))
                consistency = (
                    len(common_words) / total_unique_words
                    if total_unique_words > 0
                    else 0
                )
                consistency_scores.append(consistency)

        return {
            "consistency_score": np.mean(consistency_scores),
            "consistency_std": np.std(consistency_scores),
        }

    def run_repetition_benchmark(self) -> Dict[str, float]:
        """Test model's tendency to repeat"""

        prompts = self.benchmark_data["repetition_prompts"]
        repetition_scores = []

        for prompt in tqdm(prompts, desc="Repetition benchmark"):
            inputs = self.evaluator.tokenizer(prompt, return_tensors="pt").to(
                self.evaluator.device
            )

            with torch.no_grad():
                outputs = self.evaluator.model.generate(
                    **inputs,
                    max_length=100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.evaluator.tokenizer.pad_token_id,
                )

            text = self.evaluator.tokenizer.decode(outputs[0], skip_special_tokens=True)
            text = text[len(prompt) :].strip()

            rep_score = self.evaluator._calculate_repetition([text])
            repetition_scores.append(rep_score)

        return {
            "repetition_score": np.mean(repetition_scores),
            "repetition_std": np.std(repetition_scores),
        }

    def run_speed_benchmark(
        self, batch_sizes: List[int] = [1, 4, 8, 16]
    ) -> Dict[str, float]:
        """Benchmark inference speed"""

        prompt = "The quick brown fox jumps over the lazy dog. " * 5
        results = {}

        for batch_size in batch_sizes:
            prompts = [prompt] * batch_size

            inputs = self.evaluator.tokenizer(
                prompts[:1], return_tensors="pt", padding=True
            ).to(self.evaluator.device)
            with torch.no_grad():
                _ = self.evaluator.model.generate(**inputs, max_new_tokens=50)

            inputs = self.evaluator.tokenizer(
                prompts, return_tensors="pt", padding=True
            ).to(self.evaluator.device)

            start_time = time.time()
            with torch.no_grad():
                self.evaluator.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.evaluator.tokenizer.pad_token_id,
                )
            end_time = time.time()

            total_time = end_time - start_time
            tokens_per_second = (50 * batch_size) / total_time

            results[f"tokens_per_second_batch_{batch_size}"] = tokens_per_second
            results[f"latency_batch_{batch_size}"] = total_time

        return results

    def run_all_benchmarks(self) -> Dict[str, any]:
        """Run all benchmarks and compile results"""

        print("\n=== Running BabyLlama Benchmark Suite ===")
        print(f"Model: {self.evaluator.model_path}")

        results = {
            "model_path": str(self.evaluator.model_path),
            "model_parameters": sum(
                p.numel() for p in self.evaluator.model.parameters()
            ),
            "device": self.evaluator.device,
        }

        print("\n1. Completion Benchmark...")
        completion_results = self.run_completion_benchmark()
        results.update(completion_results)
        print(f"   Accuracy: {completion_results['completion_accuracy']:.2%}")

        print("\n2. Consistency Benchmark...")
        consistency_results = self.run_consistency_benchmark()
        results.update(consistency_results)
        print(f"   Consistency: {consistency_results['consistency_score']:.3f}")

        print("\n3. Repetition Benchmark...")
        repetition_results = self.run_repetition_benchmark()
        results.update(repetition_results)
        print(f"   Repetition: {repetition_results['repetition_score']:.3f}")

        print("\n4. Speed Benchmark...")
        speed_results = self.run_speed_benchmark()
        results.update(speed_results)
        print(
            f"   Tokens/sec (batch=1): {speed_results['tokens_per_second_batch_1']:.1f}"
        )

        overall_score = self._calculate_overall_score(results)
        results["overall_score"] = overall_score

        return results

    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate a single overall quality score"""

        weights = {
            "completion": 0.3,
            "consistency": 0.2,
            "repetition": 0.2,  # Lower is better
            "speed": 0.3,
        }

        scores = {
            "completion": results.get("completion_accuracy", 0),
            "consistency": results.get("consistency_score", 0),
            "repetition": 1 - min(results.get("repetition_score", 0), 1),
            "speed": min(results.get("tokens_per_second_batch_1", 0) / 100, 1),
        }

        overall = sum(scores[k] * weights[k] for k in weights)
        return overall


def compare_models(
    model_paths: List[str], output_path: str = "benchmark_comparison.json"
):
    """Compare multiple models using the benchmark suite"""

    all_results = []

    for model_path in model_paths:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_path}")
        print(f"{'='*60}")

        suite = BenchmarkSuite(model_path)
        results = suite.run_all_benchmarks()
        all_results.append(results)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*60}")

    print(
        f"\n{'Model':<30} {'Overall':<10} {'Compl.':<10} {'Consist.':<10} {'Repet.':<10} {'Speed':<10}"
    )
    print("-" * 80)

    for result in all_results:
        model_name = Path(result["model_path"]).name
        print(
            f"{model_name:<30} "
            f"{result['overall_score']:<10.3f} "
            f"{result['completion_accuracy']:<10.2%} "
            f"{result['consistency_score']:<10.3f} "
            f"{result['repetition_score']:<10.3f} "
            f"{result.get('tokens_per_second_batch_1', 0):<10.1f}"
        )

    print(f"\nDetailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark BabyLlama models")
    parser.add_argument("model_paths", nargs="+", help="Paths to model directories")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    if len(args.model_paths) == 1:
        suite = BenchmarkSuite(args.model_paths[0], args.device)
        results = suite.run_all_benchmarks()

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n\nOverall Score: {results['overall_score']:.3f}")
        print(f"Results saved to: {args.output}")
    else:
        compare_models(args.model_paths, args.output)


if __name__ == "__main__":
    main()
