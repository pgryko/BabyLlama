#!/usr/bin/env python3
"""
BabyLlama - A modern framework for training small language models

This is the main entry point for the BabyLlama framework.
Use this script to access all major functionality through a unified CLI.
"""

import argparse
import sys
from pathlib import Path


def print_banner():
    """Print the BabyLlama banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                         🦙 BabyLlama                         ║
    ║              Modern Language Model Training Framework        ║
    ║                                                              ║
    ║  Train • Evaluate • Benchmark • Deploy                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def show_quick_start():
    """Show quick start instructions"""
    print("🚀 Quick Start:")
    print("  1. Generate synthetic data:    python create_synthetic_data.py")
    print("  2. Train tokenizer:           python train_tokenizer.py")
    print(
        "  3. Train model:               python train.py --config config/llama-10M.yaml"
    )
    print("  4. Evaluate model:            python evaluate.py models/Llama-10M/")
    print()
    print("📚 Documentation:")
    print("  • Training Guide:             TRAINING_GUIDE.md")
    print("  • API Reference:              API_REFERENCE.md")
    print("  • Contributing:               CONTRIBUTING.md")
    print()
    print("🔧 Available Commands:")
    print("  • python main.py train        Start training workflow")
    print("  • python main.py evaluate     Start evaluation workflow")
    print("  • python main.py benchmark    Run benchmarks")
    print("  • python main.py test         Run test suite")


def train_workflow():
    """Interactive training workflow"""
    print("🚂 Training Workflow")
    print("=" * 50)

    # Check if data exists
    data_dir = Path("data")
    if not data_dir.exists() or not list(data_dir.glob("*")):
        print("📊 No training data found. Let's set up data first:")
        print("  1. Synthetic data (quick):    python create_synthetic_data.py")
        print(
            "  2. BabyLM data (research):    python prepare_data.py --babylm-10m /path/to/data"
        )
        return

    # Check if tokenizer exists
    tokenizer_path = Path("models/gpt-clean-16000.json")
    if not tokenizer_path.exists():
        print("🔤 No tokenizer found. Training tokenizer:")
        print("  Run: python train_tokenizer.py")
        return

    # Show available configs
    config_dir = Path("config")
    configs = list(config_dir.glob("*.yaml"))
    print("⚙️ Available model configurations:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config.stem}")

    print("\n🚀 Start training:")
    print("  python train.py --config config/llama-10M.yaml")


def evaluate_workflow():
    """Interactive evaluation workflow"""
    print("📊 Evaluation Workflow")
    print("=" * 50)

    # Check for trained models
    models_dir = Path("models")
    model_dirs = [
        d for d in models_dir.iterdir() if d.is_dir() and (d / "config.json").exists()
    ]

    if not model_dirs:
        print("❌ No trained models found.")
        print("   Train a model first: python main.py train")
        return

    print("🤖 Available trained models:")
    for i, model_dir in enumerate(model_dirs, 1):
        print(f"  {i}. {model_dir.name}")

    print("\n📈 Evaluation options:")
    print("  • Quick evaluation:           python evaluate.py models/Llama-10M/")
    print(
        "  • Comprehensive evaluation:   python evaluate.py models/Llama-10M/ --num-samples 1000"
    )
    print("  • Benchmark testing:          python benchmark.py models/Llama-10M/")
    print("  • Model comparison:           python benchmark.py models/*/")


def benchmark_workflow():
    """Interactive benchmark workflow"""
    print("🏁 Benchmark Workflow")
    print("=" * 50)

    models_dir = Path("models")
    model_dirs = [
        d for d in models_dir.iterdir() if d.is_dir() and (d / "config.json").exists()
    ]

    if len(model_dirs) < 1:
        print("❌ No trained models found for benchmarking.")
        return

    print("🤖 Available models for benchmarking:")
    for i, model_dir in enumerate(model_dirs, 1):
        print(f"  {i}. {model_dir.name}")

    print("\n🏆 Benchmark options:")
    if len(model_dirs) == 1:
        print(f"  • Single model:               python benchmark.py {model_dirs[0]}/")
    else:
        print("  • Compare all models:         python benchmark.py models/*/")
        print(
            "  • Compare specific models:    python benchmark.py models/Model1/ models/Model2/"
        )


def test_workflow():
    """Run test suite"""
    print("🧪 Running Test Suite")
    print("=" * 50)

    import subprocess

    try:
        subprocess.run([sys.executable, "run_tests.py"], check=True)
        print("✅ All tests passed!")
    except subprocess.CalledProcessError:
        print("❌ Some tests failed. Check output above.")
    except FileNotFoundError:
        print("❌ Test runner not found. Run: pytest")


def main():
    """Main entry point for BabyLlama CLI"""
    parser = argparse.ArgumentParser(
        description="BabyLlama - Modern Language Model Training Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Show quick start guide
  python main.py train              # Interactive training workflow
  python main.py evaluate           # Interactive evaluation workflow
  python main.py benchmark          # Interactive benchmark workflow
  python main.py test               # Run test suite

For detailed documentation, see TRAINING_GUIDE.md
        """,
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "evaluate", "benchmark", "test"],
        help="Command to run (optional)",
    )

    parser.add_argument("--version", action="version", version="BabyLlama 0.1.0")

    args = parser.parse_args()

    print_banner()

    if args.command == "train":
        train_workflow()
    elif args.command == "evaluate":
        evaluate_workflow()
    elif args.command == "benchmark":
        benchmark_workflow()
    elif args.command == "test":
        test_workflow()
    else:
        show_quick_start()


if __name__ == "__main__":
    main()
