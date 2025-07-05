"""
Modern training script with HuggingFace Trainer and datasets
"""

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    GPTJConfig,
    GPTJForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPT2TokenizerFast,
    TrainerCallback,
)
import torch
import numpy as np
from pathlib import Path
import yaml
import argparse
import json
from random import sample

from data_utils import DataProcessor


def load_config(config_path: str, args) -> dict:
    """Load and update config from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override with command line arguments
    if args.lr:
        config["training"]["lr"] = args.lr
    if args.model_name:
        config["model"]["name"] = args.model_name

    return config


def create_model(config: dict, tokenizer) -> torch.nn.Module:
    """Create model based on config"""
    model_type = config["model"]["type"]

    if model_type == "Llama":
        model_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=config["data"]["seq_length"],
            hidden_size=config["model"]["hidden_size"],
            intermediate_size=config["model"]["intermediate_size"],
            num_hidden_layers=config["model"]["n_layer"],
            num_attention_heads=config["model"]["n_head"],
            tie_word_embeddings=config["model"].get("tie_word_embeddings", False),
            pad_token_id=tokenizer.pad_token_id,
        )
        model = LlamaForCausalLM(model_config)

    elif model_type == "GPT2":
        model_config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=config["data"]["seq_length"],
            n_embd=config["model"]["hidden_size"],
            n_layer=config["model"]["n_layer"],
            n_head=config["model"]["n_head"],
            resid_pdrop=config["model"].get("resid_pdrop", 0.1),
            embd_pdrop=config["model"].get("embd_pdrop", 0.1),
            attn_pdrop=config["model"].get("attn_pdrop", 0.1),
            pad_token_id=tokenizer.pad_token_id,
        )
        model = GPT2LMHeadModel(model_config)

    elif model_type == "GPTJ":
        model_config = GPTJConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=config["data"]["seq_length"],
            n_embd=config["model"]["hidden_size"],
            n_layer=config["model"]["n_layer"],
            n_head=config["model"]["n_head"],
            resid_pdrop=config["model"].get("resid_pdrop", 0.1),
            embd_pdrop=config["model"].get("embd_pdrop", 0.1),
            attn_pdrop=config["model"].get("attn_pdrop", 0.1),
            tie_word_embeddings=config["model"].get("tie_word_embeddings", True),
            pad_token_id=tokenizer.pad_token_id,
        )
        model = GPTJForCausalLM(model_config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def prepare_datasets_modern(config: dict, tokenizer):
    """Prepare datasets using modern approach"""
    processor = DataProcessor(tokenizer)

    # Prepare datasets (caching disabled for now)
    datasets = processor.prepare_dataset(
        train_data_dir=config["data"]["train_path"],
        eval_data_dir=config["data"]["eval_path"],
        max_length=config["data"]["seq_length"],
        clean=True,
        num_proc=4,
    )

    # Sample validation set if needed
    if len(datasets["validation"]) > config["data"]["eval_samples"]:
        indices = sample(
            range(len(datasets["validation"])), config["data"]["eval_samples"]
        )
        datasets["validation"] = datasets["validation"].select(indices)

    return datasets["train"], datasets["validation"]


class PerplexityCallback(TrainerCallback):
    """Calculate and log perplexity during training"""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            perplexity = np.exp(metrics["eval_loss"])
            metrics["eval_perplexity"] = perplexity
            print(f"\nEpoch {state.epoch:.1f} - Perplexity: {perplexity:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config/llama-10M.yaml",
        help="Configuration file path",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, args)

    # Setup tokenizer
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(config["data"]["tokenizer_path"]))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"

    # Prepare datasets
    print("Loading and processing datasets...")
    train_dataset, eval_dataset = prepare_datasets_modern(config, tokenizer)

    # Set tokenizer max length
    tokenizer.model_max_length = config["data"]["seq_length"]

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Create model
    model = create_model(config, tokenizer)
    print(f"Model parameters: {model.num_parameters():,}")

    # Setup training arguments
    output_dir = Path(config["logging"]["output_dir"]) / config["model"]["name"]
    accumulation_steps = config["training"]["gradient_accumulation_steps"]
    per_device_bsz = config["training"]["batch_size"] // accumulation_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        num_train_epochs=config["training"]["num_epochs"],
        gradient_accumulation_steps=accumulation_steps,
        per_device_train_batch_size=per_device_bsz,
        per_device_eval_batch_size=per_device_bsz,
        save_total_limit=1,
        warmup_steps=config["training"]["warmup_steps"],
        lr_scheduler_type="cosine",
        learning_rate=float(config["training"]["lr"]),
        logging_steps=20,
        fp16=config["training"].get("fp16", True),
        bf16=config["training"].get("bf16", False),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        torch_compile=config["training"].get("torch_compile", False),
        dataloader_num_workers=4,
        report_to="wandb" if config["logging"].get("wandb", False) else "none",
        run_name=(
            config["model"]["name"] if config["logging"].get("wandb", False) else None
        ),
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[PerplexityCallback()],
    )

    # Setup wandb if requested
    if config["logging"].get("wandb", False):
        import wandb

        wandb.init(
            project=config["logging"]["project"],
            name=config["model"]["name"],
            config=config,
        )

    # Train
    train_result = trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config for reference
    with open(output_dir / "training_config.yaml", "w") as f:
        yaml.dump(config, f)

    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    metrics["num_parameters"] = model.num_parameters()
    metrics["model_config"] = {
        "hidden_size": config["model"]["hidden_size"],
        "n_layer": config["model"]["n_layer"],
        "n_head": config["model"]["n_head"],
        "vocab_size": tokenizer.vocab_size,
        "seq_length": config["data"]["seq_length"],
    }

    # Calculate final perplexity
    if "eval_loss" in metrics:
        metrics["eval_perplexity"] = np.exp(metrics["eval_loss"])

    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete! Model saved to: {output_dir}")
    print("Final metrics:")
    train_loss = metrics.get("train_loss", "N/A")
    eval_loss = metrics.get("eval_loss", "N/A")
    eval_perplexity = metrics.get("eval_perplexity", "N/A")
    train_runtime = metrics.get("train_runtime", "N/A")

    print(
        f"  - Training Loss: {train_loss:.4f}"
        if isinstance(train_loss, (int, float))
        else f"  - Training Loss: {train_loss}"
    )
    print(
        f"  - Eval Loss: {eval_loss:.4f}"
        if isinstance(eval_loss, (int, float))
        else f"  - Eval Loss: {eval_loss}"
    )
    print(
        f"  - Eval Perplexity: {eval_perplexity:.2f}"
        if isinstance(eval_perplexity, (int, float))
        else f"  - Eval Perplexity: {eval_perplexity}"
    )
    print(
        f"  - Training Time: {train_runtime:.1f}s"
        if isinstance(train_runtime, (int, float))
        else f"  - Training Time: {train_runtime}s"
    )


if __name__ == "__main__":
    main()
