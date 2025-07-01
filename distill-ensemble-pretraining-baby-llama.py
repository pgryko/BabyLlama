"""
Knowledge distillation script for BabyLlama
"""

import argparse
import yaml
from pathlib import Path
from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from random import sample

from data_utils import DataProcessor


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        for teacher in self.teachers:
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        with torch.no_grad():
            all_teacher_logits = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
        ) * (self.args.temperature**2)

        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config/distillation.yaml",
        help="Configuration file path",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    tokenizer_path = Path(config["data"]["tokenizer_path"])
    tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"

    processor = DataProcessor(tokenizer)
    datasets = processor.prepare_dataset(
        data_dir=config["data"]["train_path"].replace("_clean", ""),
        max_length=config["distillation"]["seq_length"],
        clean=True,
        num_proc=4,
    )
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]

    if len(eval_dataset) > config["data"]["eval_samples"]:
        indices = sample(range(len(eval_dataset)), config["data"]["eval_samples"])
        eval_dataset = Subset(eval_dataset, indices)

    tokenizer.model_max_length = config["distillation"]["seq_length"]

    student_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config["student"]["hidden_size"],
        num_hidden_layers=config["student"]["n_layer"],
        intermediate_size=config["student"]["intermediate_size"],
        num_attention_heads=config["student"]["n_head"],
        bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        max_position_embeddings=config["distillation"]["seq_length"],
    )

    student = LlamaForCausalLM(student_config)

    teacher1 = LlamaForCausalLM.from_pretrained(config["teachers"][0])
    teacher2 = GPT2LMHeadModel.from_pretrained(config["teachers"][1])
    teachers = [teacher1, teacher2]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print(f"Student model parameters: {student.num_parameters():,}")
    print(f"Teacher 1 model parameters: {teacher1.num_parameters():,}")
    print(f"Teacher 2 model parameters: {teacher2.num_parameters():,}")

    output_dir = Path(config["logging"]["output_dir"]) / config["student"]["name"]

    training_args = DistillationTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=config["distillation"]["num_epochs"],
        gradient_accumulation_steps=1,
        per_device_train_batch_size=config["distillation"]["batch_size"],
        save_total_limit=1,
        report_to="wandb" if config["logging"].get("wandb") else "none",
        warmup_steps=config["distillation"]["warmup_steps"],
        lr_scheduler_type="cosine",
        learning_rate=float(config["distillation"]["lr"]),
        logging_steps=20,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.1,
        alpha=config["distillation"]["alpha"],
        temperature=config["distillation"]["temperature"],
    )

    trainer = DistillationTrainer(
        student,
        training_args,
        teacher_models=teachers,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if config["logging"].get("wandb"):
        import wandb

        wandb.login()
        wandb.init(project=config["logging"]["project"], name=config["student"]["name"])

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
