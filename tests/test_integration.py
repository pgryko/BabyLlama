import pytest
import torch
import yaml
import json
from unittest.mock import patch, Mock

# Import modules to test
from data_utils import DataProcessor, DomainCleaners
from train import load_config, create_model, prepare_datasets_modern
from evaluate import ModelEvaluator


class TestDataPipeline:
    """Integration tests for the complete data processing pipeline"""

    def test_end_to_end_data_processing(self, temp_dir, mock_tokenizer):
        """Test complete data processing from raw files to datasets"""
        # Create test data files
        train_dir = temp_dir / "train"
        train_dir.mkdir()
        eval_dir = temp_dir / "eval"
        eval_dir.mkdir()

        # Create train files
        train_texts = [
            "This is the first training document. It contains multiple sentences.",
            "Here's another training document with different content.",
            "The third document has special characters: @#$% and numbers: 12345.",
        ]

        for i, text in enumerate(train_texts):
            (train_dir / f"doc{i}.train").write_text(
                text * 10
            )  # Repeat to ensure chunks

        # Create eval files
        eval_texts = [
            "This is an evaluation document for testing.",
            "Another eval doc with some content here.",
        ]

        for i, text in enumerate(eval_texts):
            (eval_dir / f"eval{i}.dev").write_text(text * 10)

        # Mock tokenizer to return predictable tokens
        def mock_encode(text, **kwargs):
            # Create enough tokens to make multiple chunks of max_length
            # Each character becomes a token, repeat to get enough tokens
            tokens = list(range(len(text) * 10))  # Multiply to get enough tokens
            return {"input_ids": tokens}

        mock_tokenizer.side_effect = mock_encode
        mock_tokenizer.vocab_size = 1000

        # Process data
        processor = DataProcessor(mock_tokenizer)
        dataset_dict = processor.prepare_dataset(
            train_data_dir=str(train_dir),
            eval_data_dir=str(eval_dir),
            max_length=128,
            clean=True,
            num_proc=1,  # Single process for testing
        )

        # Verify dataset structure
        assert "train" in dataset_dict
        assert "validation" in dataset_dict
        assert len(dataset_dict["train"]) > 0
        assert len(dataset_dict["validation"]) > 0

        # Verify chunking worked
        first_sample = dataset_dict["train"][0]
        assert "input_ids" in first_sample
        assert len(first_sample["input_ids"]) == 128

    def test_domain_specific_cleaning_pipeline(self, temp_dir, mock_tokenizer):
        """Test data processing with domain-specific cleaners"""
        # Create test files with domain-specific content
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        # Wikipedia-style content
        wiki_text = """== Introduction ==
        This is a Wikipedia article[1] about testing[2].
        
        
        
        == References ==
        [1] Test reference
        [2] Another reference"""

        (data_dir / "wiki.train").write_text(wiki_text)

        # Process with Wikipedia cleaner
        processor = DataProcessor(mock_tokenizer)
        dataset = processor.load_text_files(str(data_dir), "train")

        # Apply Wikipedia cleaning
        cleaned_dataset = dataset.map(
            lambda x: {"text": DomainCleaners.wikipedia(x["text"])}
        )

        # Verify cleaning
        cleaned_text = cleaned_dataset[0]["text"]
        assert "==" not in cleaned_text
        assert "[1]" not in cleaned_text
        assert "Introduction" in cleaned_text


class TestTrainingPipeline:
    """Integration tests for the training pipeline"""

    @patch("train.Trainer")
    @patch("train.GPT2TokenizerFast")
    @patch("train.DataProcessor")
    def test_config_to_training_pipeline(
        self, mock_processor_class, mock_tokenizer_class, mock_trainer_class, temp_dir
    ):
        """Test complete pipeline from config to training setup"""
        # Create config file
        config_data = {
            "model": {
                "name": "test_model",
                "type": "GPT2",
                "hidden_size": 64,
                "n_layer": 2,
                "n_head": 2,
            },
            "data": {
                "tokenizer_path": str(temp_dir / "tokenizer.json"),
                "train_path": str(temp_dir / "train_clean"),
                "seq_length": 128,
                "eval_samples": 100,
            },
            "training": {
                "lr": 0.001,
                "batch_size": 4,
                "gradient_accumulation_steps": 1,
                "num_epochs": 1,
                "warmup_steps": 10,
                "fp16": False,
                "bf16": False,
            },
            "logging": {"output_dir": str(temp_dir / "models")},
        }

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Create mock tokenizer file
        tokenizer_file = temp_dir / "tokenizer.json"
        tokenizer_file.write_text("{}")

        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.model_max_length = 128
        mock_tokenizer_class.return_value = mock_tokenizer

        # Mock datasets
        mock_train_dataset = Mock()
        mock_eval_dataset = Mock()
        mock_eval_dataset.__len__ = Mock(
            return_value=1000
        )  # Mock validation dataset size
        mock_eval_dataset.select = Mock(
            return_value=mock_eval_dataset
        )  # Mock select method

        # Create a dict-like mock that supports both getting and setting
        mock_datasets_dict = {
            "train": mock_train_dataset,
            "validation": mock_eval_dataset,
        }
        mock_datasets = Mock()
        mock_datasets.__getitem__ = Mock(
            side_effect=lambda key: mock_datasets_dict[key]
        )
        mock_datasets.__setitem__ = Mock(
            side_effect=lambda key, value: mock_datasets_dict.__setitem__(key, value)
        )
        mock_datasets.save_to_disk = Mock()
        mock_processor = Mock()
        mock_processor.prepare_dataset.return_value = mock_datasets
        mock_processor_class.return_value = mock_processor

        # Mock trainer
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        # Load config and create model
        args = Mock(lr=None, model_name=None)
        config = load_config(str(config_path), args)

        # Create model
        model = create_model(config, mock_tokenizer)

        # Verify model creation
        assert model is not None
        assert hasattr(model, "config")

        # Prepare datasets
        mock_processor.prepare_dataset.return_value = mock_datasets
        train_ds, eval_ds = prepare_datasets_modern(config, mock_tokenizer)

        assert train_ds == mock_train_dataset
        assert eval_ds == mock_eval_dataset

    def test_model_architecture_compatibility(self, mock_tokenizer):
        """Test that all model architectures can be created with same config structure"""
        base_config = {"data": {"seq_length": 128}}

        architectures = ["Llama", "GPT2", "GPTJ"]

        for arch in architectures:
            config = base_config.copy()
            config["model"] = {
                "type": arch,
                "hidden_size": 128,
                "n_layer": 2,
                "n_head": 2,
                "intermediate_size": 512,  # For Llama
            }

            # Should not raise any errors
            with patch(
                f"train.{arch}ForCausalLM"
                if arch in ["Llama", "GPTJ"]
                else f"train.{arch}LMHeadModel"
            ):
                model = create_model(config, mock_tokenizer)
                assert model is not None


class TestEvaluationPipeline:
    """Integration tests for the evaluation pipeline"""

    @patch("evaluate.GPT2TokenizerFast.from_pretrained")
    @patch("evaluate.LlamaForCausalLM.from_pretrained")
    def test_evaluation_workflow(
        self, mock_model_class, mock_tokenizer_class, temp_dir
    ):
        """Test complete evaluation workflow"""
        # Setup model directory
        model_dir = temp_dir / "model"
        model_dir.mkdir()

        # Create config file
        config_data = {"model_type": "llama"}
        with open(model_dir / "config.json", "w") as f:
            json.dump(config_data, f)

        # Setup mocks
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer_class.return_value = mock_tokenizer

        # Mock model outputs
        mock_output = Mock()
        mock_output.loss = Mock(return_value=torch.tensor(2.0))
        mock_output.loss.item.return_value = 2.0
        mock_output.logits = torch.randn(1, 10, 1000)
        mock_model.return_value = mock_output
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        # Mock tokenizer outputs
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        mock_tokenizer.decode.return_value = "Generated text output"

        # Create evaluator
        evaluator = ModelEvaluator(str(model_dir), device="cpu")

        # Run evaluations
        test_texts = ["This is a test sentence.", "Another test here."]

        # Test perplexity
        perplexity_results = evaluator.calculate_perplexity(test_texts)
        assert isinstance(perplexity_results["perplexity"], float)
        assert perplexity_results["num_sequences"] == 2

        # Test generation quality
        gen_results = evaluator.evaluate_generation_quality(
            ["Test prompt"], max_length=50, num_return_sequences=2
        )
        assert "avg_diversity_score" in gen_results
        assert "avg_repetition_score" in gen_results

        # Test token probabilities
        prob_results = evaluator.evaluate_token_probabilities(test_texts[:1])
        assert "avg_entropy" in prob_results
        assert 0 <= prob_results["avg_top_token_prob"] <= 1


class TestCrossFunctionalIntegration:
    """Test interactions between different components"""

    def test_tokenizer_consistency(self, temp_dir):
        """Test that tokenizer usage is consistent across modules"""
        # Create a mock tokenizer with consistent behavior
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.bos_token = "<s>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token = "<pad>"

        # Test in data processing
        processor = DataProcessor(mock_tokenizer)
        assert processor.tokenizer.vocab_size == 1000

        # Test in model creation
        config = {
            "model": {"type": "GPT2", "hidden_size": 128, "n_layer": 2, "n_head": 2},
            "data": {"seq_length": 128},
        }

        with patch("train.GPT2LMHeadModel"):
            create_model(config, mock_tokenizer)
            # Model creation should use tokenizer's vocab_size

    def test_config_parameter_propagation(self, temp_dir):
        """Test that config parameters propagate correctly through the pipeline"""
        # Create a comprehensive config
        seq_length = 256
        batch_size = 8

        config = {
            "model": {
                "name": "integration_test",
                "type": "GPT2",
                "hidden_size": 128,
                "n_layer": 4,
                "n_head": 4,
            },
            "data": {
                "seq_length": seq_length,
                "tokenizer_path": "tokenizer.json",
                "train_path": str(temp_dir / "train"),
                "eval_samples": 50,
            },
            "training": {
                "batch_size": batch_size,
                "gradient_accumulation_steps": 2,
                "lr": 0.0001,
                "num_epochs": 2,
                "warmup_steps": 100,
                "fp16": True,
            },
            "logging": {"output_dir": str(temp_dir / "output")},
        }

        # Verify seq_length is used correctly
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 1000

        with patch("train.GPT2LMHeadModel") as mock_model_class:
            create_model(config, mock_tokenizer)

            # Check that model config uses correct sequence length
            call_args = mock_model_class.call_args[0][0]
            assert call_args.n_positions == seq_length

    @patch("builtins.print")
    def test_error_handling_integration(self, mock_print, temp_dir):
        """Test error handling across modules"""
        # Test with invalid model type
        config = {"model": {"type": "InvalidModel"}, "data": {"seq_length": 128}}

        mock_tokenizer = Mock()

        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(config, mock_tokenizer)

        # Test with missing config file
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml", Mock())

        # Test with invalid data directory
        processor = DataProcessor(mock_tokenizer)
        dataset = processor.load_text_files("/invalid/path", "train")
        assert len(dataset) == 0  # Should return empty dataset


class TestPerformanceIntegration:
    """Test performance-related integration aspects"""

    def test_batch_processing_efficiency(self, mock_tokenizer):
        """Test that batch processing works correctly across modules"""
        # Create processor
        processor = DataProcessor(mock_tokenizer)

        # Mock tokenizer to handle batches
        def mock_batch_encode(texts, **kwargs):
            if isinstance(texts, list):
                return {"input_ids": [list(range(len(t) * 10)) for t in texts]}
            else:
                # texts is a string
                return {"input_ids": list(range(len(texts) * 10))}

        mock_tokenizer.side_effect = mock_batch_encode

        # Test with multiple texts
        texts = ["Text 1" * 50, "Text 2" * 50, "Text 3" * 50]
        examples = [{"text": t} for t in texts]

        # Process in batch
        for example in examples:
            result = processor.tokenize_and_chunk(example, max_length=128)
            assert "input_ids" in result
            assert all(len(chunk) == 128 for chunk in result["input_ids"])

    def test_memory_efficient_evaluation(self):
        """Test that evaluation handles large datasets efficiently"""
        # This is more of a design test - ensuring methods exist for batch processing
        evaluator = ModelEvaluator.__new__(ModelEvaluator)

        # Verify batch processing parameters exist
        import inspect

        # Check calculate_perplexity has batch_size parameter
        sig = inspect.signature(evaluator.calculate_perplexity)
        assert "batch_size" in sig.parameters

        # Check other methods handle batching appropriately
        sig = inspect.signature(evaluator.evaluate_generation_quality)
        assert "prompts" in sig.parameters  # Takes list for batch processing
