import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
import yaml

from train import load_config, create_model, prepare_datasets_modern, PerplexityCallback


class TestLoadConfig:
    """Test suite for config loading functionality"""

    def test_load_config_basic(self, temp_dir):
        """Test basic config loading from YAML"""
        config_data = {
            "model": {"name": "test_model", "type": "GPT2"},
            "training": {"lr": 0.001, "batch_size": 32},
            "data": {"seq_length": 128},
        }

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Mock args with no overrides
        args = Mock(lr=None, model_name=None)

        loaded_config = load_config(str(config_path), args)
        assert loaded_config == config_data

    def test_load_config_with_lr_override(self, temp_dir):
        """Test config loading with learning rate override"""
        config_data = {"model": {"name": "test_model"}, "training": {"lr": 0.001}}

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Mock args with lr override
        args = Mock(lr=0.01, model_name=None)

        loaded_config = load_config(str(config_path), args)
        assert loaded_config["training"]["lr"] == 0.01

    def test_load_config_with_model_name_override(self, temp_dir):
        """Test config loading with model name override"""
        config_data = {"model": {"name": "original_model"}, "training": {"lr": 0.001}}

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Mock args with model_name override
        args = Mock(lr=None, model_name="new_model")

        loaded_config = load_config(str(config_path), args)
        assert loaded_config["model"]["name"] == "new_model"

    def test_load_config_with_both_overrides(self, temp_dir):
        """Test config loading with both overrides"""
        config_data = {"model": {"name": "original_model"}, "training": {"lr": 0.001}}

        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Mock args with both overrides
        args = Mock(lr=0.02, model_name="new_model")

        loaded_config = load_config(str(config_path), args)
        assert loaded_config["training"]["lr"] == 0.02
        assert loaded_config["model"]["name"] == "new_model"

    def test_load_config_file_not_found(self):
        """Test error handling for missing config file"""
        args = Mock(lr=None, model_name=None)

        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml", args)


class TestCreateModel:
    """Test suite for model creation"""

    def test_create_llama_model(self, mock_tokenizer):
        """Test creating a Llama model"""
        config = {
            "model": {
                "type": "Llama",
                "hidden_size": 256,
                "intermediate_size": 1024,
                "n_layer": 4,
                "n_head": 4,
                "tie_word_embeddings": True,
            },
            "data": {"seq_length": 128},
        }

        with patch("train.LlamaForCausalLM") as mock_llama:
            create_model(config, mock_tokenizer)

            # Verify LlamaConfig was created with correct parameters
            mock_llama.assert_called_once()
            config_arg = mock_llama.call_args[0][0]
            assert config_arg.hidden_size == 256
            assert config_arg.num_hidden_layers == 4
            assert config_arg.num_attention_heads == 4
            assert config_arg.vocab_size == mock_tokenizer.vocab_size

    def test_create_gpt2_model(self, mock_tokenizer):
        """Test creating a GPT2 model"""
        config = {
            "model": {
                "type": "GPT2",
                "hidden_size": 256,
                "n_layer": 4,
                "n_head": 4,
                "resid_pdrop": 0.1,
                "embd_pdrop": 0.1,
                "attn_pdrop": 0.1,
            },
            "data": {"seq_length": 128},
        }

        with patch("train.GPT2LMHeadModel") as mock_gpt2:
            create_model(config, mock_tokenizer)

            # Verify GPT2Config was created with correct parameters
            mock_gpt2.assert_called_once()
            config_arg = mock_gpt2.call_args[0][0]
            assert config_arg.n_embd == 256
            assert config_arg.n_layer == 4
            assert config_arg.n_head == 4
            assert config_arg.vocab_size == mock_tokenizer.vocab_size

    def test_create_gptj_model(self, mock_tokenizer):
        """Test creating a GPTJ model"""
        config = {
            "model": {
                "type": "GPTJ",
                "hidden_size": 256,
                "n_layer": 4,
                "n_head": 4,
                "tie_word_embeddings": False,
            },
            "data": {"seq_length": 128},
        }

        with patch("train.GPTJForCausalLM") as mock_gptj:
            create_model(config, mock_tokenizer)

            # Verify GPTJConfig was created with correct parameters
            mock_gptj.assert_called_once()
            config_arg = mock_gptj.call_args[0][0]
            assert config_arg.n_embd == 256
            assert config_arg.n_layer == 4
            assert config_arg.n_head == 4
            assert not config_arg.tie_word_embeddings

    def test_create_model_unknown_type(self, mock_tokenizer):
        """Test error handling for unknown model type"""
        config = {"model": {"type": "UnknownModel"}, "data": {"seq_length": 128}}

        with pytest.raises(ValueError, match="Unknown model type: UnknownModel"):
            create_model(config, mock_tokenizer)

    def test_create_model_default_values(self, mock_tokenizer):
        """Test model creation with default values for optional parameters"""
        config = {
            "model": {"type": "GPT2", "hidden_size": 256, "n_layer": 4, "n_head": 4},
            "data": {"seq_length": 128},
        }

        with patch("train.GPT2LMHeadModel") as mock_gpt2:
            create_model(config, mock_tokenizer)

            # Verify default dropout values were used
            config_arg = mock_gpt2.call_args[0][0]
            assert config_arg.resid_pdrop == 0.1
            assert config_arg.embd_pdrop == 0.1
            assert config_arg.attn_pdrop == 0.1


class TestPrepareDatasets:
    """Test suite for dataset preparation"""

    @patch("train.Path.exists")
    @patch("datasets.load_from_disk")
    def test_prepare_datasets_cached(self, mock_load_disk, mock_exists, mock_tokenizer):
        """Test loading datasets from cache"""
        mock_exists.return_value = True
        mock_datasets = MagicMock()
        mock_datasets.__getitem__.side_effect = lambda x: (
            MagicMock() if x in ["train", "validation"] else None
        )
        mock_datasets["validation"].__len__.return_value = 100
        mock_load_disk.return_value = mock_datasets

        config = {
            "model": {"name": "test_model"},
            "data": {
                "train_path": "/data/train_clean",
                "seq_length": 128,
                "eval_samples": 1000,
            },
        }

        train_ds, val_ds = prepare_datasets_modern(config, mock_tokenizer)

        # Verify cache was loaded
        mock_load_disk.assert_called_once()
        assert train_ds is not None
        assert val_ds is not None

    @patch("train.Path.exists")
    @patch("train.Path.mkdir")
    @patch("train.DataProcessor")
    def test_prepare_datasets_no_cache(
        self, mock_processor_class, mock_mkdir, mock_exists, mock_tokenizer
    ):
        """Test preparing datasets without cache"""
        mock_exists.return_value = False

        # Mock DataProcessor instance
        mock_processor = MagicMock()
        mock_datasets = MagicMock()
        mock_datasets.__getitem__.side_effect = lambda x: (
            MagicMock() if x in ["train", "validation"] else None
        )
        mock_datasets["validation"].__len__.return_value = 100
        mock_processor.prepare_dataset.return_value = mock_datasets
        mock_processor_class.return_value = mock_processor

        config = {
            "model": {"name": "test_model"},
            "data": {
                "train_path": "/data/train_10M_clean",
                "seq_length": 128,
                "eval_samples": 1000,
            },
        }

        train_ds, val_ds = prepare_datasets_modern(config, mock_tokenizer)

        # Verify dataset was prepared and saved
        mock_processor.prepare_dataset.assert_called_once_with(
            train_data_dir="/data/train_10M",
            eval_data_dir="/data/train_dev",
            max_length=128,
            clean=True,
            num_proc=4,
        )
        mock_datasets.save_to_disk.assert_called_once()

    @patch("train.Path.exists")
    @patch("datasets.load_from_disk")
    @patch("train.sample")
    def test_prepare_datasets_sample_validation(
        self, mock_sample, mock_load_disk, mock_exists, mock_tokenizer
    ):
        """Test validation set sampling when too large"""
        mock_exists.return_value = True

        # Mock large validation dataset
        mock_val_dataset = MagicMock()
        mock_val_dataset.__len__.return_value = 2000
        mock_val_dataset.select.return_value = mock_val_dataset

        mock_datasets = {"train": MagicMock(), "validation": mock_val_dataset}
        mock_load_disk.return_value = mock_datasets

        # Mock sample to return specific indices
        mock_sample.return_value = list(range(500))

        config = {
            "model": {"name": "test_model"},
            "data": {
                "train_path": "/data/train_clean",
                "seq_length": 128,
                "eval_samples": 500,
            },
        }

        train_ds, val_ds = prepare_datasets_modern(config, mock_tokenizer)

        # Verify sampling was performed
        mock_sample.assert_called_once_with(range(2000), 500)
        mock_val_dataset.select.assert_called_once()


class TestPerplexityCallback:
    """Test suite for PerplexityCallback"""

    def test_perplexity_calculation(self):
        """Test perplexity calculation from loss"""
        callback = PerplexityCallback()

        # Mock arguments and state
        args = Mock()
        state = Mock(epoch=1.5)
        control = Mock()

        # Test with eval_loss
        metrics = {"eval_loss": 2.0}
        callback.on_evaluate(args, state, control, metrics=metrics)

        # Verify perplexity was calculated correctly
        expected_perplexity = pytest.approx(7.389, rel=1e-3)
        assert metrics["eval_perplexity"] == expected_perplexity

    def test_perplexity_no_eval_loss(self):
        """Test callback when no eval_loss is present"""
        callback = PerplexityCallback()

        args = Mock()
        state = Mock(epoch=1.0)
        control = Mock()

        # Test without eval_loss
        metrics = {"other_metric": 1.0}
        callback.on_evaluate(args, state, control, metrics=metrics)

        # Verify perplexity was not added
        assert "eval_perplexity" not in metrics

    def test_perplexity_no_metrics(self):
        """Test callback when metrics is None"""
        callback = PerplexityCallback()

        args = Mock()
        state = Mock(epoch=1.0)
        control = Mock()

        # Test with None metrics
        callback.on_evaluate(args, state, control, metrics=None)

        # Should not raise any errors

    @patch("builtins.print")
    def test_perplexity_logging(self, mock_print):
        """Test that perplexity is logged correctly"""
        callback = PerplexityCallback()

        args = Mock()
        state = Mock(epoch=2.5)
        control = Mock()
        metrics = {"eval_loss": 1.0}

        callback.on_evaluate(args, state, control, metrics=metrics)

        # Verify print was called with correct format
        pytest.approx(2.718, rel=1e-3)
        mock_print.assert_called_once()
        print_call = mock_print.call_args[0][0]
        assert "Epoch 2.5" in print_call
        assert "Perplexity: 2.72" in print_call


class TestMainFunction:
    """Test suite for main function integration"""

    @patch("train.argparse.ArgumentParser")
    @patch("train.load_config")
    @patch("train.GPT2TokenizerFast")
    @patch("train.prepare_datasets_modern")
    @patch("train.create_model")
    @patch("train.Trainer")
    @patch("train.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("train.yaml.dump")
    @patch("train.json.dump")
    def test_main_integration(
        self,
        mock_json_dump,
        mock_yaml_dump,
        mock_open,
        mock_mkdir,
        mock_trainer_class,
        mock_create_model,
        mock_prepare_datasets,
        mock_tokenizer_class,
        mock_load_config,
        mock_argparse,
    ):
        """Test main function integration"""
        # Setup mocks
        mock_args = Mock(config="test_config.yaml", lr=None, model_name=None)
        mock_parser = Mock()
        mock_parser.parse_args.return_value = mock_args
        mock_argparse.return_value = mock_parser

        mock_config = {
            "model": {"name": "test_model", "type": "GPT2"},
            "data": {"tokenizer_path": "tokenizer.json", "seq_length": 128},
            "training": {
                "lr": 0.001,
                "batch_size": 32,
                "gradient_accumulation_steps": 1,
                "num_epochs": 3,
                "warmup_steps": 100,
                "fp16": True,
                "bf16": False,
            },
            "logging": {"output_dir": "./models"},
        }
        mock_load_config.return_value = mock_config

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.return_value = mock_tokenizer

        # Mock datasets
        mock_train_ds = Mock()
        mock_train_ds.__len__ = Mock(return_value=1000)
        mock_eval_ds = Mock()
        mock_eval_ds.__len__ = Mock(return_value=100)
        mock_prepare_datasets.return_value = (mock_train_ds, mock_eval_ds)

        # Mock model
        mock_model = Mock()
        mock_model.num_parameters.return_value = 1000000
        mock_create_model.return_value = mock_model

        # Mock trainer
        mock_trainer = Mock()
        mock_train_result = Mock()
        mock_train_result.metrics = {
            "train_loss": 1.5,
            "train_runtime": 100,
            "eval_loss": 2.0,
            "eval_perplexity": 7.39,
        }
        mock_trainer.train.return_value = mock_train_result
        mock_trainer_class.return_value = mock_trainer

        # Import and run main
        from train import main

        main()

        # Verify correct sequence of calls
        mock_load_config.assert_called_once_with("test_config.yaml", mock_args)
        mock_tokenizer_class.assert_called_once()
        mock_prepare_datasets.assert_called_once()
        mock_create_model.assert_called_once_with(mock_config, mock_tokenizer)
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
