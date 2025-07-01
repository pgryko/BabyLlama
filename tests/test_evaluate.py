import pytest
from unittest.mock import Mock, patch, mock_open
import torch
import numpy as np
import json
from pathlib import Path

from evaluate import ModelEvaluator


class TestModelEvaluator:
    """Test suite for ModelEvaluator class"""

    @patch("evaluate.GPT2TokenizerFast.from_pretrained")
    @patch("evaluate.LlamaForCausalLM.from_pretrained")
    @patch("builtins.open", new_callable=mock_open)
    def test_init_llama_model(self, mock_file, mock_llama, mock_tokenizer):
        """Test initialization with Llama model"""
        # Mock config file
        config_data = {"model_type": "llama"}
        mock_file.return_value.read.return_value = json.dumps(config_data)

        # Mock model and tokenizer
        mock_model = Mock()
        mock_llama.return_value = mock_model
        mock_tokenizer.return_value = Mock()

        # Create evaluator
        evaluator = ModelEvaluator("/path/to/model", device="cuda")

        # Verify initialization
        assert evaluator.model == mock_model
        mock_model.to.assert_called_once_with("cuda")
        mock_model.eval.assert_called_once()
        mock_llama.assert_called_once_with("/path/to/model")

    @patch("evaluate.GPT2TokenizerFast.from_pretrained")
    @patch("evaluate.GPT2LMHeadModel.from_pretrained")
    @patch("builtins.open", new_callable=mock_open)
    def test_init_gpt2_model(self, mock_file, mock_gpt2, mock_tokenizer):
        """Test initialization with GPT2 model"""
        config_data = {"model_type": "gpt2"}
        mock_file.return_value.read.return_value = json.dumps(config_data)

        mock_model = Mock()
        mock_gpt2.return_value = mock_model
        mock_tokenizer.return_value = Mock()

        evaluator = ModelEvaluator("/path/to/model", device="cpu")

        assert evaluator.model == mock_model
        mock_model.to.assert_called_once_with("cpu")
        mock_gpt2.assert_called_once_with("/path/to/model")

    @patch("evaluate.GPT2TokenizerFast.from_pretrained")
    @patch("evaluate.GPTJForCausalLM.from_pretrained")
    @patch("builtins.open", new_callable=mock_open)
    def test_init_gptj_model(self, mock_file, mock_gptj, mock_tokenizer):
        """Test initialization with GPTJ model"""
        config_data = {"model_type": "gptj"}
        mock_file.return_value.read.return_value = json.dumps(config_data)

        mock_model = Mock()
        mock_gptj.return_value = mock_model
        mock_tokenizer.return_value = Mock()

        evaluator = ModelEvaluator("/path/to/model")

        assert evaluator.model == mock_model
        mock_gptj.assert_called_once_with("/path/to/model")

    @patch("evaluate.GPT2TokenizerFast.from_pretrained")
    @patch("builtins.open", new_callable=mock_open)
    def test_init_unknown_model_type(self, mock_file, mock_tokenizer):
        """Test initialization with unknown model type"""
        config_data = {"model_type": "unknown"}
        mock_file.return_value.read.return_value = json.dumps(config_data)

        with pytest.raises(ValueError, match="Unknown model type: unknown"):
            ModelEvaluator("/path/to/model")

    def test_calculate_perplexity(self, mock_model, mock_tokenizer):
        """Test perplexity calculation"""
        # Setup evaluator with mocks
        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer
        evaluator.device = "cpu"

        # Mock tokenizer output
        mock_input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
        mock_attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        mock_tokenizer.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask,
        }

        # Mock model output
        mock_output = Mock()
        mock_output.loss = Mock()
        mock_output.loss.item.return_value = 2.0
        mock_model.return_value = mock_output

        # Test perplexity calculation
        texts = ["Test text 1", "Test text 2"]
        results = evaluator.calculate_perplexity(texts, batch_size=2)

        # Verify results structure
        assert "perplexity" in results
        assert "average_loss" in results
        assert "std_loss" in results
        assert "num_sequences" in results
        assert results["num_sequences"] == 2
        assert results["total_tokens"] == 7  # 4 + 3 (excluding padding)

    def test_calculate_perplexity_empty_texts(self, mock_model, mock_tokenizer):
        """Test perplexity calculation with empty texts"""
        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer
        evaluator.device = "cpu"

        results = evaluator.calculate_perplexity([], batch_size=8)

        assert results["num_sequences"] == 0
        assert results["total_tokens"] == 0

    def test_evaluate_generation_quality(self, mock_model, mock_tokenizer):
        """Test generation quality evaluation"""
        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer
        evaluator.device = "cpu"

        # Mock tokenizer encode/decode
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.side_effect = lambda x, **kwargs: "Generated text output"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2

        # Mock model generate
        mock_model.generate.return_value = torch.tensor(
            [[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9], [1, 2, 3, 10, 11, 12]]
        )

        # Test generation
        prompts = ["Test prompt"]
        results = evaluator.evaluate_generation_quality(
            prompts, max_length=50, num_return_sequences=3
        )

        # Verify results
        assert "avg_diversity_score" in results
        assert "avg_repetition_score" in results
        assert "generation_samples" in results
        assert len(results["diversity_scores"]) == 1
        assert len(results["repetition_scores"]) == 1

    def test_calculate_diversity(self):
        """Test diversity calculation"""
        evaluator = ModelEvaluator.__new__(ModelEvaluator)

        # Test with diverse texts
        texts = ["The cat sat on mat", "A dog ran in park", "Birds fly high"]
        diversity = evaluator._calculate_diversity(texts, n=2)
        assert diversity > 0.8  # High diversity expected

        # Test with repetitive texts
        texts = ["The cat sat", "The cat sat", "The cat sat"]
        diversity = evaluator._calculate_diversity(texts, n=2)
        assert diversity < 0.5  # Low diversity expected

        # Test with empty texts
        diversity = evaluator._calculate_diversity([], n=2)
        assert diversity == 0.0

        # Test with single word texts
        texts = ["Hello", "World"]
        diversity = evaluator._calculate_diversity(texts, n=3)
        assert diversity == 0.0  # No 3-grams possible

    def test_calculate_repetition(self):
        """Test repetition calculation"""
        evaluator = ModelEvaluator.__new__(ModelEvaluator)

        # Test with no repetition
        texts = ["The cat sat on mat", "A dog ran fast"]
        repetition = evaluator._calculate_repetition(texts)
        assert repetition == 0.0

        # Test with repetition
        texts = ["The the cat cat sat", "Dog dog ran ran"]
        repetition = evaluator._calculate_repetition(texts)
        assert repetition > 0.5

        # Test with empty texts
        repetition = evaluator._calculate_repetition([])
        assert repetition == 0.0

        # Test with single word texts
        texts = ["Hello", "World"]
        repetition = evaluator._calculate_repetition(texts)
        assert repetition == 0.0

    def test_evaluate_token_probabilities(self, mock_model, mock_tokenizer):
        """Test token probability evaluation"""
        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer
        evaluator.device = "cpu"

        # Mock tokenizer
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        # Mock model output with logits
        mock_logits = torch.randn(1, 3, 1000)  # batch=1, seq_len=3, vocab_size=1000
        mock_output = Mock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output

        # Test probability evaluation
        texts = ["Test text"]
        results = evaluator.evaluate_token_probabilities(texts)

        # Verify results
        assert "avg_top_token_prob" in results
        assert "std_top_token_prob" in results
        assert "avg_entropy" in results
        assert "std_entropy" in results
        assert "low_confidence_ratio" in results

        # Check value ranges
        assert 0 <= results["avg_top_token_prob"] <= 1
        assert results["avg_entropy"] >= 0
        assert 0 <= results["low_confidence_ratio"] <= 1

    @patch("evaluate.plt.savefig")
    @patch("evaluate.plt.subplots")
    def test_plot_metrics(self, mock_subplots, mock_savefig):
        """Test metrics plotting"""
        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.model_path = Path("/path/to/model")

        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Test metrics
        metrics = {
            "diversity_scores": [0.5, 0.6, 0.7, 0.8],
            "repetition_scores": [0.1, 0.2, 0.1, 0.15],
            "perplexity": 25.5,
            "avg_diversity_score": 0.65,
            "avg_repetition_score": 0.14,
        }

        # Plot metrics
        evaluator.plot_metrics(metrics, save_path="test_plot.png")

        # Verify plotting calls
        mock_fig.suptitle.assert_called_once()
        assert mock_axes[0, 0].hist.called
        assert mock_axes[0, 1].hist.called
        mock_savefig.assert_called_once_with(
            "test_plot.png", dpi=150, bbox_inches="tight"
        )


class TestMetricCalculations:
    """Test specific metric calculation methods"""

    def test_diversity_edge_cases(self):
        """Test diversity calculation edge cases"""
        evaluator = ModelEvaluator.__new__(ModelEvaluator)

        # Test with punctuation and special characters
        texts = ["Hello, world!", "Test: 123", "Special @#$ chars"]
        diversity = evaluator._calculate_diversity(texts, n=2)
        assert diversity > 0

        # Test with very long n-gram size
        texts = ["Short text", "Another short"]
        diversity = evaluator._calculate_diversity(texts, n=10)
        assert diversity == 0.0

        # Test with identical texts
        texts = ["Same text"] * 5
        diversity = evaluator._calculate_diversity(texts, n=2)
        assert diversity < 0.3

    def test_repetition_patterns(self):
        """Test various repetition patterns"""
        evaluator = ModelEvaluator.__new__(ModelEvaluator)

        # Test alternating repetition
        texts = ["A B A B A B", "C D C D C D"]
        repetition = evaluator._calculate_repetition(texts)
        assert repetition > 0

        # Test partial repetition
        texts = ["The cat sat on the mat", "A dog ran in the park"]
        repetition = evaluator._calculate_repetition(texts)
        assert repetition < 0.5

        # Test with numbers
        texts = ["1 2 3 1 2 3", "4 5 6 4 5 6"]
        repetition = evaluator._calculate_repetition(texts)
        assert repetition > 0


class TestIntegration:
    """Integration tests for the evaluator"""

    @patch("evaluate.GPT2TokenizerFast.from_pretrained")
    @patch("evaluate.LlamaForCausalLM.from_pretrained")
    @patch("builtins.open", new_callable=mock_open)
    def test_full_evaluation_pipeline(
        self, mock_file, mock_llama, mock_tokenizer_class
    ):
        """Test complete evaluation pipeline"""
        # Setup mocks
        config_data = {"model_type": "llama"}
        mock_file.return_value.read.return_value = json.dumps(config_data)

        mock_model = Mock()
        mock_llama.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer_class.return_value = mock_tokenizer

        # Mock tokenizer calls
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        mock_tokenizer.decode.return_value = "Generated text"

        # Mock model outputs
        mock_output = Mock()
        mock_output.loss = Mock()
        mock_output.loss.item.return_value = 2.5
        mock_output.logits = torch.randn(1, 4, 1000)
        mock_model.return_value = mock_output
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        # Create evaluator and run evaluation
        evaluator = ModelEvaluator("/path/to/model", device="cpu")

        # Test perplexity
        perplexity_results = evaluator.calculate_perplexity(["Test text"])
        assert perplexity_results["perplexity"] > 0

        # Test generation quality
        generation_results = evaluator.evaluate_generation_quality(
            ["Test prompt"], num_return_sequences=1
        )
        assert "avg_diversity_score" in generation_results

        # Test token probabilities
        prob_results = evaluator.evaluate_token_probabilities(["Test text"])
        assert "avg_entropy" in prob_results
