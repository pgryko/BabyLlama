from unittest.mock import patch, MagicMock

from data_utils import DataProcessor, DomainCleaners, create_cleaner_registry


class TestDataProcessor:
    """Test suite for DataProcessor class"""

    def test_load_text_files(self, temp_dir, mock_tokenizer):
        """Test loading text files from directory"""
        # Create test files
        train_files = []
        for i in range(3):
            file_path = temp_dir / f"file{i}.train"
            file_path.write_text(f"This is train file {i}")
            train_files.append(file_path)

        # Create processor and load files
        processor = DataProcessor(mock_tokenizer)
        dataset = processor.load_text_files(str(temp_dir), "train")

        # Verify dataset
        assert len(dataset) == 3
        assert all("text" in sample for sample in dataset)
        assert all("source" in sample for sample in dataset)
        assert dataset[0]["text"] == "This is train file 0"
        assert dataset[0]["source"] == "file0"

    def test_load_text_files_empty_directory(self, temp_dir, mock_tokenizer):
        """Test loading from empty directory"""
        processor = DataProcessor(mock_tokenizer)
        dataset = processor.load_text_files(str(temp_dir), "train")
        assert len(dataset) == 0

    def test_clean_text_basic(self, mock_tokenizer):
        """Test basic text cleaning functionality"""
        processor = DataProcessor(mock_tokenizer)

        # Test multiple spaces
        example = {"text": "This  has   multiple    spaces"}
        cleaned = processor.clean_text(example)
        assert cleaned["text"] == "This has multiple spaces"

        # Test space before punctuation
        example = {"text": "Hello , world !"}
        cleaned = processor.clean_text(example)
        assert cleaned["text"] == "Hello, world!"

        # Test missing space after punctuation
        example = {"text": "Hello.World"}
        cleaned = processor.clean_text(example)
        assert cleaned["text"] == "Hello. World"

        # Test combined issues
        example = {"text": "  This  is  a  test .Next  sentence  "}
        cleaned = processor.clean_text(example)
        assert cleaned["text"] == "This is a test. Next sentence"

    def test_clean_text_preserves_other_fields(self, mock_tokenizer):
        """Test that clean_text preserves other fields in example"""
        processor = DataProcessor(mock_tokenizer)
        example = {"text": "Test  text", "source": "file1", "id": 123}
        cleaned = processor.clean_text(example)

        assert cleaned["text"] == "Test text"
        assert cleaned["source"] == "file1"
        assert cleaned["id"] == 123

    def test_tokenize_and_chunk_full_chunks(self, mock_tokenizer):
        """Test tokenization and chunking with full chunks"""
        # Mock tokenizer to return predictable tokens
        mock_tokenizer.return_value = {"input_ids": list(range(300))}

        processor = DataProcessor(mock_tokenizer)
        example = {"text": "Some text"}
        result = processor.tokenize_and_chunk(example, max_length=128)

        # Should have 2 full chunks of 128 tokens each
        assert len(result["input_ids"]) == 2
        assert all(len(chunk) == 128 for chunk in result["input_ids"])
        assert result["input_ids"][0] == list(range(128))
        assert result["input_ids"][1] == list(range(128, 256))

    def test_tokenize_and_chunk_partial_chunk_excluded(self, mock_tokenizer):
        """Test that partial chunks are excluded"""
        # Mock tokenizer to return tokens not divisible by max_length
        mock_tokenizer.return_value = {"input_ids": list(range(150))}

        processor = DataProcessor(mock_tokenizer)
        example = {"text": "Some text"}
        result = processor.tokenize_and_chunk(example, max_length=128)

        # Should only have 1 full chunk, partial chunk excluded
        assert len(result["input_ids"]) == 1
        assert len(result["input_ids"][0]) == 128

    def test_tokenize_and_chunk_short_text(self, mock_tokenizer):
        """Test tokenization with text shorter than max_length"""
        # Mock tokenizer to return few tokens
        mock_tokenizer.return_value = {"input_ids": list(range(50))}

        processor = DataProcessor(mock_tokenizer)
        example = {"text": "Short"}
        result = processor.tokenize_and_chunk(example, max_length=128)

        # Should have no chunks as text is too short
        assert len(result["input_ids"]) == 0

    @patch("data_utils.DataProcessor.load_text_files")
    @patch("data_utils.DataProcessor.clean_text")
    @patch("data_utils.DataProcessor.tokenize_and_chunk")
    def test_prepare_dataset(self, mock_chunk, mock_clean, mock_load, mock_tokenizer):
        """Test full dataset preparation pipeline"""
        # Setup mocks
        mock_dataset = MagicMock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.flatten_indices.return_value = mock_dataset
        mock_load.return_value = mock_dataset
        mock_clean.side_effect = lambda x: x
        mock_chunk.return_value = {"input_ids": [[1, 2, 3]]}

        processor = DataProcessor(mock_tokenizer)
        dataset_dict = processor.prepare_dataset(
            train_data_dir="/train",
            eval_data_dir="/eval",
            max_length=128,
            clean=True,
            num_proc=4,
        )

        # Verify calls
        assert mock_load.call_count == 2
        mock_load.assert_any_call("/train", "train")
        mock_load.assert_any_call("/eval", "dev")

        # Verify dataset structure
        assert "train" in dataset_dict
        assert "validation" in dataset_dict


class TestDomainCleaners:
    """Test suite for domain-specific cleaners"""

    def test_wikipedia_cleaner(self):
        """Test Wikipedia text cleaning"""
        text = """== Section Header ==
        Some text with citation[1] and another[23].
        
        
        
        More text here."""

        cleaned = DomainCleaners.wikipedia(text)
        assert "==" not in cleaned
        assert "[1]" not in cleaned
        assert "[23]" not in cleaned
        assert "\n\n\n" not in cleaned
        assert "Section Header" in cleaned

    def test_subtitles_cleaner(self):
        """Test subtitle text cleaning"""
        text = """1
00:01:23,456 --> 00:01:26,789
This is a subtitle.

2
00:01:27,000 --> 00:01:30,123
Another subtitle here.

Subtitles by Someone"""

        cleaned = DomainCleaners.subtitles(text)
        assert "00:01:23" not in cleaned
        assert "-->" not in cleaned
        assert "Subtitles by" not in cleaned
        assert "This is a subtitle" in cleaned
        assert "Another subtitle here" in cleaned

    def test_dialogue_cleaner(self):
        """Test dialogue text cleaning"""
        text = """JOHN: Hello there!
        MARY: Hi [waves hand] how are you?
        JOHN: I'm fine (smiling broadly).
        NARRATOR: They continued talking."""

        cleaned = DomainCleaners.dialogue(text)
        assert "JOHN:" not in cleaned
        assert "MARY:" not in cleaned
        assert "[waves hand]" not in cleaned
        assert "(smiling broadly)" not in cleaned
        assert "Hello there!" in cleaned
        assert "Hi how are you?" in cleaned

    def test_empty_string_cleaners(self):
        """Test cleaners with empty strings"""
        assert DomainCleaners.wikipedia("") == ""
        assert DomainCleaners.subtitles("") == ""
        assert DomainCleaners.dialogue("") == ""

    def test_cleaners_preserve_regular_text(self):
        """Test that cleaners preserve regular text"""
        regular_text = "This is just regular text without any special formatting."

        assert DomainCleaners.wikipedia(regular_text).strip() == regular_text
        assert DomainCleaners.subtitles(regular_text).strip() == regular_text
        assert DomainCleaners.dialogue(regular_text).strip() == regular_text


class TestCleanerRegistry:
    """Test cleaner registry functionality"""

    def test_create_cleaner_registry(self):
        """Test that registry contains all expected cleaners"""
        registry = create_cleaner_registry()

        expected_cleaners = [
            "wikipedia",
            "simple_wikipedia",
            "subtitles",
            "open_subtitles",
            "dialogue",
            "switchboard",
            "default",
        ]

        for cleaner in expected_cleaners:
            assert cleaner in registry
            assert callable(registry[cleaner])

    def test_default_cleaner_identity(self):
        """Test that default cleaner returns input unchanged"""
        registry = create_cleaner_registry()
        text = "This should not be changed"
        assert registry["default"](text) == text

    def test_wikipedia_cleaners_same_function(self):
        """Test that wikipedia and simple_wikipedia use same cleaner"""
        registry = create_cleaner_registry()
        assert registry["wikipedia"] == registry["simple_wikipedia"]

    def test_subtitle_cleaners_same_function(self):
        """Test that subtitles and open_subtitles use same cleaner"""
        registry = create_cleaner_registry()
        assert registry["subtitles"] == registry["open_subtitles"]

    def test_dialogue_cleaners_same_function(self):
        """Test that dialogue and switchboard use same cleaner"""
        registry = create_cleaner_registry()
        assert registry["dialogue"] == registry["switchboard"]


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_unicode_handling(self, mock_tokenizer):
        """Test handling of unicode characters"""
        processor = DataProcessor(mock_tokenizer)
        example = {"text": "Hello 🌍 world 你好"}
        cleaned = processor.clean_text(example)
        assert "🌍" in cleaned["text"]
        assert "你好" in cleaned["text"]

    def test_very_long_text(self, mock_tokenizer):
        """Test handling of very long text"""
        mock_tokenizer.return_value = {"input_ids": list(range(10000))}

        processor = DataProcessor(mock_tokenizer)
        example = {"text": "x" * 10000}
        result = processor.tokenize_and_chunk(example, max_length=128)

        # Should create many full chunks
        assert len(result["input_ids"]) == 78  # 10000 // 128
        assert all(len(chunk) == 128 for chunk in result["input_ids"])

    def test_punctuation_edge_cases(self, mock_tokenizer):
        """Test various punctuation edge cases"""
        processor = DataProcessor(mock_tokenizer)

        # Multiple punctuation marks
        example = {"text": "Hello... World!!!"}
        cleaned = processor.clean_text(example)
        assert cleaned["text"] == "Hello... World!!!"

        # Punctuation at start/end
        example = {"text": ".Starting with period"}
        cleaned = processor.clean_text(example)
        assert cleaned["text"] == ". Starting with period"

        # Mixed punctuation and spaces
        example = {"text": "Test , . ; ! ? end"}
        cleaned = processor.clean_text(example)
        assert cleaned["text"] == "Test,.; !? end"
