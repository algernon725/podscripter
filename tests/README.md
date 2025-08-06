# Punctuation Restoration Tests

This directory contains all test scripts for the punctuation restoration functionality.

## Test Files

- **`test_improved_punctuation.py`** - Tests the improved punctuation restoration with various languages
- **`test_model_change.py`** - Verifies the model change to paraphrase-multilingual-MiniLM-L12-v2
- **`test_spanish_questions.py`** - Comprehensive test for Spanish question detection
- **`test_spanish_introductions.py`** - Tests Spanish introductions and statements (should NOT be questions)
- **`model_comparison.py`** - Compares different SentenceTransformer models
- **`run_all_tests.py`** - Test runner to execute all tests

## Running Tests

### Run All Tests
```bash
# From the project root directory
docker run --rm -v $(pwd):/app podscripter python tests/run_all_tests.py
```

### Run Individual Tests
```bash
# From the project root directory
docker run --rm -v $(pwd):/app podscripter python tests/test_spanish_introductions.py
docker run --rm -v $(pwd):/app podscripter python tests/test_improved_punctuation.py
docker run --rm -v $(pwd):/app podscripter python tests/model_comparison.py
```

## Test Categories

### Core Functionality Tests
- **Punctuation Restoration**: Tests the main punctuation restoration functionality
- **Model Performance**: Compares different SentenceTransformer models
- **Language Support**: Tests multilingual capabilities

### Spanish-Specific Tests
- **Question Detection**: Ensures Spanish questions are properly detected
- **Introduction Handling**: Verifies Spanish introductions are NOT incorrectly detected as questions
- **Statement Recognition**: Tests that Spanish statements are properly handled

## Expected Results

- **Spanish Introductions**: Should be 100% accurate (no false question detection)
- **Question Detection**: Should be >90% accurate for real questions
- **Model Performance**: Should show clear differences between models 