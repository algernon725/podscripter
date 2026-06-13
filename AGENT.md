# AI Agent Guidelines for podscripter

This file is a lean hub. Detailed technical content lives in modular `.agent/` files:

- Pipeline, model caching, volume mounts, `SentenceFormatter`, chunking vs single-call, data flow, diarization -> [.agent/architecture/pipeline.md](.agent/architecture/pipeline.md)
- CLI flags, exit codes, logging levels, environment variables -> [.agent/architecture/cli-spec.md](.agent/architecture/cli-spec.md)
- Bug history, language edge cases, open limitations, testing quirks -> [.agent/troubleshooting/history.md](.agent/troubleshooting/history.md)
- System internals and data structures -> [ARCHITECTURE.md](ARCHITECTURE.md)
- Release history -> [CHANGELOG.md](CHANGELOG.md)
- Test corpus and bring-up -> [tests/README.md](tests/README.md)

> Curation policy: Keep this hub and `.agent/` docs lean. Do not auto-append or restructure them; propose concise edits for human review. The one sanctioned exception is appending ledger-style bug entries to [history.md](.agent/troubleshooting/history.md) (see checklist).

## Project Overview

**podscripter** is a multilingual audio transcription tool that generates accurate, punctuated transcriptions for language-learning platforms like LingQ. It uses Docker containerization and state-of-the-art NLP models for punctuation restoration.

### Core technologies
- **Whisper** (faster-whisper): OpenAI speech-to-text for transcription.
- **Sentence-Transformers**: semantic understanding and punctuation restoration.
- **Hugging Face Hub (caches)**: used by `sentence-transformers`; managed via `HF_HOME` and optional offline mode.
- **spaCy (mandatory)**: capitalization and entity awareness; models baked into the Docker image (`en_core_web_sm`, `es_core_news_sm`, `fr_core_news_sm`, `de_core_news_sm`).
- **pyannote.audio 4.0.4**: optional speaker diarization.
- **Docker**: reproducible environments.
- **Python 3.10+**: primary language.

### Supported languages
- Primary focus: English (en), Spanish (es), French (fr).
- German (de) is experimental (was previously primary). German-specific code paths (preposition guards, auxiliary verbs, past-participle detection, greeting commas, capitalization) and tests remain, so `--language de` and Whisper auto-detect still work. See [history: language edge cases](.agent/troubleshooting/history.md#language-edge-cases).
- Other languages may work via Whisper auto-detect but are experimental.

## Architectural Principles

1. **Container-first design**: all development and testing inside Docker; dependencies via the Dockerfile; model caching via Docker volumes. See [pipeline: volume mounts](.agent/architecture/pipeline.md#volume-mounts).
2. **Model caching strategy**: use `HF_HOME` (avoid deprecated `TRANSFORMERS_CACHE`); singleton loaders; offline when warm. See [pipeline: model caching](.agent/architecture/pipeline.md#model-caching-strategy).
3. **Modular processing pipeline**: Audio -> Chunking -> Whisper -> Dedup/Globalize -> Punctuation -> Sentence Splitting -> Post-processing Merge -> Output. See [pipeline: data flow](.agent/architecture/pipeline.md#data-flow).
4. **Unified splitting and formatting**: `SentenceSplitter` owns boundaries; `SentenceFormatter` owns merges (never merges different speakers). See [pipeline: post-processing](.agent/architecture/pipeline.md#post-processing-formatting-sentenceformatter).

## Coding Style & Standards

- Follow PEP 8; descriptive names; comprehensive docstrings; explicit imports (no wildcards).
- File organization: core logic in `punctuation_restorer.py`; orchestration in `podscripter.py`; boundaries in `sentence_splitter.py`; merges in `sentence_formatter.py`; domain utilities in `domain_utils.py`; tests in `tests/`.
- Error handling: raise typed exceptions at the source, handle centrally in the CLI. See [cli-spec: exit codes](.agent/architecture/cli-spec.md#exit-codes-and-typed-exceptions).
- Logging: single `podscripter` logger; levels via CLI flags. See [cli-spec: logging](.agent/architecture/cli-spec.md#logging).
- Punctuation: use the centralized `_should_add_terminal_punctuation()` rather than scattered `text += '.'`. See [pipeline: centralized punctuation](.agent/architecture/pipeline.md#centralized-punctuation-system).

## Development Workflow

### Feature development
- Start with a focused test to define the requirement.
- Implement with proper error handling.
- Test across all supported languages.
- Update documentation as needed.

### Bug fixes
- Reproduce with a minimal `test_[specific_issue].py`.
- Prefer general solutions over specific hacks; document the reasoning.
- Use `assert` with descriptive messages; use `@pytest.mark.parametrize` for multi-input logic; test positive and negative cases.
- Verify across all supported languages.
- See [history: testing quirks](.agent/troubleshooting/history.md#testing-quirks) for markers and how to run the suite.

### Testing
- Run tests inside Docker with model caches mounted; use `python3` (not `python`).
- Create focused test files; test both components and the full pipeline.

## Common Pitfalls to Avoid

- **Environment**: don't run tests outside Docker; don't use deprecated env vars; don't forget cache mounts.
- **Code quality**: don't create one-off fixes for individual sentences; don't hardcode language rules unnecessarily; don't ignore punctuation edge cases; don't add scattered period insertion.
- **Testing**: don't skip multi-language testing; don't create overly broad test files; don't ignore Docker requirements.

## Checklist for AI Agents

Before submitting any changes:

- [ ] Code runs inside the Docker container
- [ ] Tests pass across all supported languages
- [ ] No deprecated environment variables used
- [ ] Model caching properly configured
- [ ] Documentation updated if needed
- [ ] Fixes are general, not specific hacks
- [ ] Error handling included for edge cases
- [ ] Code follows the project's architectural patterns
- [ ] Undocumented quirk fixed? Append a ledger bullet to [history.md Resolved bugs](.agent/troubleshooting/history.md#resolved-bugs) in the existing format: `- <Bug> (Fixed vX.X.X): <technical cause>; functions changed. Tests: test_x.py.`

## Project Goals

- **Accurate transcription** from audio.
- **Proper punctuation** across languages.
- **Language-learning support** (e.g., LingQ).
- **Ease of use** via simple Docker setup.

Quality standards: high accuracy across languages, consistent punctuation, reliable reproducible results, simple setup.

---

This project prioritizes accuracy, maintainability, and ease of use. Always consider the impact of changes across all supported languages and the overall user experience.
