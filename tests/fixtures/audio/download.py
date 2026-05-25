"""Downloader for audio fixtures hosted on HuggingFace.

The audio used by `tests/test_audio_fixtures.py` lives in the public dataset
`podscripter-project/test-fixtures` on HF, pinned to a specific revision (commit hash)
so audio and `.expected.json` files can never drift apart unintentionally.

Usage:

    from tests.fixtures.audio.download import ensure_fixtures, resolve_audio_path

    fixtures_root = ensure_fixtures()         # idempotent; downloads on first call only
    audio = resolve_audio_path("en/foo.wav")  # absolute Path inside HF cache

The cache lives under `HF_HOME` (default `/root/.cache/huggingface` in the Docker container,
mounted from `models/huggingface/` on the host), so no additional volume mounts are required
beyond what podscripter already documents.

Bumping the dataset:

    1. Push new clips to the HF dataset.
    2. Look up the new commit hash (the dataset's "Files and versions" tab).
    3. Update `HF_REVISION` below.
    4. Commit the change together with the new `.expected.json` files in the same PR.

The pinned revision keeps PRs reproducible: a checkout of any commit in the podscripter repo
fetches exactly the audio that matches its tests.
"""

from __future__ import annotations

import os
from pathlib import Path

# The HF dataset that hosts the audio. Public; no auth required for downloads.
HF_DATASET = "podscripter-project/test-fixtures"

# Pin a specific dataset revision (commit hash) so audio matches the test code exactly.
# IMPORTANT: bump this whenever a new clip is added to the dataset, in the SAME PR as the
# new `.expected.json` files referencing those clips.
HF_REVISION = "6b6296b2e33a232af3c3df7782cf025e11515a34"

# Repo type for HF Hub helpers. Datasets and models use different URL prefixes.
_HF_REPO_TYPE = "dataset"


def ensure_fixtures(
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Ensure the audio fixtures dataset is present in the HF cache.

    Idempotent: subsequent calls are no-ops once the snapshot is cached. Honors
    `HF_HUB_OFFLINE=1` (will not hit the network and will raise if the snapshot is missing).

    Returns the absolute path to the cached snapshot root. Audio files live under
    `<snapshot>/en/`, `<snapshot>/fr/`, etc.
    """
    from huggingface_hub import snapshot_download

    rev = revision or HF_REVISION
    patterns = allow_patterns or ["*.wav", "*.flac", "*.mp3", "README.md"]

    return Path(
        snapshot_download(
            repo_id=HF_DATASET,
            repo_type=_HF_REPO_TYPE,
            revision=rev,
            cache_dir=cache_dir,
            allow_patterns=patterns,
        )
    )


def resolve_audio_path(
    audio_file: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Resolve `<snapshot>/<audio_file>` to an absolute path, downloading if needed."""
    snapshot = ensure_fixtures(revision=revision, cache_dir=cache_dir, allow_patterns=[audio_file])
    path = snapshot / audio_file
    if not path.exists():
        raise FileNotFoundError(
            f"Audio file {audio_file!r} not found in HF dataset snapshot at {snapshot}. "
            f"It may not have been uploaded yet, or the pinned HF_REVISION may need to be "
            f"bumped after a recent dataset update."
        )
    return path


def _main() -> int:
    """CLI for manual cache priming or CI pre-warm: `python -m tests.fixtures.audio.download`."""
    rev = os.environ.get("PODSCRIPTER_FIXTURES_REVISION") or HF_REVISION
    snapshot = ensure_fixtures(revision=rev)
    print(f"podscripter test fixtures cached at: {snapshot}")
    print(f"  dataset:  {HF_DATASET}")
    print(f"  revision: {rev}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
