# Licenses and attribution for audio fixtures

This file is the consolidated NOTICE for every third-party audio source redistributed in the
[`podscripter-project/test-fixtures`](https://huggingface.co/datasets/podscripter-project/test-fixtures)
HuggingFace dataset. It is also the source-of-truth allowlist used by
[`_validate_licensing.py`](_validate_licensing.py).

## Aggregate dataset license

The aggregate test-fixtures dataset is licensed as **CC-BY 4.0** — the most restrictive license
among its components (CC0 + CC-BY 4.0). Downstream users redistributing the dataset must comply
with CC-BY 4.0 (attribution, license notice, indication of changes, no DRM).

License text: <https://creativecommons.org/licenses/by/4.0/legalcode>

## Permissive-license allowlist

Only the following licenses are accepted for new fixtures. The validator
([`_validate_licensing.py`](_validate_licensing.py)) rejects any other value:

| License value | Notes |
|---|---|
| `CC-BY-4.0` | Requires `attribution` and `modifications` fields in the fixture. |
| `CC0-1.0` | Attribution recommended but not required. |
| `public-domain` | Reserved for genuinely PD sources (e.g., US government recordings). |

Excluded (do not add fixtures from these sources):

- `CC-BY-NC-*` — NonCommercial clauses incompatible with open redistribution
- `CC-BY-ND` / `CC-BY-NC-ND` — NoDerivatives forbids trimming
- TED-LIUM, MuST-C (both CC-BY-NC-ND)
- Spotify Podcast Dataset (research-only)
- Any source requiring registration or per-user license acceptance to redistribute

## Source corpora attribution

### LibriSpeech (CC-BY 4.0)

- **Source**: <https://www.openslr.org/12>
- **License**: CC-BY 4.0 — <https://creativecommons.org/licenses/by/4.0/>
- **Attribution**: V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: An ASR corpus
  based on public domain audio books", IEEE ICASSP 2015.
- **Modifications**: Individual FLAC files extracted from the `test-clean` and `dev-clean`
  splits; converted to 16 kHz mono WAV; clipped per fixture as documented in each
  `.expected.json` `modifications` field.

### Common Voice (CC0 1.0)

- **Source**: <https://commonvoice.mozilla.org/en/datasets>
- **License**: CC0 1.0 Universal — <https://creativecommons.org/publicdomain/zero/1.0/>
- **Attribution**: Mozilla Common Voice (CC0; attribution courteous but not required).
- **Modifications**: Single-utterance clips extracted from public splits; converted to 16 kHz
  mono WAV.

### VoxPopuli (CC0 1.0)

- **Source**: <https://github.com/facebookresearch/voxpopuli>
- **License**: CC0 1.0 Universal — <https://creativecommons.org/publicdomain/zero/1.0/>
- **Attribution**: C. Wang et al., "VoxPopuli: A Large-Scale Multilingual Speech Corpus for
  Representation Learning, Semi-Supervised Learning and Interpretation", ACL 2021.
- **Modifications**: Multi-speaker segments extracted from European Parliament debates; clipped
  per fixture as documented in each `.expected.json` `modifications` field.

### AMI Meeting Corpus (CC-BY 4.0)

- **Source**: <https://groups.inf.ed.ac.uk/ami/corpus/>
- **License**: CC-BY 4.0 — <https://creativecommons.org/licenses/by/4.0/>
- **Attribution**: J. Carletta et al., "The AMI Meeting Corpus: A pre-announcement", MLMI 2005.
- **Modifications**: Mix-Headset recordings clipped to per-fixture lengths and trim windows as
  documented in each `.expected.json` `modifications` field.

### VoxConverse (CC-BY 4.0)

- **Source**: <https://www.robots.ox.ac.uk/~vgg/data/voxconverse/>
- **License**: CC-BY 4.0 — <https://creativecommons.org/licenses/by/4.0/>
- **Attribution**: J. S. Chung et al., "Spot the conversation: speaker diarisation in the wild",
  Interspeech 2020.
- **Modifications**: Dev-set recordings clipped to per-fixture lengths and trim windows as
  documented in each `.expected.json` `modifications` field. Verbatim transcripts produced
  separately (VoxConverse ships RTTM speaker turns only, not text).

### Multilingual LibriSpeech (MLS, CC-BY 4.0)

- **Source**: <https://www.openslr.org/94/>
- **License**: CC-BY 4.0 — <https://creativecommons.org/licenses/by/4.0/>
- **Attribution**: V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, R. Collobert, "MLS: A Large-Scale
  Multilingual Dataset for Speech Research", Interspeech 2020.
- **Modifications**: French and Spanish audiobook clips extracted from the test split's
  per-speaker/per-book OPUS files; transcoded to 16 kHz mono signed-16-bit WAV (short
  fixtures) or 16 kHz mono FLAC (long fixtures). Long multi-speaker ES/FR fixtures are
  built off-repo by concatenating utterances from two distinct MLS test speakers with
  0.5 s silence between speakers, then re-encoding as a single FLAC.

### FLEURS (CC-BY 4.0)

- **Source**: <https://huggingface.co/datasets/google/fleurs>
- **License**: CC-BY 4.0 — <https://creativecommons.org/licenses/by/4.0/>
- **Attribution**: A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia, J. Riesa,
  C. Rivera, A. Bapna, "FLEURS: Few-shot Learning Evaluation of Universal Representations
  of Speech", arXiv:2205.12446, 2022.
- **Modifications**: Single-utterance WAV files extracted from the `en_us` / `es_419` / `fr_fr`
  test splits' `audio/test.tar.gz`; re-encoded as 16 kHz mono signed-16-bit WAV. No trimming.

## Per-fixture attribution

Every fixture's `.expected.json` independently declares its `source`, `license`,
`attribution`, and `modifications`. The dataset's HF README mirrors the same per-clip
table. This file is the human-readable summary; the JSON files are the machine-readable
source of truth that the validator enforces on every PR.
