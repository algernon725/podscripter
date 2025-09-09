#!/usr/bin/env python3
import os
import sys
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import punctuation_restorer as pr  # noqa: E402
from tests.spanish_samples import SPANISH_ASR_SEGMENTS, HUMAN_REFERENCE_TEXT  # noqa: E402


def _normalize(s: str) -> str:
    import re
    s = s.strip()
    # Normalize line breaks - replace multiple newlines with single space
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"\s+", " ", s)
    # Normalize quotes and punctuation spacing
    s = re.sub(r"([.!?])\s+", r"\1 ", s)
    return s.lower()


def score_similarity(generated: str, reference: str) -> dict:
    import re
    from difflib import SequenceMatcher
    gen = _normalize(generated)
    ref = _normalize(reference)
    ratio = SequenceMatcher(None, gen, ref).ratio()
    # Token-level F1 (rough)
    gen_tokens = gen.split()
    ref_tokens = ref.split()
    common = sum((min(gen_tokens.count(t), ref_tokens.count(t)) for t in set(gen_tokens + ref_tokens)))
    precision = common / max(1, len(gen_tokens))
    recall = common / max(1, len(ref_tokens))
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    # Sentence-level alignment: split by single newlines for both texts
    # This gives us individual sentences rather than paragraphs
    gen_sentences = [s.strip() for s in generated.split('\n') if s.strip()]
    ref_sentences = [s.strip() for s in reference.split('\n') if s.strip()]
    gen_norm = [_normalize(s) for s in gen_sentences]
    ref_norm = [_normalize(s) for s in ref_sentences]
    matched = 0
    used = set()
    for s in gen_norm:
        for idx, r in enumerate(ref_norm):
            if idx in used:
                continue
            # consider it a match if high character overlap
            from difflib import SequenceMatcher as SM
            if SM(None, s, r).ratio() >= 0.70:
                matched += 1
                used.add(idx)
                break
    sent_precision = matched / max(1, len(gen_norm))
    sent_recall = matched / max(1, len(ref_norm))
    sent_f1 = 0.0 if (sent_precision + sent_recall) == 0 else 2 * sent_precision * sent_recall / (sent_precision + sent_recall)
    return {
        "ratio": ratio,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sent_precision": sent_precision,
        "sent_recall": sent_recall,
        "sent_f1": sent_f1,
    }


def test_spanish_transcription_scoring():
    # Use the full podscripter pipeline to simulate real transcription
    from podscripter import _assemble_sentences
    
    # Concatenate ASR segments like the real pipeline does
    all_text = ' '.join(SPANISH_ASR_SEGMENTS)
    
    # Run through the full pipeline including sanitization
    sentences = _assemble_sentences(all_text, 'es', quiet=True)
    
    generated_text = "\n\n".join(sentences)

    reference_text = HUMAN_REFERENCE_TEXT

    metrics = score_similarity(generated_text, reference_text)
    print("Spanish transcription similarity:", metrics)

    # Loose thresholds to guard against regressions while allowing incremental improvement
    # Note: Program puts each sentence on its own line, so sentence counts differ from reference
    assert metrics["ratio"] > 0.30  # Character-level similarity
    assert metrics["f1"] > 0.70     # Token-level F1 should be good
    # Sentence-level alignment is challenging due to different sentence grouping
    assert metrics["sent_f1"] > 0.10  # Very loose threshold for sentence-level matching

"""
Focused unit tests for Spanish helper utilities in punctuation_restorer.py

Covers:
- Tag question normalization
- Collocation repairs
- Possessive/aux+gerund/capitalized-split merges
- Inverted question pairing
- Greeting/lead-in commas
- Imperative exclamation wrapping
"""


def test_es_normalize_tag_questions():
    # Unchanged when ending with dot at end-of-string
    assert pr._es_normalize_tag_questions("Está bien, ¿no.") == "Está bien, ¿no."
    # Normalizes missing comma before tag
    assert pr._es_normalize_tag_questions("Está bien ¿verdad?") == "Está bien, ¿verdad?"


def test_es_fix_collocations():
    # Repair "por? supuesto" and add comma when sentence-initial
    s = "por? supuesto vamos a empezar."
    out = pr._es_fix_collocations(s)
    assert out.startswith("Por supuesto,")


def test_es_merge_possessive_splits():
    assert pr._es_merge_possessive_splits("tu. Español") == "tu español"
    assert pr._es_merge_possessive_splits("Mi. Amigo") == "Mi amigo"


def test_es_merge_aux_gerund():
    # Helper preserves inner capitalization; full pipeline may lowercase later
    assert pr._es_merge_aux_gerund("Estamos. Hablando") == "Estamos Hablando"


def test_es_merge_capitalized_one_word_sentences():
    assert pr._es_merge_capitalized_one_word_sentences("Estados. Unidos.") == "Estados Unidos."


def test_es_pair_inverted_questions():
    assert pr._es_pair_inverted_questions("¿Dónde está.") == "¿Dónde está?"
    assert pr._es_pair_inverted_questions("Cómo estás?") == "¿Cómo estás?"


def test_es_greeting_and_leadin_commas():
    # Moves period to comma before a following question; does not rewrite content
    assert pr._es_greeting_and_leadin_commas("Hola como estan. ¿Listos?") == "Hola como estan, ¿Listos?"
    assert pr._es_greeting_and_leadin_commas("Hola, a todos") == "Hola a todos"
    assert pr._es_greeting_and_leadin_commas("Como siempre vamos a revisar") == "Como siempre, vamos a revisar"
    # New: do not introduce double comma when greeting already ends with comma
    assert pr._es_greeting_and_leadin_commas("Hola para todos, ¿Cómo están?") == "Hola para todos, ¿Cómo están?"
    # New: add comma before inverted mark when missing
    assert pr._es_greeting_and_leadin_commas("Hola para todos ¿Cómo están?") == "Hola para todos, ¿Cómo están?"


def test_es_wrap_imperative_exclamations():
    assert pr._es_wrap_imperative_exclamations("Vamos a empezar.") == "¡Vamos a empezar!"
    assert pr._es_wrap_imperative_exclamations("Bienvenidos a Españolistos.") == "¡Bienvenidos a Españolistos!"


def test_es_mid_sentence_exclamation_closure():
    # Should close exclamation when '¡' appears after a lead-in
    s = "Entonces, ¡empecemos."
    out = pr._spanish_cleanup_postprocess(s)
    assert out.endswith("¡empecemos!") or out == "Entonces, ¡empecemos!"
    # Other variants
    s2 = "Bueno, ¡vamos."
    out2 = pr._spanish_cleanup_postprocess(s2)
    assert out2.endswith("¡vamos!") or out2 == "Bueno, ¡vamos!"


if __name__ == "__main__":
    # Run tests directly
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All Spanish helper tests passed")


