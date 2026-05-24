"""
Regression test for the Episodio272 speaker-boundary-inside-exclamation bug.

Bug: When the diarizer placed a speaker change inside a single Whisper
segment that was an unclosed Spanish exclamation, the sentence splitter
broke the construct in half, producing:

    "Así que, ¡empecemos."
    " Nate! Bueno, Andrea, antes de empezar..."

The expected behavior is to keep the inverted exclamation/question intact
and let the sentence break happen at the next valid boundary:

    "Así que, ¡empecemos, Nate!"
    "Bueno, Andrea, antes de empezar..."

Root cause: PRIORITY 1 (speaker boundary) in
``SentenceSplitter._should_end_sentence_here`` previously fired
unconditionally — unlike PRIORITY 4 (Whisper boundary, v0.6.2) which already
honored unclosed Spanish ``¡...!``/``¿...?`` constructs. The fix shifts
speaker boundaries that fall inside an unclosed inverted mark forward to
the word that closes the construct.
"""

import unittest

import pytest

from sentence_splitter import SentenceSplitter
from punctuation_restorer import _get_language_config


pytestmark = pytest.mark.core


class MockModel:
    """Mock SentenceTransformer model; we never reach semantic scoring in these tests."""

    def encode(self, texts):
        import numpy as np
        return np.random.rand(len(texts), 384)


class TestSpeakerBoundaryInsideSpanishExclamation(unittest.TestCase):
    """Speaker boundaries inside Spanish ``¡...!`` / ``¿...?`` must not split mid-construct."""

    def setUp(self):
        self.model = MockModel()
        self.es_config = _get_language_config('es')

    def test_episodio272_empecemos_nate_stays_together(self):
        """The exact Episodio272 case: speaker change between "¡empecemos," and "Nate!"."""
        splitter = SentenceSplitter('es', self.model, self.es_config)

        # Reproduces Whisper segments 19 + 20 from Episodio272_raw.txt
        text = (
            "Así que, ¡empecemos, Nate! "
            "Bueno, Andrea, antes de empezar, una pregunta que siempre estoy preguntándote."
        )

        # Diarization placed a speaker change inside segment 19 (the
        # exclamation): SPEAKER_01 ends after "¡empecemos," (word 2) and
        # SPEAKER_00 starts at "Nate!" (word 3). Segment 20 stays with
        # SPEAKER_00.
        # Whitespace-split word indices:
        #   0:"Así"  1:"que,"  2:"¡empecemos,"  3:"Nate!"
        #   4:"Bueno,"  5:"Andrea,"  6:"antes"  7:"de"  8:"empezar,"
        #   9:"una"  10:"pregunta"  11:"que"  12:"siempre"
        #   13:"estoy"  14:"preguntándote."
        speaker_segments = [
            {'start_word': 0, 'end_word': 3, 'speaker': 'SPEAKER_01'},   # "Así que, ¡empecemos,"
            {'start_word': 3, 'end_word': 15, 'speaker': 'SPEAKER_00'},  # "Nate! Bueno, ..."
        ]

        sentences, _ = splitter.split(text, speaker_segments=speaker_segments)

        joined = [s.text for s in sentences]

        # The exclamation must remain a single, balanced construct.
        for s in joined:
            opens_excl = s.count('¡')
            closes_excl = s.count('!')
            self.assertEqual(
                opens_excl, closes_excl,
                f"Unbalanced ¡/! in sentence {s!r} (sentences={joined})"
            )

        # No sentence should orphan the trailing "Nate!" by starting with it.
        for s in joined:
            self.assertFalse(
                s.lstrip().startswith('Nate'),
                f"Sentence starts with orphaned 'Nate!' fragment: {s!r}"
            )

        # The "¡empecemos" opener and its closing "Nate!" must appear in the
        # same sentence.
        found_intact = any(
            '¡empecemos' in s and 'Nate!' in s for s in joined
        )
        self.assertTrue(
            found_intact,
            f"Expected '¡empecemos' and 'Nate!' to share a sentence; got {joined}"
        )

    def test_speaker_boundary_inside_inverted_question_shifts_to_close(self):
        """Same protection applies to ``¿...?``: speaker change inside the question is deferred."""
        splitter = SentenceSplitter('es', self.model, self.es_config)

        # Two-word question "¿qué pasa?" with the diarizer placing a (likely
        # spurious) speaker change between "¿qué" and "pasa?". The opening
        # "¿" and closing "?" must end up in the same sentence.
        text = "Hola amigo, ¿qué pasa? Estoy aquí esperando tu respuesta."
        # Word indices:
        #   0:"Hola"  1:"amigo,"  2:"¿qué"  3:"pasa?"
        #   4:"Estoy"  5:"aquí"  6:"esperando"  7:"tu"  8:"respuesta."
        speaker_segments = [
            {'start_word': 0, 'end_word': 3, 'speaker': 'SPEAKER_00'},  # "Hola amigo, ¿qué"
            {'start_word': 3, 'end_word': 9, 'speaker': 'SPEAKER_01'},  # "pasa? Estoy aquí..."
        ]

        sentences, _ = splitter.split(text, speaker_segments=speaker_segments)

        joined = [s.text for s in sentences]

        for s in joined:
            self.assertEqual(
                s.count('¿'), s.count('?'),
                f"Unbalanced ¿/? in sentence {s!r} (sentences={joined})"
            )

        found_intact = any('¿qué' in s and 'pasa?' in s for s in joined)
        self.assertTrue(
            found_intact,
            f"Expected '¿qué' and 'pasa?' to share a sentence; got {joined}"
        )

    def test_normal_speaker_boundary_outside_marks_still_splits(self):
        """Sanity check: when no unclosed inverted mark is open, the speaker boundary still splits."""
        splitter = SentenceSplitter('es', self.model, self.es_config)

        text = "Hola soy Andrea de Colombia. Y yo soy Nate de Texas."
        # Speaker change after "Colombia."
        speaker_segments = [
            {'start_word': 0, 'end_word': 5, 'speaker': 'SPEAKER_00'},
            {'start_word': 5, 'end_word': 11, 'speaker': 'SPEAKER_01'},
        ]

        sentences, _ = splitter.split(text, speaker_segments=speaker_segments)

        # Should produce at least two sentences (one per speaker).
        self.assertGreaterEqual(
            len(sentences), 2,
            f"Expected speaker boundary outside unclosed marks to still split; got {[s.text for s in sentences]}"
        )


class TestShiftBoundaryHelper(unittest.TestCase):
    """Unit-level tests for ``_shift_boundary_past_unclosed_mark``."""

    def setUp(self):
        self.es_splitter = SentenceSplitter('es', None, _get_language_config('es'))
        self.en_splitter = SentenceSplitter('en', None, _get_language_config('en'))

    def test_shift_past_exclamation_close(self):
        words = "Así que, ¡empecemos, Nate! Bueno, Andrea,".split()
        # Boundary on "¡empecemos," (index 2). Closing "!" is at index 3.
        self.assertEqual(
            self.es_splitter._shift_boundary_past_unclosed_mark(2, words),
            3,
        )

    def test_shift_past_question_close(self):
        words = "Hola amigo, ¿qué pasa? Estoy bien.".split()
        # Boundary on "¿qué" (index 2). Closing "?" is at index 3.
        self.assertEqual(
            self.es_splitter._shift_boundary_past_unclosed_mark(2, words),
            3,
        )

    def test_no_shift_when_balanced(self):
        words = "Hola amigo. Estoy bien.".split()
        # No inverted marks at all → boundary unchanged.
        self.assertEqual(
            self.es_splitter._shift_boundary_past_unclosed_mark(1, words),
            1,
        )

    def test_no_shift_when_already_past_close(self):
        words = "Así que, ¡empecemos, Nate! Bueno,".split()
        # Boundary on "Nate!" (index 3) — construct already closed.
        self.assertEqual(
            self.es_splitter._shift_boundary_past_unclosed_mark(3, words),
            3,
        )

    def test_no_shift_for_non_spanish(self):
        words = "So, let's start, Nate! Well,".split()
        # English splitter: even though "Nate!" contains "!", we should not
        # interpret it as an unclosed inverted exclamation.
        self.assertEqual(
            self.en_splitter._shift_boundary_past_unclosed_mark(2, words),
            2,
        )

    def test_no_shift_when_closing_mark_missing(self):
        # Lookahead exhausted (no closing mark anywhere): leave boundary alone
        # rather than walking to the end of the document.
        words = ("¡empecemos " + ("palabra " * 40)).split()
        # Boundary in the middle of the run — no "!" ever appears.
        self.assertEqual(
            self.es_splitter._shift_boundary_past_unclosed_mark(5, words),
            5,
        )


if __name__ == '__main__':
    unittest.main()
