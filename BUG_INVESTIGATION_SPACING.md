# Bug Investigation: Missing Space After Periods (Bug #3)

**Date:** 2026-01-11  
**Status:** ✅ RESOLVED (v0.6.0.3)  
**Priority:** HIGH - Affects readability of transcripts

## Resolution Summary

**Root Cause Found:** The `_fix_mid_sentence_capitals()` function was consuming whitespace without preserving it.

**The Bug:** The regex pattern `r'([\s.,;:!?¿¡])\s*' + word + r'\b'` with replacement `r'\1' + word.lower()` matched whitespace with `\s*` but didn't capture it. The replacement only included group 1 (the punctuation), discarding the space.

**The Fix:** Changed the pattern to `r'([\s.,;:!?¿¡])(\s*)' + word + r'\b'` with replacement `r'\1\2' + word.lower()` to capture and preserve whitespace.

**Why the Safety Net Didn't Work:** The safety net regex WAS correctly adding spaces (". El" → ". El"), but then `_fix_mid_sentence_capitals()` was called AFTER and removed them (". El" → ".el").

---

---

## Problem Statement

Words are being concatenated without spaces after sentence-ending punctuation in the final output:

**Examples from Episodio213.txt:**
- Line 137: `"yo.el método"` should be `"yo. el método"`
- Line 141: `"dos.en vez"` should be `"dos. en vez"`
- Line 141: `"conversación.en la"` should be `"conversación. en la"`

**Full context of problematic lines:**
```
Line 137: Ella habla claro, así como yo.el método de enseñanza es como el mío.
Line 141: Ok, vamos para el método número dos.en vez de tener un tutor, pues puedes tener un compañero de conversación.
```

---

## Raw Whisper Segments (CORRECT)

From `Episodio213_raw.txt`:
```
Segment 126: 591.18s - 594.18s
Text: ella habla claro, así como yo.

Segment 127: 594.18s - 596.18s
Text: El método de enseñanza es como el mío.

Segment 129: 610.18s - 614.18s
Text: Ajá. Ok, vamos para el método número dos.

Segment 130: 614.18s - 620.18s
Text: En vez de tener un tutor, pues puedes tener un compañero de conversación.
```

**Key observation:** Whisper segments are SEPARATE and CORRECT. The concatenation happens in our pipeline.

---

## What We Know

### ✅ Works Correctly

1. **Unit Tests:** ALL 35 unit tests pass (including domain handling tests)
   ```bash
   docker exec $(docker ps -q | head -1) bash -c "cd /app && python tests/run_all_tests.py"
   # Result: All 35 tests pass ✓
   ```

2. **Isolated Regex Test:** The safety net regex works perfectly
   ```python
   import re
   test = 'yo.el método'
   result = re.sub(r'([.!?])([A-ZÁÉÍÓÚÑa-záéíóúñ¿¡])', r'\1 \2', test)
   # Result: 'yo. el método' ✓
   ```

3. **Full Processing Sequence Test:** Works correctly with exact problematic strings
   ```python
   import re
   from domain_utils import fix_spaced_domains
   
   test = 'Ella habla claro, así como yo.el método de enseñanza es como el mío.'
   # Step 1: Safety net
   after_safety = re.sub(r'([.!?])([A-ZÁÉÍÓÚÑa-záéíóúñ¿¡])', r'\1 \2', test)
   # Result: 'yo. el método' ✓
   
   # Step 2: Fix domains
   after_domains = fix_spaced_domains(after_safety, use_exclusions=True, language='es')
   # Result: Still 'yo. el método' ✓ (domains not affected)
   ```

4. **Domain Fix:** Domains ARE correctly preserved in production output
   ```
   Line 71: Spanish55.com ✓ (no unwanted space)
   Line 93: italki.com ✓ (no unwanted space)
   ```
   **This proves the updated code IS running in production!**

### ❌ Doesn't Work in Production

Running the full transcription still produces concatenated output:
```bash
docker exec $(docker ps -q | head -1) bash -c "cd /app && python podscripter.py audio-files/Episodio213.mp3 --output_dir audio-files --single --language es --enable-diarization"

# Output still contains:
# yo.el método
# dos.en vez
```

---

## Current Implementation

### Safety Net Location
File: `podscripter.py` function `_write_txt()` (lines ~480-495)

```python
# Single speaker sentence - write as one paragraph
s = (sentence_obj.text or "").strip()
if not s:
    continue

# SAFETY NET: Ensure space after sentence-ending punctuation
# This catches any concatenations that slipped through earlier stages
s = re.sub(r'([.!?])([A-ZÁÉÍÓÚÑa-záéíóúñ¿¡])', r'\1 \2', s)

# Fix domains AFTER safety net (removes incorrectly added spaces from domains)
s = fix_spaced_domains(s, use_exclusions=True, language=language)
s = _fix_mid_sentence_capitals(s)
s = _capitalize_first_letter(s)
f.write(f"{s}\n\n")
```

### Operation Order
1. Apply safety net regex (adds spaces after periods)
2. Fix spaced domains (removes spaces from real domains like "Spanish55.com")
3. Fix mid-sentence capitals
4. Capitalize first letter
5. Write to file

### Code Paths in _write_txt()
The safety net is applied in **4 places**:
1. Line ~416: Backward compatibility (string sentences)
2. Line ~467: Multi-speaker utterances (substantial splits)
3. Line ~476: Multi-speaker utterances (merged, too short to split)
4. Line ~487: Single speaker sentences ← **Most likely path for these sentences**

---

## The Mystery

### Contradiction
- ✅ Safety net regex works in isolation
- ✅ Full processing sequence works with exact strings
- ✅ Domains ARE fixed (proving updated code runs)
- ❌ But yo.el and dos.en are NOT fixed in production

### Questions

1. **Are these sentences even reaching `_write_txt()`?**
   - Added debug logging with `logger.error()` but got NO output
   - This suggests either:
     - Logging isn't working
     - These sentences take a different path
     - The bug happens BEFORE `_write_txt()`

2. **Could the concatenation happen earlier in the pipeline?**
   - If `sentence_obj.text` already contains "yo.el método" when the Sentence object is created
   - Then the safety net might not be matching for some reason

3. **Is there a character encoding issue?**
   - Maybe the period or letters in production have different Unicode codepoints?
   - Would explain why isolated tests work but production doesn't

4. **Is there caching somewhere?**
   - Deleted output file before test ✓
   - Cleared `__pycache__` ✓
   - Copied updated file to Docker ✓
   - Still getting old behavior

---

## What We've Tried

### Attempt 1: Safety Net in _assemble_sentences()
- Added regex AFTER whitespace normalization (line ~1148)
- Result: Didn't work, added debug logging but messages never appeared

### Attempt 2: Safety Net in _write_txt()  
- Added regex in all 4 code paths before writing
- Result: Works for domains but NOT for "yo.el" / "dos.en"

### Attempt 3: Reordering Operations
- Moved `fix_spaced_domains()` AFTER safety net
- Result: Domains fixed ✓, but bug still present ❌

### Attempt 4: Debug Logging
- Added `logger.info()` → No output
- Changed to `logger.error()` → Still no output (???)
- Searched for "FOUND PROBLEM" → grep exit code 1 (not found)

---

## File Locations

### Modified Files
- `podscripter.py`: Added safety net in `_write_txt()` function
- `CHANGELOG.md`: Documented in v0.6.0.2

### Test Files
- `audio-files/Episodio213.mp3`: Full episode (35+ minutes)
- `audio-files/Episodio213.txt`: Output file with bug
- `audio-files/Episodio213_raw.txt`: Raw Whisper output (correct, separate segments)
- `audio-files/Episodio213_diarization.txt`: Speaker diarization debug dump

### Relevant Code Sections
1. `podscripter.py` lines 405-495: `_write_txt()` function
2. `podscripter.py` lines 1140-1160: `_assemble_sentences()` whitespace normalization
3. `sentence_splitter.py` lines 635-660: Sentence creation (joins with `' '.join()`)
4. `domain_utils.py` lines 134-186: `fix_spaced_domains()` function

---

## Timeline of Bug Discovery

1. **Initial Fix (v0.6.0.2):** Added safety net regex in `_write_txt()`
2. **Test 1:** Bug still present
3. **Regression Found:** Safety net broke domains ("Spanish55. com")
4. **Fix:** Reordered operations (domains AFTER safety net)
5. **Test 2:** Domains fixed ✓, but original bug still present ❌
6. **Isolated Tests:** Confirmed regex works perfectly
7. **Current State:** Mystery why it works in isolation but not in production

---

## Next Steps to Investigate

### 1. Verify Text Encoding
```python
# In _write_txt(), before safety net:
s_bytes = s.encode('utf-8')
logger.error(f"Text bytes: {s_bytes}")
# Check if period is actually U+002E or something else
```

### 2. Check Sentence.text Creation
Look at where `Sentence(text=sentence_text, ...)` is created:
- `sentence_splitter.py` line ~655
- Is `sentence_text` already concatenated at creation time?

### 3. Trace Full Path
Add logging at EVERY stage:
- After `' '.join(current_chunk)` in sentence_splitter.py
- After `unmask_domains()` in sentence_splitter.py  
- In `SentenceFormatter.format()` before/after merges
- In `_write_txt()` at entry point

### 4. Check if Sentences Are Being Merged
```python
# In SentenceFormatter._merge_domains() or other merge functions
# Maybe domains are being "unmerged" but regular text is being incorrectly merged?
```

### 5. Test with Minimal Example
Create a minimal test case that ONLY processes these two specific segments:
```python
# segments = [
#     {'text': 'ella habla claro, así como yo.', 'start': 591.18, 'end': 594.18},
#     {'text': 'El método de enseñanza es como el mío.', 'start': 594.18, 'end': 596.18}
# ]
# Process these through the full pipeline and see where concatenation happens
```

---

## Environment

- **Docker Container:** Running updated code (confirmed by domain fix)
- **Python Version:** (check with `python --version` in container)
- **Test Command:**
  ```bash
  docker exec $(docker ps -q | head -1) bash -c "cd /app && python podscripter.py audio-files/Episodio213.mp3 --output_dir audio-files --single --language es --enable-diarization"
  ```
- **Test Duration:** ~35 minutes per full run

---

## Regex Pattern Details

### Current Pattern
```python
r'([.!?])([A-ZÁÉÍÓÚÑa-záéíóúñ¿¡])'
```

### What It Matches
- Group 1: Period, question mark, or exclamation mark
- Group 2: Any uppercase or lowercase letter (including Spanish accented characters and inverted punctuation)

### Replacement
```python
r'\1 \2'
```
Adds a space between the two groups.

### Test Cases That Work
```python
'yo.el' → 'yo. el' ✓
'dos.en' → 'dos. en' ✓  
'Spanish55.com' → 'Spanish55. com' then fixed back to 'Spanish55.com' by fix_spaced_domains() ✓
```

---

## Questions for Next Investigator

1. Why does the safety net work in isolation but not in production?
2. Why is debug logging with `logger.error()` not appearing?
3. Are these sentences even reaching `_write_txt()` or is there another output path?
4. Could there be a character encoding difference between test strings and production text?
5. Is `sentence_obj.text` already concatenated when the Sentence object is created?

---

## Contact

Original investigation by: AI Assistant (Claude Sonnet 4.5)  
Date: January 11, 2026  
Git branch: `fix/merge-refactor`  
Last commit: v0.6.0.2 (attempted fix)
