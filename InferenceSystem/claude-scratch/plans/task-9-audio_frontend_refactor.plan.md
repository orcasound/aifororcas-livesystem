# Plan: Audio Frontend Refactor — Move Preprocessing into audio_segment_generator

## Context

Task 8 complete. This is a new refactor task for `src/model_v1/audio_frontend.py`.

**Goal**: Move waveform-level preprocessing (downmix + resample) so it is performed by `audio_segment_generator` (on the full audio before segmenting) rather than by `load_audio` (per segment after the fact). Rename `load_audio` to reflect that it is now the waveform preprocessing function.

**Approach**: Option A — generator loads + resamples full audio, exports already-20kHz WAV segments. `load_audio` is renamed to `load_processed_waveform`. Yielded interface of `audio_segment_generator` stays `(str, float, float)`.

---

## Changes

### Change 1: Rename `load_audio` → `load_processed_waveform`

**File**: `src/model_v1/audio_frontend.py`

Rename the function and update its docstring to reflect that it is now used for waveform-level preprocessing (downmix + resample). Keep the signature identical:

```python
def load_processed_waveform(file_path: str, audio_config: Dict) -> Tuple[torch.Tensor, int]:
```

Update all internal call sites within `audio_frontend.py`:
- `prepare_audio()` (line ~282): `load_audio(...)` → `load_processed_waveform(...)`

---

### Change 2: `audio_segment_generator` — accept `audio_config` and preprocess before segmenting

**File**: `src/model_v1/audio_frontend.py`

Add `audio_config: Optional[Dict] = None` parameter to `audio_segment_generator`.

When `audio_config` is provided:
1. Call `load_processed_waveform(audio_file_path, audio_config)` to get `(waveform, sr)` — this downmixes + resamples the entire file
2. Write the resampled waveform to a temp WAV at the target sample rate using `soundfile`
3. Use that resampled temp WAV as the source for pydub segmenting (instead of the original file)
4. `get_duration` should be called on the resampled temp WAV (or computed from waveform shape / target_sr)

When `audio_config` is None (default), behavior is identical to current — raw file is segmented as-is. This preserves backward compatibility for callers that don't pass `audio_config`.

Yielded interface is unchanged: `(segment_path: str, start_s: float, end_s: float)`.

---

### Change 3: `AudioPreprocessor.process_segments` — pass `audio_config` to generator

**File**: `src/model_v1/audio_frontend.py`

Update `AudioPreprocessor.process_segments()` to pass `self.config.audio.as_dict()` (or equivalent) as `audio_config` to `audio_segment_generator`. This ensures the full pipeline uses preprocessed segments.

---

### Change 4: Update `test_audio_preprocessing.py` — rename import

**File**: `tests/test_audio_preprocessing.py`

Three unit tests import `load_audio` directly and call it on a WAV file for testing sub-stages B/C. Update these to use the new name `load_processed_waveform`:

```python
# Before
from model_v1.audio_frontend import load_audio
waveform, sr = load_audio(sample_1min_wav, v1_config["audio"])

# After
from model_v1.audio_frontend import load_processed_waveform
waveform, sr = load_processed_waveform(sample_1min_wav, v1_config["audio"])
```

Affected tests: `test_load_audio`, `test_featurize_waveform`, `test_standardize`, `test_stage_b_parity` (4 call sites).

**The rest of the test file stays untouched** — `audio_segment_generator`, `prepare_audio`, `featurize_waveform`, `standardize` interfaces are unchanged.

---

### Change 5: Update `scripts/run_inference.py` and `scripts/test_audio_segments.py` if needed

Scan both scripts for any direct calls to `load_audio`. If found, update to `load_processed_waveform`. (Both scripts currently only call `detect_srkw_from_file` or `prepare_audio` — likely no changes needed.)

---

## Critical Files

| File | Change |
|------|--------|
| `src/model_v1/audio_frontend.py` | Rename `load_audio`, add `audio_config` param to `audio_segment_generator`, update `AudioPreprocessor.process_segments` |
| `tests/test_audio_preprocessing.py` | Update 4 import/call sites from `load_audio` → `load_processed_waveform` |
| `scripts/run_inference.py` | Check for `load_audio` calls (likely none) |
| `scripts/test_audio_segments.py` | Check for `load_audio` calls (likely none) |

---

## Key Design Notes

- **Temp WAV for resampled audio**: When `audio_config` is provided, `audio_segment_generator` writes a temp WAV of the resampled waveform. This can be done in the existing `_temp_segment_dir` context or a separate temp file. The resampled WAV is deleted after generation (or just placed in the existing temp dir).
- **`get_duration` on segments**: Since segments are exported from an already-resampled source, `load_processed_waveform` on each segment WAV will be a no-op resample (orig_sr == target_sr). This is the efficiency win.
- **No change to `prepare_audio`**: It still calls `load_processed_waveform` internally on segment WAVs. Since segments are already at target SR, the resample step is skipped.
- **`audio_config` in `audio_segment_generator` is optional**: All existing callers (tests, `test_audio_segments.py`) that don't pass `audio_config` get the old behavior unchanged.

---

## Workspace Docs

Per CLAUDE.md: copy this plan to `InferenceSystem/claude-scratch/plans/task-9-audio_frontend_refactor.plan.md` as first action during implementation.

---

## Verification

```bash
cd InferenceSystem
source model-v1-venv/bin/activate

# Run unit tests
python -m pytest tests/test_audio_preprocessing.py -v

# Run test_audio_segments on FLAC (should still produce 5 segments + spectrograms)
python scripts/test_audio_segments.py /Users/Akash/SideProjects/ai4orcas/DORI-SRKW/benchmark/test/ICLISTENHF1266_20230111T122000.045Z.flac

# Run inference
python scripts/run_inference.py /Users/Akash/SideProjects/ai4orcas/DORI-SRKW/benchmark/test/ICLISTENHF1266_20230111T122000.045Z.flac
```

Expected: all existing tests pass, FLAC inference still works.
