# Task 8: FLAC Audio Frontend Support

**Date**: 2026-02-11
**Branch**: `akash/inference-v1-nofastai`
**Status**: IN PROGRESS

---

## Goal

Enable the audio frontend to accept high-sample-rate FLAC files (and any other ffmpeg-supported audio format) in addition to WAV.

**Test file**: `/Users/Akash/SideProjects/ai4orcas/DORI-SRKW/benchmark/test/ICLISTENHF1266_20230111T122000.045Z.flac`
- 24-bit FLAC, mono, 128 kHz, 300 seconds (~5 minutes)

---

## Root Cause

`audio_segment_generator` in `src/model_v1/audio_frontend.py` used `AudioSegment.from_wav()`, which only reads WAV files (Python's built-in `wave` module). FLAC and other formats raise `wave.Error`.

The fix is minimal: replace `from_wav()` with `from_file()`, which routes non-WAV formats through ffmpeg.

---

## Changes

### 1. `src/model_v1/audio_frontend.py` (1 line)

Line 349:
```python
# Before
audio = AudioSegment.from_wav(audio_file_path)

# After
audio = AudioSegment.from_file(audio_file_path)
```

No other changes needed:
- `librosa.get_duration(path=...)` already uses soundfile backend (handles FLAC)
- Segment export stays as WAV (correct for downstream `load_audio → sf.read()`)
- `load_audio` already uses soundfile which handles FLAC natively

### 2. `scripts/run_inference.py` (renamed from `test_local_wav_model_v1.py`)

- Renamed to reflect generic audio support (WAV, FLAC, etc.)
- Updated docstring, variable names (`wav_path` → `audio_path`), print messages
- No logic changes — `detect_srkw_from_file` was already format-agnostic after the frontend fix

---

## Test

```bash
cd InferenceSystem
source model-v1-venv/bin/activate

# Section test: segment generation from FLAC
python claude-scratch/tmp/test_flac_segments.py

# End-to-end inference on FLAC
python scripts/run_inference.py /Users/Akash/SideProjects/ai4orcas/DORI-SRKW/benchmark/test/ICLISTENHF1266_20230111T122000.045Z.flac
```

**Expected from test_flac_segments.py**:
- pydub: `frame_rate=128000, sample_width=3, channels=1`
- 5 WAV segment files in `claude-scratch/tmp/flac_segments/`
- `load_audio` returns shape `(1, ~1200000)` at 20 kHz
- All checks PASS

---

## Prerequisites

- ffmpeg must be installed (`brew install ffmpeg` on Mac)
- `model-v1-venv` with `pydub`, `soundfile`, `librosa` installed
