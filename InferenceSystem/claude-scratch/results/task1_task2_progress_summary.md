# Model Port Progress Summary

**Date**: 2026-01-11
**Status**: Tasks 0-1 complete, Task 2 in progress (audio parity debugging)

## Completed

### Task 0: Environment Setup ✅
- Created `claude-scratch/requirements-model-v1.txt`
- Created `claude-scratch/SETUP_MODEL_V1.md`
- Created directory structure: `src/model_v1/`

### Task 1: Audio Frontend ✅ (unit tests pass)
- Created `src/model_v1/audio_frontend.py`
  - `load_audio()` - uses soundfile (same as patched torchaudio)
  - `downmix_to_mono()` - matches fastai_audio
  - `resample_audio()` - resamples to 20kHz
  - `pad_or_trim()` - pads/trims audio to 4 seconds
  - `compute_mel_spectrogram()` - uses sample_rate=16000 (critical quirk!)
  - `crop_spectrogram_to_duration()` - crops to 312 frames
  - `prepare_audio()` - full pipeline

### Task 2: Model Inference (in progress)
- Created `src/model_v1/resnet_model.py`
  - `OrcaResNet` class wrapping modified ResNet18
  - `create_resnet18_for_spectrograms()` - 1-channel input, 2-class output
- Created `src/model_v1/inference.py`
  - `OrcaDetectionModel` class matching FastAIModel interface

## Test Infrastructure

Created in `tests/`:
- `conftest.py` - fixtures + torchaudio patch
- `test_audio_preprocessing.py` - unit tests + parity tests
- `test_model_inference.py` - unit tests + parity tests
- `reference_outputs/` - for fastai reference spectrograms

### Test Results
```
Unit tests (audio): 5/5 PASS
Unit tests (model): 3/3 PASS
Parity tests: FAILING - value mismatch ~77 dB
```

## Key Issue: Audio Parity Not Achieved (ROOT CAUSE IDENTIFIED)

**Problem**: Spectrograms differ by ~77 dB max between model_v1 and fastai_audio.

**Shapes match** ✓ (312 frames) after adding `crop_spectrogram_to_duration()`.

**Root cause identified**: fastai's `tfm_pad_spectro` pads dB-scale spectrograms with 0.0 dB instead of minimum dB value.
- This is physically incorrect (0.0 dB = reference power, not silence)
- But it's what the model was trained on
- High-frequency bins (243-255) contain these 0.0 dB padding values
- Model_v1 crops (313→312) instead of padding, so doesn't execute the buggy padding

**Current status**:
- Median difference: 1.57 dB (most values match well)
- Sample-level accuracy in non-padded regions: 0.001-0.14 dB ✓
- ~50% of values differ by >10 dB (due to padding in high-frequency bins)
- Resampling fixed: using scipy.signal.resample_poly (matches fastai) ✓

**Decision needed**: Replicate fastai's bug for exact parity vs. keep correct implementation with relaxed tolerance.

See `claude-scratch/results/audio_parity_debugging_checkpoint.md` for full analysis and options.

## Files Created

```
InferenceSystem/
├── src/model_v1/
│   ├── __init__.py
│   ├── audio_frontend.py      # Audio preprocessing
│   ├── resnet_model.py        # PyTorch ResNet18
│   └── inference.py           # OrcaDetectionModel class
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Fixtures + torchaudio patch
│   ├── test_audio_preprocessing.py
│   ├── test_model_inference.py
│   └── reference_outputs/
├── claude-scratch/
│   ├── requirements-model-v1.txt
│   ├── SETUP_MODEL_V1.md
│   └── instructions/
│       └── model_port_hf.plan.md
```

## Next Steps

1. **Debug audio parity** - Compare intermediate values (raw audio, resampled audio) between fastai_audio and model_v1
2. **May need to match fastai_audio's resampling exactly** - Check if it uses a different resampling algorithm
3. Once audio parity achieved, test model inference parity
4. Create weight extraction script (Task 3)
5. Upload to HuggingFace (Task 4)
6. Update CI (Task 5)

## Commands to Run Tests

```bash
cd InferenceSystem
source inference-venv/bin/activate

# Unit tests (should pass)
python -m pytest tests/test_audio_preprocessing.py::TestAudioPreprocessingUnit -v
python -m pytest tests/test_model_inference.py::TestModelInferenceUnit -v

# Generate parity references (run in fastai env)
python -m pytest tests/test_audio_preprocessing.py::TestAudioPreprocessingParity::test_generate_reference_outputs -v

# Test parity (currently failing)
python -m pytest tests/test_audio_preprocessing.py::TestAudioPreprocessingParity::test_audio_parity_against_reference -v
```
