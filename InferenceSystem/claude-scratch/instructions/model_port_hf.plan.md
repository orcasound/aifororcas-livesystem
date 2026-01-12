# Model Port to HuggingFace Plan

**Last Updated**: 2026-01-12
**Current Status**: Task 2 - refactoring into 3-stage structure

## Overview

Port the orca detection model from fastai to HuggingFace transformers, removing the complicated fastai dependency while maintaining exact inference parity.

## Current State Analysis

### fastai_audio Usage (audio processing)
- **File**: `InferenceSystem/src/model/fastai_inference.py`
- **Imports**: `AudioConfig`, `SpectrogramConfig`, `AudioList` from `audio.data`
- **Spectrogram Parameters**:
  - `f_min=0.0`, `f_max=10000`
  - `hop_length=256`, `n_fft=2560`, `n_mels=256`
  - `to_db_scale=True`, `top_db=100`
  - `resample_to=20000` Hz
  - `duration=4000` ms (4 sec padding)

### fastai Model Inference
- **Loading**: `load_learner()` from `fastai.basic_train`
- **Model file**: `model.pkl` (production: `11-15-20.FastAI.R1-12`)
- **Architecture**: ResNet18 transfer learning
- **Input**: Mel spectrograms from 2-second audio segments
- **Output**: Confidence scores (0-1) aggregated to global prediction

### Existing Tests
- No pytest framework - integration tests via `LiveInferenceOrchestrator.py`
- Config-based tests in `config/Test/Positive/`, `Negative/`, `Fail/`
- CI runs on Windows + Ubuntu with Python 3.11

---

## Task 0: Create Separate Testing Environment

### Goal
Create an isolated virtual environment for testing the ported model without affecting the existing `inference-venv`.

### Implementation Steps

1. **Create new virtual environment**
   ```bash
   cd InferenceSystem
   python -m venv model-v1-venv
   source model-v1-venv/bin/activate  # Mac/Linux
   ```

2. **Create minimal requirements file** (`claude-scratch/requirements-model-v1.txt`)
   - Core dependencies only (no fastai):
     - `torch>=2.0`
     - `torchaudio>=2.0`
     - `transformers>=4.30`
     - `librosa>=0.10`
     - `numpy`
     - `pandas`
     - `pyyaml`

3. **Document setup** in `claude-scratch/SETUP_MODEL_V1.md`

---

## Task 1: Remove fastai_audio Dependency

### Goal
Rewrite audio preprocessing without fastai_audio, using only torchaudio/librosa.

### Analysis

**Current fastai_audio flow** (`fastai_inference.py:149-172`):
1. Load WAV files via `AudioList.from_folder()`
2. Resample to 20kHz
3. Downmix to mono
4. Pad/trim to 4 seconds
5. Compute mel spectrogram with specific parameters
6. Convert to dB scale

**Replacement approach**:
- Use `torchaudio` for loading and resampling
- Use `torchaudio.transforms.MelSpectrogram` for spectrogram generation
- Use `torchaudio.transforms.AmplitudeToDB` for dB conversion

### Implementation Steps

1. **Create audio frontend module** (`src/model_v1/audio_frontend.py`)
   - `load_audio(path)` - Load and resample WAV to 20kHz mono
   - `compute_mel_spectrogram(audio)` - Generate mel spectrogram with exact params
   - `prepare_input(wav_path)` - End-to-end pipeline

2. **Create parity tests** (`src/model_v1/tests/test_audio_parity.py`)
   - Load same audio file with both fastai_audio and new implementation
   - Compare intermediate outputs (raw audio, spectrograms)
   - Assert numerical closeness (atol=1e-5 or similar)

3. **Spectrogram parameters to match** (IMPORTANT: see quirk below):
   ```python
   # CRITICAL: fastai_audio uses sample_rate=16000 (default) NOT 20000!
   # See audio/data.py:46-48 - mel_args() excludes sample_rate
   MelSpectrogram(
       sample_rate=16000,  # Default! Not the actual 20kHz audio rate
       n_fft=2560,
       hop_length=256,
       n_mels=256,
       f_min=0.0,
       f_max=10000,
   )
   AmplitudeToDB(top_db=100)
   ```

   **Why this matters**: The `sample_rate` parameter affects mel filterbank
   frequency mapping. Using 16kHz when audio is 20kHz shifts frequency bins.
   Must replicate this for exact parity.

### Critical Files
- `InferenceSystem/src/model/fastai_inference.py` (reference)
- `InferenceSystem/src/audio/data.py` (fastai_audio source)
- **New**: `InferenceSystem/src/model_v1/audio_frontend.py`
- **New**: `InferenceSystem/src/model_v1/tests/test_audio_parity.py`

---

## Task 2: Remove fastai Dependency for Inference

### Goal
Replace fastai model loading/inference with pure PyTorch ResNet (no HuggingFace needed for ResNet).

### Analysis

**Current fastai model loading** (`fastai_inference.py:70-71`):
```python
from fastai.basic_train import load_learner
def load_model(mPath, mName="stg2-rn18.pkl"):
    return load_learner(mPath, mName)
```

**Model architecture**: fastai's `cnn_learner` with ResNet18 backbone
- Input: Mel spectrogram tensor (1 channel, variable time × 256 freq bins)
- Output: 2-class probabilities (whale call / not whale call)

**Inference call** (`fastai_inference.py:178`):
```python
predictions.append(self.model.predict(item)[2][1])
# Returns: (class_index, class_label, tensor([prob_class0, prob_class1]))
# [2][1] extracts the probability of class 1 (whale call)
```

### Implementation Steps

1. **Create weight extraction script** (`scripts/extract_fastai_weights.py`)
   - Load fastai learner
   - Extract the underlying PyTorch model: `learner.model`
   - Save state_dict to standard PyTorch format
   - Document model architecture details

2. **Create PyTorch model wrapper** (`src/model_v1/resnet_model.py`)
   - Use `torchvision.models.resnet18`
   - Modify first conv layer for 1-channel input (spectrograms)
   - Modify final fc layer for 2-class output
   - Load extracted weights

3. **Create inference class** (`src/model_v1/inference.py`)
   - `OrcaDetectionModel` class matching `FastAIModel` interface
   - `__init__(model_path, threshold, min_num_positive_calls_threshold)`
   - `predict(wav_file_path)` → same output format as fastai version

4. **Create parity tests** (`src/model_v1/tests/test_model_parity.py`)
   - Load same audio with both implementations
   - Compare intermediate outputs (spectrograms)
   - Compare final confidence scores
   - Assert closeness within 1e-5 tolerance

### Critical Implementation Details

**Spectrogram quirk to preserve**:
The fastai_audio `MelSpectrogram` is called WITHOUT `sample_rate` parameter:
```python
# audio/data.py:364 - mel_args() excludes sample_rate!
mel = MelSpectrogram(**(self.config.sg_cfg.mel_args()))(item.sig)
```
torchaudio's default `sample_rate=16000` is used even though audio is at 20kHz.
This affects mel filterbank calculation. Must replicate this for exact parity.

**ResNet18 modifications for audio**:
- `conv1`: Changed from (3, 64, 7×7) to (1, 64, 7×7) for single-channel spectrograms
- `fc`: Changed from 512→1000 to 512→2 for binary classification

### Critical Files
- `InferenceSystem/src/model/fastai_inference.py` (reference)
- `InferenceSystem/inference-venv/.../audio/data.py` (spectrogram creation)
- **New**: `InferenceSystem/src/model_v1/resnet_model.py`
- **New**: `InferenceSystem/src/model_v1/inference.py`
- **New**: `InferenceSystem/src/model_v1/tests/test_model_parity.py`
- **New**: `InferenceSystem/scripts/extract_fastai_weights.py`

---

## Task 3: Create Conversion Scripts

### Goal
- Checkpoint conversion script (fastai → PyTorch format)
- Standalone inference example for testing

### Implementation Steps

1. **Weight conversion script** (`scripts/convert_fastai_to_pytorch.py`)
   ```python
   # Load fastai learner
   learner = load_learner(model_path, model_name)

   # Extract PyTorch model
   pytorch_model = learner.model

   # Save state dict
   torch.save({
       'model_state_dict': pytorch_model.state_dict(),
       'model_architecture': str(pytorch_model),
       'classes': learner.data.classes,
   }, output_path)
   ```

2. **Standalone inference example** (`scripts/inference_example.py`)
   - Load a WAV file
   - Process through model_v1 pipeline
   - Print predictions in same format as current system
   - Useful for debugging and validation

3. **Batch inference script** (`scripts/batch_inference.py`)
   - Process multiple WAV files
   - Output CSV with results
   - Compare against fastai output

### Critical Files
- **New**: `InferenceSystem/scripts/convert_fastai_to_pytorch.py`
- **New**: `InferenceSystem/scripts/inference_example.py`
- **New**: `InferenceSystem/scripts/batch_inference.py`

---

## Task 4: Update HuggingFace Model Repo

### Goal
Upload converted model to https://huggingface.co/orcasound/orcahello-srkw-detect-v1

### Implementation Steps

1. **Prepare model files**
   - `pytorch_model.bin` - Extracted PyTorch weights
   - `config.json` - Model configuration (architecture, preprocessing params)
   - `preprocessor_config.json` - Audio preprocessing parameters

2. **Create model card** (`README.md`)
   - Model description and purpose
   - Architecture details (ResNet18)
   - Preprocessing pipeline specification
   - Example usage code
   - License (RAIL - ref: https://github.com/user-attachments/files/16329105/OrcaHello.Real.Time.Inference.System-RAIL.md)
   - Citation information

3. **Upload to HuggingFace**
   ```bash
   huggingface-cli login
   huggingface-cli upload orcasound/orcahello-srkw-detect-v1 ./model_files
   ```

### HuggingFace Model Structure
```
orcasound/orcahello-srkw-detect-v1/
├── README.md                   # Model card
├── pytorch_model.bin           # Model weights
├── config.json                 # Model config
├── preprocessor_config.json    # Audio preprocessing
└── example.py                  # Usage example
```

---

## Task 5: Update CI

### Goal
Add parity tests to CI workflow to ensure model_v1 produces identical results to fastai.

### Implementation Steps

1. **Add pytest framework** to `InferenceSystem/`
   - Create `pytest.ini` or configure in `pyproject.toml`
   - Add pytest to requirements

2. **Create test suite** (`src/model_v1/tests/`)
   - `test_audio_parity.py` - Audio preprocessing tests
   - `test_model_parity.py` - Model inference tests
   - `conftest.py` - Shared fixtures (test audio files)

3. **Update CI workflow** (`.github/workflows/InferenceSystem.yaml`)
   - Add step to run pytest for model_v1 tests
   - Ensure both Windows and Ubuntu coverage
   - Add parity comparison step

4. **Test coverage requirements**
   - Audio loading and resampling
   - Mel spectrogram generation
   - Model forward pass
   - End-to-end prediction
   - All test audio files in `config/Test/Positive/`

### CI Addition Example
```yaml
- name: Run model_v1 parity tests
  run: |
    python -m pytest src/model_v1/tests/ -v --tb=short
```

---

## Decisions Made

Based on user clarification:

1. **Numerical tolerance**: 1e-5 tolerance for floating point comparisons
2. **Integration**: model_v1 will be standalone for testing, NOT integrated into LiveInferenceOrchestrator
3. **HuggingFace version**: Upload to existing empty v1 repo (`orcasound/orcahello-srkw-detect-v1`)

---

## Progress Log

### 2026-01-11

**Completed:**
- Task 0: Environment setup ✅
- Task 1: Audio frontend created ✅ (unit tests pass)
- Task 2: Model wrapper created, tests created

**Current blocker:**
- Audio parity test failing (~77 dB difference)
- Shapes match (312 frames) after adding spectrogram cropping
- Values differ significantly - need to debug resampling differences

**Files created:**
- `src/model_v1/audio_frontend.py`
- `src/model_v1/resnet_model.py`
- `src/model_v1/inference.py`
- `tests/conftest.py`
- `tests/test_audio_preprocessing.py`
- `tests/test_model_inference.py`

**Key discovery:**
fastai_audio's `tfm_pad_spectro` uses random zero-padding. Fixed by using `pad_mode="zeros-after"` for deterministic testing.

**Root cause identified:**
- fastai's `tfm_pad_spectro` pads dB spectrograms with 0.0 dB (bug)
- High-frequency bins (243-255) affected
- Model_v1 crops instead of padding
- Sample-level accuracy in good regions: 0.001-0.14 dB
- Median difference: 1.57 dB

**Next step:**
Refactor audio processing into 3 clear stages (see below).

---

## Task 2 Update: 3-Stage Audio Processing Refactor

### Decision Made (2026-01-12)
Table the frame count difference for now. Restructure code into 3 stages and ensure parity tests pass for stages A and B.

### The 3 Stages

| Stage | Description | Input → Output |
|-------|-------------|----------------|
| **A** | Sliding window clips | 60s WAV → list of 2s clips (with 1s hop) |
| **B** | Mel spectrogram creation | 2s audio clip → mel spectrogram tensor |
| **C** | Standardization/padding | raw spectrogram → model-ready tensor (312 frames) |

### Implementation Steps

1. **Refactor `audio_frontend.py`** into 3 clear stage functions:
   - `audio_to_mel_spectrogram(audio, sr)` - Stage B: raw audio → mel spec (before standardization)
   - `standardize_spectrogram(spec, target_frames)` - Stage C: crop/pad to 312 frames

2. **Update reference generation** to save intermediate outputs at Stage B (before padding)

3. **Add stage-specific parity tests**:
   - Stage B: Compare overlapping frames with strict tolerance
   - Stage C: Mark as `xfail` (known padding bug)

4. **Regenerate references** & verify Stage B passes

### Stage B Parity Strategy

**Approach: Compare overlapping frames only**
- Let `N = min(model_v1_frames, fastai_frames)`
- Compare first N frames with strict numerical tolerance
- This tables the frame count difference while validating the core mel spectrogram computation

```python
def test_stage_b_parity(...):
    model_v1_spec = audio_to_mel_spectrogram(audio, sr)  # shape: (1, 256, 313)
    fastai_spec = reference["stage_b_spectrogram"]        # shape: (1, 256, 311)

    n_frames = min(model_v1_spec.shape[2], fastai_spec.shape[2])
    torch.testing.assert_close(
        model_v1_spec[:, :, :n_frames],
        fastai_spec[:, :, :n_frames],
        atol=1e-4, rtol=1e-4
    )
```

### Files to Modify

| File | Changes |
|------|---------|
| `src/model_v1/audio_frontend.py` | Refactor into 3 clear stage functions |
| `tests/test_audio_preprocessing.py` | Add stage-specific parity tests, update reference generation |
| `tests/reference_outputs/*.pt` | Will contain intermediate stage outputs |

### Expected Outcomes

1. **Stage A**: No parity test needed (just timestamp math)
2. **Stage B**: Should PASS - mel spectrogram computation matches on overlapping frames
3. **Stage C**: Known to fail - document the padding discrepancy, mark test as `xfail`
