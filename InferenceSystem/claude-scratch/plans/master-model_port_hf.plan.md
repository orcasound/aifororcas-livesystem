# Model Port to HuggingFace - Master Plan

**Last Updated**: 2026-01-13
**Branch**: `akash/inference-v1-nofastai`
**Status**: Tasks 0-4.5 Complete, Ready for HuggingFace Upload (Task 5)

## Overview

Port the OrcaHello SRKW detection model from fastai to pure PyTorch, removing the fastai dependency to enable HuggingFace upload and simplified deployment.

**Key Discovery**: The model is ResNet50 (not ResNet18 as named in a few places) with a custom classification head.

---

## Task Summary

| Task | Description | Status | Detailed Plan |
|------|-------------|--------|---------------|
| 0 | Environment Setup | COMPLETE | - |
| 1 | Audio Frontend | COMPLETE | - |
| 2 | Audio Parity Testing | COMPLETE | - |
| 2.5 | Audio Interface Cleanup | COMPLETE | `audio_interface_cleanup.plan.md` |
| 3 | Weight Extraction | COMPLETE | `task3_weight_extraction.plan.md` |
| 4 | Full-File Inference | COMPLETE | - |
| 4.5 | Investigate/Fix Differences | **COMPLETE** | `segment_count_investigation.md` |
| 5 | HuggingFace Upload | NOT STARTED | (create when starting) |
| 6 | CI Integration | NOT STARTED | (create when starting) |

---

## Task 0: Environment Setup (COMPLETE)

**Goal**: Create isolated Python 3.11 environment without fastai for testing the ported model.

**Key Files**:
- `model-v1-venv/` - New environment (no fastai)
- `inference-venv/` - Legacy environment (has fastai, for reference)

**Outcome**: Two separate virtual environments allow testing model_v1 independently while maintaining fastai for reference generation.

---

## Task 1: Audio Frontend (COMPLETE)

**Goal**: Rewrite audio preprocessing using torchaudio/librosa instead of fastai_audio.

**Key Files**:
- `src/model_v1/audio_frontend.py` - Config-driven mel spectrogram generation

**Outcome**: Clean audio processing pipeline with configurable parameters (sample_rate, n_mels, hop_length, etc.) matching fastai_audio behavior.

---

## Task 2: Audio Parity Testing (COMPLETE)

**Goal**: Verify new audio frontend produces identical spectrograms to fastai_audio.

**Key Files**:
- `tests/test_audio_preprocessing.py` - Stage-based parity tests
- `tests/reference_outputs/` - FastAI reference spectrograms

**Outcome**: Stage B (mel spectrogram) and Stage C (standardization) pass within tolerance. Known difference: fastai pads with 0 dB (a bug), model_v1 crops to 312 frames.

---

## Task 2.5: Audio Interface Cleanup (COMPLETE)

**Goal**: Refactor audio processing into clear 3-stage pipeline for maintainability.

**Key Files**:
- `src/model_v1/audio_frontend.py` - Refactored with clear stage separation

**Detailed Plan**: `plans/task-2_5-audio_interface_cleanup.plan.md`

**Outcome**: Clean separation between sliding window (A), spectrogram creation (B), and standardization (C).

---

## Task 3: Weight Extraction (COMPLETE)

**Goal**: Extract PyTorch weights from fastai learner and create standalone checkpoint.

**Key Files**:
- `scripts/extract_fastai_weights.py` - One-time conversion script
- `model/model_v1.pt` - Extracted PyTorch checkpoint (98MB)
- `src/model_v1/inference.py` - OrcaHelloSRKWDetectorV1 class

**Detailed Plan**: `plans/task-3-weight_extraction.plan.md`

**Key Discovery**: Model is ResNet50 with custom head:
- AdaptiveConcatPool2d with [max_pool, avg_pool] order
- Head: BN(4096) → Linear(512) → BN(512) → Linear(2)
- Class 1 = "positive" (orca call detected)

**Outcome**: `predict_call()` produces identical results to fastai within 1e-5 tolerance.

---

## Task 4: Full-File Inference (COMPLETE)

**Goal**: Implement end-to-end inference on WAV files matching fastai behavior.

**Key Files**:
- `src/model_v1/inference.py` - `detect_srkw_from_file()` method
- `scripts/test_local_wav_model_v1.py` - WAV file testing
- `scripts/compare_inference.py` - FastAI vs model_v1 comparison

**Outcome**:
- Global predictions match (both detect orca on test data)
- 91.4% segment agreement (differences due to fastai's rolling window smoothing)
- 10 unit tests passing

---

## Task 4.5: Investigate/Fix Differences (COMPLETE)

**Goal**: Investigate and resolve differences between model_v1 and fastai full-file inference before HuggingFace upload.

**Outcome**: Successfully resolved segment count difference by adding `strict_segments` parameter.

**Key Changes**:
- Added `strict_segments` boolean to `InferenceConfig` (default: `True`)
- Modified `audio_segment_generator()` to support partial final segments
- `strict_segments=False` matches FastAI's 59 segments
- `strict_segments=True` (default) generates only complete segments (58)

**Parity Results** (After FastAI bugfix - 2026-01-13):

Strict Mode (default, 58 segments):
- Global prediction: ✓ MATCH
- Segment agreement: **94.8%** (3 mismatches)
- Mean confidence diff: 5.3%

Non-Strict Mode (59 segments):
- Segment count: ✓ MATCH (59 vs 59)
- Global prediction: ✓ MATCH
- Segment agreement: **93.2%** (4 mismatches)
- Mean confidence diff: 6.5%

**Decision**: Keep `strict_segments=True` as default for production (consistent segment durations). Use `strict_segments=False` for FastAI parity testing.

**Key Achievement**: Excellent parity - global predictions match, >93% segment agreement

**Detailed Results**: `results/segment_count_investigation.md`

---

## Task 5: HuggingFace Upload (NOT STARTED)

**Goal**: Package and upload model to `orcasound/orcahello-srkw-detector-v1` on HuggingFace Hub.

**Planned Deliverables**:
- `pytorch_model.bin` or `model.safetensors` - Model weights
- `config.json` - Model architecture and parameters
- `preprocessor_config.json` - Audio preprocessing parameters
- `README.md` - Model card with usage examples

**Key Considerations**:
- Follow HuggingFace audio model conventions
- Include preprocessing config for reproducibility
- Document the sample_rate quirk (16kHz filterbank on 20kHz audio)
- Add license (RAIL)

---

## Task 6: CI Integration (NOT STARTED)

**Goal**: Add model_v1 tests to CI pipeline and prepare for deployment.

**Planned Deliverables**:
- Update `.github/workflows/InferenceSystem.yaml` with pytest for model_v1
- Docker image with model_v1 (no fastai dependency)
- AKS deployment updates

**Key Considerations**:
- Tests should run without fastai (use saved references)
- Consider separate CI job for model_v1 vs legacy fastai
- Gradual migration path for production

---

## Quick Reference

### Model Architecture
```
ResNet50 backbone (3,4,6,3 Bottleneck blocks)
  ↓
AdaptiveConcatPool2d [max, avg] → 4096 features
  ↓
BatchNorm1d(4096) → Dropout(0.25) → Linear(512) → ReLU
  ↓
BatchNorm1d(512) → Dropout(0.5) → Linear(2)
  ↓
Softmax → [P(negative), P(positive)]
```

### Key Interfaces
```python
# Load model
model = OrcaHelloSRKWDetectorV1.from_checkpoint("model/model_v1.pt", config=config)

# Single spectrogram inference
prob = model.predict_call(spectrogram)  # Returns P(positive)

# Full file inference
result = model.detect_srkw_from_file(wav_path, config)
# Returns DetectionResult with local/global predictions
```

### Test Commands
```bash
cd InferenceSystem
source model-v1-venv/bin/activate

# Run unit tests
python -m pytest tests/test_model_inference.py -v

# Test on WAV file
python scripts/test_local_wav_model_v1.py

# Compare with fastai (requires inference-venv)
source inference-venv/bin/activate
python scripts/compare_inference.py
```

---

## Decisions Made

1. **Numerical tolerance**: 1e-5 for floating point comparisons of local model predictions
2. **Integration**: model_v1 is standalone, NOT integrated into LiveInferenceOrchestrator
3. **HuggingFace**: Upload to existing `orcasound/orcahello-srkw-detector-v1` repo
4. **Padding behavior**: model_v1 crops and pads model inputs with a standarize() function that matches fastai's buggy 0 dB padding
5. **Smoothing in file level inference**: model_v1 uses raw confidences (fastai uses rolling window smoothing)
