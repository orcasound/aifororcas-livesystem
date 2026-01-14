# Checkpoint: Model V1 Port to PyTorch

**Date**: 2026-01-13
**Branch**: `akash/inference-v1-nofastai`
**Status**: Task 4.5 Complete - Ready for HuggingFace upload

---

## Quick Status

| Task | Description | Status |
|------|-------------|--------|
| 0 | Environment setup | COMPLETE |
| 1 | Audio frontend | COMPLETE |
| 2 | Audio parity testing | COMPLETE |
| 2.5 | Audio interface cleanup | COMPLETE |
| 3 | Weight extraction | COMPLETE |
| 4 | Full-file inference | COMPLETE |
| 4.5 | **Investigate/fix differences** | **IN PROGRESS** |
| 5 | HuggingFace upload | NOT STARTED |
| 6 | CI integration | NOT STARTED |

---

## Current Focus: Investigating Differences (Task 4.5)

After Task 4 completion, some cleanup was done and now investigating/fixing the differences between model_v1 and fastai before proceeding to HuggingFace upload.

### Known Differences from FastAI
1. **Rolling window smoothing**: FastAI applies rolling window smoothing to confidences; model_v1 uses raw per-segment confidences
2. **Segment count**: FastAI generates 59 segments, model_v1 generates 58 (final segment handling)
3. **Confidence values**: Mean diff ~6.3% due to smoothing differences
4. **Segment predictions**: 5 mismatches out of 58 (91.4% agreement)

### Investigation Goals
- Determine root cause of segment count difference
- Decide whether to replicate fastai's rolling window smoothing
- Ensure global predictions match for production use cases

---

## Model Architecture

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

**Key Discovery**: Model is ResNet50 (NOT ResNet18 as originally documented)

---

## Key Files

### Model Files
| File | Description |
|------|-------------|
| `model/model.pkl` | Original fastai learner (98MB) |
| `model/model_v1.pt` | Converted PyTorch checkpoint (98MB) |

### Source Code
| File | Description |
|------|-------------|
| `src/model_v1/inference.py` | OrcaHelloSRKWDetectorV1 class |
| `src/model_v1/audio_frontend.py` | Audio preprocessing pipeline |
| `tests/test_model_inference.py` | Unit + parity tests (10 passing) |
| `tests/conftest.py` | Test fixtures |
| `tests/test_config.yaml` | Config parameters |

### Test Scripts (scripts/)
| File | Description |
|------|-------------|
| `scripts/test_local_wav_model_v1.py` | Test model_v1 on WAV files |
| `scripts/compare_inference.py` | Compare fastai vs model_v1 |
| `scripts/test_local_wav.py` | Original fastai test script |

---

## Key Interfaces

```python
# Load model with config
model = OrcaHelloSRKWDetectorV1.from_checkpoint(
    "model/model_v1.pt",
    config=config
)

# Single spectrogram inference
prob = model.predict_call(spectrogram)  # Returns P(positive)

# Full file inference
result = model.detect_srkw_from_file(wav_path, config)
# Returns: DetectionResult(
#   local_predictions: List[int],
#   local_confidences: List[float],
#   global_prediction: int,
#   global_confidence: float,
#   segment_predictions: List[SegmentPrediction],
#   wav_file_path: str
# )
```

---

## Virtual Environments

```bash
# Legacy environment (has fastai) - for reference generation
source inference-venv/bin/activate

# New environment (no fastai) - for model_v1 testing
source model-v1-venv/bin/activate
```

---

## Test Commands

```bash
cd InferenceSystem

# Run unit tests (no fastai needed)
source model-v1-venv/bin/activate
python -m pytest tests/test_model_inference.py -v

# Test on WAV file
python scripts/test_local_wav_model_v1.py

# Compare with fastai
source inference-venv/bin/activate
python scripts/compare_inference.py

# Regenerate reference (needs fastai)
python -m pytest tests/test_model_inference.py::TestPredictCallParity::test_generate_reference -v
```

---

## Completed Task Summaries

### Task 3: Weight Extraction
- Created `scripts/extract_fastai_weights.py` for one-time conversion
- Discovered ResNet50 architecture (not ResNet18)
- Fixed pooling order: [max_pool, avg_pool]
- Fixed class index: class 1 = "positive"
- Parity test passes within 1e-5 tolerance

### Task 4: Full-File Inference
- Implemented `detect_srkw_from_file()` method
- Added config dict support
- Global predictions match fastai on test data
- Created comparison scripts for validation

---

## Git Status

```
Branch: akash/inference-v1-nofastai
Latest commits:
- 269c8fc rename orcahello-srkw-detector-v1
- 45a957a rename Orca**DectectorV1
- b1507be cleanup config handling in OrcaHelloSRKWDetector
- de26cbd feat: implement full-file inference for model_v1
- 99fa64c Add model_v1 PyTorch inference with fastai parity testing
```

---

## Related Documentation

- **Master Plan**: `plans/model_port_hf.plan.md`
- **Task 3 Details**: `plans/task3_weight_extraction.plan.md`
- **Task 4 Results**: `results/task4_full_file_inference.md`
- **Comparison Output**: `results/inference_comparison.txt`

---

**Last Updated**: 2026-01-13
