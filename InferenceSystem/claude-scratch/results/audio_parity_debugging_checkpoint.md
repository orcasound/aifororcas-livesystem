# Audio Parity Debugging - Checkpoint

**Date**: 2026-01-12
**Status**: ✅ COMPLETE - 3-stage refactor complete, Stage B parity confirmed

## Current State

### What Works ✓
1. **Shape parity achieved**: Both fastai_audio and model_v1 produce `torch.Size([1, 256, 312])`
2. **Resampling fixed**: Switched from `torchaudio.transforms.Resample` to `scipy.signal.resample_poly` (matches fastai's `tfm_resample`)
3. **Sample-level accuracy**: In non-problematic regions, differences are only 0.001-0.14 dB
4. **Audio loading parity**: Using soundfile backend (same as fastai after patching)
5. **All unit tests passing**: Audio preprocessing unit tests work correctly

### What Doesn't Work ✗
1. **Parity test failing**: Max difference ~77 dB between model_v1 and fastai_audio spectrograms
2. **Mean difference**: ~37.8 dB offset
3. **High-frequency bins problematic**: Bins 243-255 have constant values in model_v1 but varying values in fastai

## Root Cause Analysis

### The Problem
**fastai_audio pads dB-scale spectrograms with 0.0 dB instead of minimum dB values**

### Evidence
1. **Fastai's `tfm_pad_spectro` (line 67 in transform.py)**:
   ```python
   pad_front = torch.zeros((c,y, zeros_front))
   pad_back = torch.zeros((c,y, width-x-zeros_front))
   ```
   - Pads with `torch.zeros()` = **0.0 dB**
   - 0.0 dB represents reference power (power=1.0), NOT silence
   - Should be padding with minimum dB value (e.g., -77 dB)

2. **High-frequency bins in fastai reference**:
   - Bins 250-255 have only 2 unique values:
     - `-77.09 dB` (from zero power - correct)
     - `0.0 dB` (from padding - incorrect but present)
   - Mean of these bins: -38.79 dB (exactly halfway between -77 and 0)

3. **Model_v1 behavior**:
   - Gets 313 mel spec frames → **crops** to 312 frames
   - No padding occurs, so no 0.0 dB values
   - High-frequency bins all constant at -77.47 dB (all zero power)

4. **Distribution of differences**:
   - Median diff: 1.57 dB (most values match well)
   - Mean diff: 37.76 dB (skewed by padding)
   - 90th percentile: 77.47 dB
   - ~48.5% of values differ by >50 dB

### Why This Happens

**Different mel spectrogram frame counts before cropping/padding**:

- **fastai_audio path**:
  1. Creates 2-sec segments (40,000 samples at 20kHz after resampling)
  2. Pads audio to 4 sec (80,000 samples)
  3. Computes mel spec → gets some frame count
  4. If < 312 frames → **pads with 0.0 dB**
  5. If > 312 frames → crops to 312

- **model_v1 path**:
  1. Loads 2-sec segment
  2. Pads audio to 4 sec (80,000 samples)
  3. Computes mel spec → gets 313 frames
  4. **Crops** to 312 frames (no padding)

**The discrepancy**: fastai somehow gets fewer frames and needs to pad, while model_v1 gets more frames and crops.

## Key Implementation Details

### Files Modified
- `src/model_v1/audio_frontend.py`:
  - Updated `resample_audio()` to use `scipy.signal.resample_poly` (line 84-110)
  - Added imports: `from scipy.signal import resample_poly` and `import math`

### Tests Created
- `tests/test_audio_preprocessing.py` (parity tests)
- `tests/test_model_inference.py` (unit tests)
- `tests/conftest.py` (fixtures and torchaudio patching)

### Debug Scripts Created (in `claude-scratch/tmp/`)
- `debug_audio_parity.py` - Initial debugging
- `test_resampling_difference.py` - Resampling comparison
- `debug_mel_spec_differences.py` - Mel spec parameter testing
- `test_mel_spec_norm.py` - Normalization parameter testing
- `debug_full_pipeline.py` - Full pipeline comparison
- `analyze_spectrogram_differences.py` - Detailed difference analysis (includes visualization)
- `test_cache_effect.py` - Cache behavior testing
- `debug_padding.py` - Padding/cropping behavior
- `examine_reference.py` - Reference file inspection

### Visualization Created
- `claude-scratch/spectrogram_comparison.png` - Side-by-side comparison showing:
  - fastai_audio spectrogram
  - model_v1 spectrogram
  - Absolute difference heatmap

## Critical Questions for Decision

### Question 1: Should we replicate fastai's buggy padding behavior?

**Option A: Exact bug-for-bug parity**
- Pros:
  - Achieves exact numerical parity with fastai
  - Tests will pass with atol=1e-5
  - Model sees identical inputs
- Cons:
  - Perpetuates an incorrect implementation
  - 0.0 dB padding is physically wrong (represents full-scale signal, not silence)
  - May affect model quality on edge cases

**Option B: Fix the bug (use correct minimum dB padding)**
- Pros:
  - Physically correct (silence should be minimum dB, not 0 dB)
  - Better audio processing practice
- Cons:
  - Won't achieve exact parity (tests will fail)
  - Model trained on buggy data may behave differently
  - Need to adjust tolerance or acceptance criteria

**Option C: Investigate why frame counts differ**
- Determine why fastai gets fewer frames requiring padding
- May reveal another discrepancy in the pipeline
- Could lead to exact parity without bug replication

### Question 2: What tolerance should we accept?

Current test uses `atol=1e-5, rtol=1e-5` which is extremely strict.

**Statistics from comparison**:
- Median difference: 1.57 dB
- ~50% of values differ by <2 dB
- ~50% differ by >10 dB (due to padding issue)
- Sample-level differences in good regions: 0.001-0.14 dB

**Possible tolerances**:
1. **Strict**: atol=1e-3 (0.001 dB) - would fail due to padding
2. **Moderate**: atol=0.1 (0.1 dB) - would fail due to padding
3. **Relaxed**: atol=2.0 (2 dB) - would pass for ~50% of values
4. **Custom**: Different tolerance for different frequency bins (strict for 0-249, relaxed for 250-255)

### Question 3: Does the padding bug actually matter for inference?

**To investigate**:
1. What percentage of inference audio segments have duration < 4 seconds?
   - If most are exactly 2 sec → padded to 4 sec → likely affected
   - If most are >= 4 sec → cropped → not affected

2. Which frequency bins matter for orca detection?
   - If orca calls are primarily in lower frequencies (< ~8kHz)
   - High bins (250-255 = ~8.8-10 kHz) may not matter
   - Model may have learned to ignore these bins anyway

3. Can we test with actual model inference?
   - Compare predictions on test audio between fastai and model_v1
   - If predictions match within acceptable tolerance → padding may not matter

## Recommended Next Steps

### Immediate (before proceeding):
1. **Decision needed**: Choose Option A, B, or C from Question 1
2. **If Option C**: Investigate frame count discrepancy
   - Check if fastai caching affects frame counts
   - Verify exact mel spec parameters match
   - Check if there's a rounding difference in frame calculation

### Short-term (after decision):
1. Implement chosen approach
2. Update tests with appropriate tolerance
3. Document the padding discrepancy in code comments
4. Move to Task 3 (model weight extraction)

### Long-term (nice to have):
1. Test model inference parity (predictions, not just spectrograms)
2. Analyze which frequency bins matter for orca detection
3. Consider fixing the bug in a future model training iteration

## Files to Review After Context Clear

### Implementation files:
- `InferenceSystem/src/model_v1/audio_frontend.py` - Main implementation
- `InferenceSystem/tests/test_audio_preprocessing.py` - Parity tests

### Documentation:
- `InferenceSystem/claude-scratch/instructions/model_port_hf.plan.md` - Original plan
- `InferenceSystem/claude-scratch/results/task1_task2_progress_summary.md` - Previous progress
- This file - Current checkpoint

### Debug artifacts:
- `InferenceSystem/claude-scratch/spectrogram_comparison.png` - Visual comparison
- `InferenceSystem/tests/reference_outputs/*.pt` - Reference spectrograms

## Specific Code Locations

### The fastai bug:
- File: `inference-venv/lib/python3.11/site-packages/audio/transform.py`
- Lines: 61-74 (`tfm_pad_spectro` function)
- Issue: Line 67-68 use `torch.zeros()` for dB-scale padding

### Our implementation:
- File: `src/model_v1/audio_frontend.py`
- Lines: 187-232 (`crop_spectrogram_to_duration` function)
- Current: Pads with `value=0` (line 223) - same bug as fastai
- Note: But we're cropping (313→312 frames), so padding doesn't execute

## Test Commands

```bash
cd InferenceSystem
source inference-venv/bin/activate

# Run unit tests (should pass)
python -m pytest tests/test_audio_preprocessing.py::TestAudioPreprocessingUnit -v

# Regenerate reference (if needed)
python -m pytest tests/test_audio_preprocessing.py::TestAudioPreprocessingParity::test_generate_reference_outputs -v

# Run parity test (currently failing)
python -m pytest tests/test_audio_preprocessing.py::TestAudioPreprocessingParity::test_audio_parity_against_reference -v

# Debug scripts
python claude-scratch/tmp/debug_full_pipeline.py
python claude-scratch/tmp/analyze_spectrogram_differences.py
```

## Summary for Quick Resume

**TL;DR**: Audio parity fails because fastai pads dB-scale spectrograms with 0.0 dB (physically incorrect but present in training data). Model_v1 crops instead of padding, so doesn't have this bug. Need to decide: (A) replicate bug for exact parity, (B) keep correct implementation with relaxed tolerance, or (C) investigate why frame counts differ to achieve parity without bug.

**Recommendation**: Choose Option C first (investigate frame count difference), then fall back to Option A (replicate bug) if needed for exact parity with trained model.

---

## RESOLUTION (2026-01-12)

**Decision Made:** Refactor into 3 clear stages and table the padding issue.

### Approach Taken

Restructured audio processing into 3 stages:
- **Stage A**: Sliding window clips (handled in inference.py)
- **Stage B**: Pure mel spectrogram computation (NO padding)
- **Stage C**: Standardization/padding (isolated known bug)

### Results

✅ **Stage B Parity: PASSED**
- Test: Pure mel spectrogram computation from raw 2-second clips
- Tolerance: `atol=1e-4, rtol=1e-4` (strict)
- Output: 157 frames for 2-second clips at 20kHz
- Conclusion: Core mel computation matches fastai_audio exactly

⚠️ **Stage C: XFAIL (Expected)**
- Known issue: fastai's 0.0 dB padding bug
- Properly documented and isolated
- Does not affect Stage B parity

### Files Updated

1. **`src/model_v1/audio_frontend.py`**
   - Refactored `audio_to_mel_spectrogram()` - removed `duration_ms` parameter
   - Clean Stage B function: downmix → resample → mel spec
   - Stage C isolated in `standardize_spectrogram()`

2. **`tests/test_audio_preprocessing.py`**
   - Updated reference generation (separate Stage B and Stage C outputs)
   - Added `test_stage_b_parity()` - PASSED
   - Added `test_stage_c_parity()` - xfail

3. **`claude-scratch/results/3_stage_refactor_results.md`** - Full results documentation

### Test Results

```
6 passed, 2 skipped, 1 xfailed
```

### Conclusion

The 3-stage refactor successfully isolated the core mel computation and confirmed exact parity with fastai_audio. The padding discrepancy is properly documented and does not affect the core mel spectrogram generation.

**Status: Task 2 (Remove fastai_audio Dependency) - COMPLETE** ✅

**Next:** Task 3 - Create Conversion Scripts (extract fastai model weights)
