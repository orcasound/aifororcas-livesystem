#!/usr/bin/env python
"""Compare model_v1 and fastai inference results."""
import os
import sys
import yaml

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_v1.inference import OrcaHelloSRKWDetectorV1
from model.fastai_inference import FastAIModel

DEFAULT_WAV_PATH = os.path.join(os.path.dirname(__file__), "test_data", "rpi_sunset_bay_2025_09_18_01_12_06_PDT--f6b3fcd7-2036-433a-8a18-76a6b3b4f0c9.wav")


def main():
    # Default to the test WAV file
    wav_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WAV_PATH

    # Paths
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    model_v1_path = os.path.join(model_dir, 'model_v1.pt')
    config_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_config.yaml')

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Testing WAV file: {wav_path}")
    print("=" * 80)

    # Run fastai inference
    print("\n[1/2] Running fastai inference...")
    fastai_model = FastAIModel(
        model_path=model_dir,
        model_name="model.pkl",
        threshold=0.5,
        min_num_positive_calls_threshold=3
    )
    fastai_result = fastai_model.predict(wav_path)

    # Run model_v1 inference
    print("[2/2] Running model_v1 inference...")
    model_v1 = OrcaHelloSRKWDetectorV1.from_checkpoint(model_v1_path, config)
    v1_result = model_v1.detect_srkw_from_file(wav_path, config)

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Compare lengths
    fastai_len = len(fastai_result['local_predictions'])
    v1_len = len(v1_result.local_predictions)
    print(f"\nNumber of segments:")
    print(f"  fastai:   {fastai_len}")
    print(f"  model_v1: {v1_len}")
    if fastai_len != v1_len:
        print(f"  ⚠ WARNING: Different number of segments!")

    # Compare global results
    print(f"\nGlobal Prediction:")
    print(f"  fastai:   {fastai_result['global_prediction']}")
    print(f"  model_v1: {v1_result.global_prediction}")
    if fastai_result['global_prediction'] == v1_result.global_prediction:
        print(f"  ✓ MATCH")
    else:
        print(f"  ✗ MISMATCH")

    print(f"\nGlobal Confidence:")
    print(f"  fastai:   {fastai_result['global_confidence']:.3f}")
    print(f"  model_v1: {v1_result.global_confidence:.3f}")
    diff = abs(fastai_result['global_confidence'] - v1_result.global_confidence)
    print(f"  diff:     {diff:.3f}")

    # Compare local predictions
    min_len = min(fastai_len, v1_len)
    mismatches = []
    large_diffs = []

    for i in range(min_len):
        fastai_conf = fastai_result['local_confidences'][i]
        v1_conf = v1_result.local_confidences[i]
        conf_diff = abs(fastai_conf - v1_conf)

        fastai_pred = fastai_result['local_predictions'][i]
        v1_pred = v1_result.local_predictions[i]

        if fastai_pred != v1_pred:
            mismatches.append((i+1, fastai_pred, v1_pred, fastai_conf, v1_conf, conf_diff))

        if conf_diff > 0.05:  # 5% threshold
            large_diffs.append((i+1, fastai_conf, v1_conf, conf_diff))

    print(f"\nLocal Predictions:")
    print(f"  Total segments compared: {min_len}")
    print(f"  Prediction mismatches: {len(mismatches)}")
    print(f"  Confidence diffs > 0.05: {len(large_diffs)}")

    if mismatches:
        print(f"\n  Prediction Mismatches:")
        print(f"  {'Seg':<6} {'fastai_pred':<12} {'v1_pred':<10} {'fastai_conf':<13} {'v1_conf':<11} {'diff':<8}")
        print(f"  {'-'*70}")
        for seg, fp, vp, fc, vc, d in mismatches[:10]:  # Show first 10
            print(f"  {seg:<6} {fp:<12} {vp:<10} {fc:<13.3f} {vc:<11.3f} {d:<8.3f}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches)-10} more")

    if large_diffs:
        print(f"\n  Large Confidence Differences (> 0.05):")
        print(f"  {'Seg':<6} {'fastai_conf':<13} {'v1_conf':<11} {'diff':<8}")
        print(f"  {'-'*40}")
        for seg, fc, vc, d in large_diffs[:15]:  # Show first 15
            print(f"  {seg:<6} {fc:<13.3f} {vc:<11.3f} {d:<8.3f}")
        if len(large_diffs) > 15:
            print(f"  ... and {len(large_diffs)-15} more")

    # Overall stats
    all_diffs = [abs(fastai_result['local_confidences'][i] - v1_result.local_confidences[i])
                 for i in range(min_len)]
    mean_diff = sum(all_diffs) / len(all_diffs)
    max_diff = max(all_diffs)

    print(f"\nConfidence Statistics:")
    print(f"  Mean absolute difference: {mean_diff:.4f}")
    print(f"  Max absolute difference:  {max_diff:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
