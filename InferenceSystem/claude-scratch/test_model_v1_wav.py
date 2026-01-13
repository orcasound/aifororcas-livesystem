#!/usr/bin/env python
"""Test model_v1 inference on a local WAV file and compare with fastai."""
import os
import sys
import yaml

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_v1.inference import OrcaHelloSRKWDetector

DEFAULT_WAV_PATH = os.path.join(os.path.dirname(__file__), "test_data", "rpi_sunset_bay_2025_09_18_01_12_06_PDT--f6b3fcd7-2036-433a-8a18-76a6b3b4f0c9.wav")


def main():
    # Default to the test WAV file in claude-scratch/test_data
    wav_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WAV_PATH

    # Model and config paths relative to InferenceSystem directory
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model_v1.pt')
    config_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_config.yaml')

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loading model from: {model_path}")
    print(f"Using config from: {config_path}")
    print(f"Testing WAV file: {wav_path}")
    print("-" * 60)

    # Load model
    model = OrcaHelloSRKWDetector.from_checkpoint(model_path, config)

    # Run inference
    result = model.detect_srkw_from_file(wav_path, config)

    # Print local predictions and confidences in aligned table format
    print("\nLocal Predictions (per 2-second segment):")
    print("-" * 60)
    print(f"{'Segment':<10} {'Start (s)':<12} {'Duration (s)':<14} {'Prediction':<12} {'Confidence':<12}")
    print("-" * 60)

    for i, (pred, conf, seg) in enumerate(zip(result.local_predictions, result.local_confidences, result.segment_predictions)):
        print(f"{i+1:<10} {seg.start_time_s:<12.1f} {seg.duration_s:<14.1f} {pred:<12} {conf:<12.3f}")

    print("-" * 60)

    # Summary statistics
    num_positive = sum(result.local_predictions)
    positive_segments = [i+1 for i, pred in enumerate(result.local_predictions) if pred == 1]

    print(f"\nSummary: {num_positive}/{len(result.local_predictions)} segments predicted positive")
    if positive_segments:
        print(f"Detected in segments: {positive_segments}")

    # global_confidence (float)
    print(f"\nglobal_confidence: {result.global_confidence:.3f}")

    # global_prediction (int)
    print(f"global_prediction: {result.global_prediction}")

    if result.global_prediction == 1:
        print("\n*** ORCA DETECTED! ***")
        return 0
    else:
        print("\nNo orca detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
