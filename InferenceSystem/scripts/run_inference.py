#!/usr/bin/env python
"""Run model_v1 inference on a local audio file (WAV, FLAC, or any ffmpeg-supported format).

Usage:
    python scripts/run_inference.py                          # uses default WAV test file
    python scripts/run_inference.py path/to/audio.wav
    python scripts/run_inference.py path/to/audio.flac
    python scripts/run_inference.py path/to/audio.wav --output results.json
"""
import argparse
import json
import os
import sys
from dataclasses import asdict

import yaml

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from torch.backends.mps import is_available as is_torch_mps_available
from model_v1.inference import OrcaHelloSRKWDetectorV1

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_data')
DEFAULT_AUDIO_PATH = os.path.join(TEST_DATA_DIR, "rpi_sunset_bay_2025_09_18_01_12_06_PDT--f6b3fcd7-2036-433a-8a18-76a6b3b4f0c9.wav")


def main():
    parser = argparse.ArgumentParser(description="Run model_v1 inference on audio files")
    parser.add_argument("audio_path", nargs="?", default=DEFAULT_AUDIO_PATH,
                        help="Path to audio file (WAV, FLAC, or ffmpeg-supported format)")
    parser.add_argument("--output", "-o", help="Output JSON file path for serialized results")
    args = parser.parse_args()

    audio_path = args.audio_path

    # Model and config paths relative to InferenceSystem directory
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model_v1.pt')
    config_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'config.yaml')

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loading model from: {model_path}")
    print(f"Using config from: {config_path}")
    print(f"Testing audio file: {audio_path}")
    print("-" * 60)

    # Load model
    model = OrcaHelloSRKWDetectorV1.from_checkpoint(model_path, config)
    if is_torch_mps_available():
        print("Using MPS backend for inference")
        model = model.to("mps")

    # Run inference
    result = model.detect_srkw_from_file(audio_path, config)

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
    positive_segment_times = [result.segment_predictions[i-1].start_time_s for i in positive_segments]

    print(f"\nSummary: {num_positive}/{len(result.local_predictions)} segments predicted positive")
    if positive_segments:
        print(f"Detected in segments: {positive_segments}")
        print(f"Detected in times: {positive_segment_times}")

    print(f"\nglobal_confidence: {result.global_confidence:.3f}")
    print(f"global_prediction: {result.global_prediction}")

    # Print performance info
    meta = result.metadata
    print(f"\n--- Performance ---")
    print(f"File duration:    {meta.file_duration_s:.2f}s")
    print(f"Processing time:  {meta.processing_time_s:.2f}s")
    print(f"Realtime factor:  {meta.realtime_factor:.2f}x")

    # Serialize to JSON if requested
    if args.output:
        result_dict = asdict(result)
        # Add computed realtime_factor to metadata
        result_dict['metadata']['realtime_factor'] = meta.realtime_factor
        with open(args.output, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    if result.global_prediction == 1:
        print("\n*** ORCA DETECTED! ***")
        return 0
    else:
        print("\nNo orca detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
