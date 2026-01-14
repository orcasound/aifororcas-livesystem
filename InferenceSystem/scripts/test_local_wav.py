#!/usr/bin/env python
"""Test inference on a local WAV file."""
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.fastai_inference import FastAIModel

DEFAULT_WAV_PATH = os.path.join(os.path.dirname(__file__), "test_data", "rpi_sunset_bay_2025_09_18_01_12_06_PDT--f6b3fcd7-2036-433a-8a18-76a6b3b4f0c9.wav")


def main():
    # Default to the test WAV file in test_data
    wav_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WAV_PATH

    # Model path relative to InferenceSystem directory
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model')

    print(f"Loading model from: {model_path}")
    print(f"Testing WAV file: {wav_path}")
    print("-" * 60)

    model = FastAIModel(
        model_path=model_path,
        model_name="model.pkl",
        threshold=0.5,
        min_num_positive_calls_threshold=3
    )

    result = model.predict(wav_path)

    # Print local predictions and confidences in aligned table format
    print("\nLocal Predictions (per 2-second segment):")
    print("-" * 40)
    print(f"{'Segment':<10} {'Prediction':<12} {'Confidence':<12}")
    print("-" * 40)
    
    local_preds = result['local_predictions']
    local_confs = result['local_confidences']
    
    for i, (pred, conf) in enumerate(zip(local_preds, local_confs)):
        print(f"{i+1:<10} {pred:<12} {conf:<12.3f}")
    
    print("-" * 40)
    
    # Summary statistics
    num_positive = sum(local_preds)
    positive_segments = [i+1 for i, pred in enumerate(local_preds) if pred == 1]
    
    print(f"\nSummary: {num_positive}/{len(local_preds)} segments predicted positive")
    if positive_segments:
        print(f"Detected in segments: {positive_segments}")

    # global_confidence (float)
    print(f"\nglobal_confidence: {result['global_confidence']:.3f}")

    # global_prediction (int or bool), print as is
    print(f"global_prediction: {result['global_prediction']}")

    if result['global_prediction'] == 1:
        print("\n*** ORCA DETECTED! ***")
        return 0
    else:
        print("\nNo orca detected.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
