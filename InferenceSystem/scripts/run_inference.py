#!/usr/bin/env python
"""Run model_v1 inference on a local audio file or directory.

Usage:
    python scripts/run_inference.py                              # uses default WAV test file
    python scripts/run_inference.py path/to/audio.wav
    python scripts/run_inference.py path/to/audio.flac
    python scripts/run_inference.py path/to/audio.wav --output results.json
    python scripts/run_inference.py path/to/audio_dir/          # recurse over wav/flac/mp3
    python scripts/run_inference.py path/to/audio_dir/ --output path/to/out_dir/
"""
import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_v1.inference import OrcaHelloSRKWDetectorV1

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_data')
DEFAULT_AUDIO_PATH = os.path.join(TEST_DATA_DIR, "rpi_sunset_bay_2025_09_18_01_12_06_PDT--f6b3fcd7-2036-433a-8a18-76a6b3b4f0c9.wav")

AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3'}


def find_audio_files(root: Path) -> list[Path]:
    """Recursively find all audio files under root."""
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS)


def run_single(model, audio_path: Path, config: dict, output_path: Path | None, verbose: bool = True) -> dict:
    """Run inference on a single file. Returns a summary dict."""
    if verbose:
        print(f"Processing: {audio_path}")

    result = model.detect_srkw_from_file(str(audio_path), config)

    if verbose:
        print("\nLocal Predictions (per segment):")
        print("-" * 60)
        print(f"{'Segment':<10} {'Start (s)':<12} {'Duration (s)':<14} {'Prediction':<12} {'Confidence':<12}")
        print("-" * 60)
        for i, (pred, conf, seg) in enumerate(zip(result.local_predictions, result.local_confidences, result.segment_predictions)):
            print(f"{i+1:<10} {seg.start_time_s:<12.1f} {seg.duration_s:<14.1f} {pred:<12} {conf:<12.3f}")
        print("-" * 60)

        num_positive = sum(result.local_predictions)
        positive_segments = [i+1 for i, pred in enumerate(result.local_predictions) if pred == 1]
        positive_segment_times = [result.segment_predictions[i-1].start_time_s for i in positive_segments]

        if positive_segments:
            print(f"Detected in segments: {positive_segments}")
            print(f"Detected in times:    {positive_segment_times}")

        print(f"\n--- Summary ---")
        print(f"{num_positive}/{len(result.local_predictions)} segments predicted positive")
        print(f"global_confidence: {result.global_confidence:.3f}")
        print(f"global_prediction: {result.global_prediction}")

        meta = result.metadata
        print(f"\n--- Performance ---")
        print(f"File duration:    {meta.file_duration_s:.2f}s")
        print(f"Processing time:  {meta.processing_time_s:.2f}s")
        print(f"Realtime factor:  {meta.realtime_factor:.2f}x")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_dict = asdict(result)
        result_dict['metadata']['realtime_factor'] = result.metadata.realtime_factor
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    return {
        'file_path': str(audio_path),
        'global_prediction': result.global_prediction,
        'global_confidence': result.global_confidence,
        'num_positive_segments': sum(result.local_predictions),
        'num_total_segments': len(result.local_predictions),
    }


def run_directory(model, input_dir: Path, output_dir: Path, config: dict):
    """Run inference on all audio files in input_dir, mirroring structure to output_dir."""
    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} audio file(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    summary_rows = []
    for audio_path in audio_files:
        rel = audio_path.relative_to(input_dir)
        output_json = output_dir / rel.parent / (rel.stem + '.json')
        print(f"\n[{audio_files.index(audio_path)+1}/{len(audio_files)}] {rel}")
        row = run_single(model, audio_path, config, output_json, verbose=False)
        row['file_path'] = str(rel)  # store relative path in CSV
        # Print compact per-file summary
        print(f"  global_prediction={row['global_prediction']}  "
              f"global_confidence={row['global_confidence']:.3f}  "
              f"positive_segments={row['num_positive_segments']}/{row['num_total_segments']}")
        summary_rows.append(row)

    # Write summary CSV
    csv_path = output_dir / 'summary.csv'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file_path', 'global_prediction', 'global_confidence',
            'num_positive_segments', 'num_total_segments'
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 60)
    print(f"Summary CSV written to: {csv_path}")
    num_detected = sum(1 for r in summary_rows if r['global_prediction'] == 1)
    print(f"Files with orca detected: {num_detected}/{len(summary_rows)}")


def main():
    parser = argparse.ArgumentParser(description="Run model_v1 inference on audio files or directories")
    parser.add_argument("audio_path", nargs="?", default=DEFAULT_AUDIO_PATH,
                        help="Path to audio file or directory (WAV, FLAC, MP3)")
    parser.add_argument("--output", "-o",
                        help="Output path: JSON file (single-file mode) or directory (directory mode)")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)

    # Model and config paths relative to InferenceSystem directory
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model_v1.pt')
    config_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loading model from: {model_path}")
    print(f"Using config from:  {config_path}")
    print("-" * 60)

    model = OrcaHelloSRKWDetectorV1.from_checkpoint(model_path, config)
    print(f"Device: {model._device}  |  Dtype: {model._dtype}")
    print("-" * 60)

    if audio_path.is_dir():
        output_dir = Path(args.output) if args.output else audio_path.parent / (audio_path.name + '_inference_output')
        run_directory(model, audio_path, output_dir, config)
    else:
        print(f"Testing audio file: {audio_path}")
        print("-" * 60)
        output_path = Path(args.output) if args.output else None
        row = run_single(model, audio_path, config, output_path, verbose=True)
        if row['global_prediction'] == 1:
            print("\n*** ORCA DETECTED! ***")
            return 0
        else:
            print("\nNo orca detected.")
            return 1


if __name__ == "__main__":
    sys.exit(main())
