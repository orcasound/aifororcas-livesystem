"""
Audio Preprocessing Tests - Verify model_v1 audio preprocessing matches fastai_audio

These tests are modular:
- Can generate reference outputs from fastai_audio (in inference-venv)
- Can compare model_v1 outputs against saved references (in model-v1-venv)
"""

from pathlib import Path

import pytest
import torch
from model_v1.audio_frontend import (
    audio_segment_generator,
    featurize_waveform,
    load_processed_waveform,
    prepare_audio,
    standardize,
)
from tests.utils import diff_specs, plot_spec_comparison


def _make_segments(sample_1min_wav, v1_config, max_segments, segments_start_s):
    """Create the standard audio segment generator for parity tests."""
    return audio_segment_generator(
        sample_1min_wav,
        segment_duration_s=v1_config["inference"]["window_s"],
        segment_hop_s=v1_config["inference"]["window_hop_s"],
        max_segments=max_segments,
        start_time_s=segments_start_s or 0.0,
    )


class TestAudioPreprocessingUnit:
    """Unit tests for model_v1 audio preprocessing components"""

    def test_load_audio(self, sample_1min_wav, v1_config):
        """Test that audio loads correctly with config"""
        waveform, sr = load_processed_waveform(sample_1min_wav, v1_config["audio"])

        assert isinstance(waveform, torch.Tensor)
        assert waveform.ndim == 2  # (channels, samples)
        assert waveform.shape[0] == 1  # mono after downmix
        assert sr == v1_config["audio"]["resample_rate"]

    def test_featurize_waveform(self, sample_1min_wav, v1_config):
        """Test mel spectrogram feature extraction"""
        waveform, sr = load_processed_waveform(sample_1min_wav, v1_config["audio"])
        features, times, freqs = featurize_waveform(waveform, sr, v1_config["spectrogram"])

        # Check feature shape
        assert features.ndim == 3  # (channels, n_mels, n_frames)
        assert features.shape[1] == v1_config["spectrogram"]["mel_n_filters"]

        # Check times axis
        assert times.ndim == 1
        assert len(times) == features.shape[2]

        # Check freqs axis
        assert freqs.ndim == 1
        assert len(freqs) == v1_config["spectrogram"]["mel_n_filters"]

    def test_standardize(self, sample_1min_wav, v1_config):
        """Test spectrogram standardization (pad/crop)"""
        waveform, sr = load_processed_waveform(sample_1min_wav, v1_config["audio"])
        features, _, _ = featurize_waveform(waveform, sr, v1_config["spectrogram"])

        # Add resample_rate to model_config
        model_config = {**v1_config["model"], "resample_rate": v1_config["audio"]["resample_rate"]}
        standardized = standardize(features, model_config, v1_config["spectrogram"])

        # Check output has expected frame count
        expected_frames = int(
            v1_config["model"]["input_pad_s"]
            * v1_config["audio"]["resample_rate"]
            / v1_config["spectrogram"]["hop_length"]
        )
        assert standardized.shape[2] == expected_frames

    def test_prepare_audio_full_pipeline(self, sample_1min_wav, v1_config):
        """Test complete audio preparation pipeline"""
        mel_spec = prepare_audio(sample_1min_wav, v1_config)

        assert isinstance(mel_spec, torch.Tensor)
        assert mel_spec.ndim == 3
        assert mel_spec.shape[1] == v1_config["spectrogram"]["mel_n_filters"]

        # Check standardized frame count
        expected_frames = int(
            v1_config["model"]["input_pad_s"]
            * v1_config["audio"]["resample_rate"]
            / v1_config["spectrogram"]["hop_length"]
        )
        assert mel_spec.shape[2] == expected_frames


class TestAudioPreprocessingParity:
    """
    Parity tests comparing model_v1 with fastai_audio.

    These tests verify that model_v1 produces identical outputs to fastai_audio.
    """

    def test_generate_reference_outputs(
        self, sample_1min_wav, reference_dir, fastai_available, v1_config, max_segments, segments_start_s
    ):
        """
        Generate reference outputs from fastai_audio for later comparison.

        Run this in the fastai environment to create reference files.
        Saves mel_raw (pure mel, no padding) and mel_standardized (full pipeline) outputs.
        """
        if not fastai_available:
            pytest.skip(
                "fastai_audio not available - run in fastai environment to generate references"
            )

        import random

        random.seed(42)  # Fix random seed for reproducible padding

        from model_v1.legacy_fastai_frontend import prepare_audio as fastai_prepare_audio
        from model_v1.legacy_fastai_frontend import prepare_audio_mel_raw

        wav_name = Path(sample_1min_wav).stem
        references = {}
        for window_idx, (segment_path, _, _) in enumerate(
            _make_segments(sample_1min_wav, v1_config, max_segments, segments_start_s)
        ):
            # mel_raw: Pure mel spectrogram (no padding/standardization)
            mel_raw_spec = prepare_audio_mel_raw(segment_path, v1_config)

            # mel_standardized: Full pipeline with standardization
            mel_standardized_spec = fastai_prepare_audio(segment_path, v1_config)

            references[f"segment_{window_idx}"] = {
                "mel_raw": mel_raw_spec,
                "mel_standardized": mel_standardized_spec,
            }

        # Save reference spectrograms
        reference_file = reference_dir / f"{wav_name}_audio_reference.pt"
        torch.save(references, reference_file)
        print(f"Saved reference outputs to {reference_file}")

        # Print shape info for debugging
        for seg_name, specs in references.items():
            print(
                f"  {seg_name}: mel_raw={specs['mel_raw'].shape}, mel_standardized={specs['mel_standardized'].shape}"
            )

        # Save spectrogram comparison images (mel_raw vs mel_standardized for each segment)
        images_dir = reference_dir / f"{wav_name}_spectrograms"
        images_dir.mkdir(exist_ok=True)

        for seg_name, specs in references.items():
            img_path = images_dir / f"{seg_name}_comparison.png"
            plot_spec_comparison(
                specs["mel_raw"],
                specs["mel_standardized"],
                img_path,
                f"{wav_name} {seg_name}",
            )

        print(f"Saved spectrogram images to {images_dir}")

    def test_mel_raw_parity(self, sample_1min_wav, audio_references, v1_config, max_segments, segments_start_s):
        """
        MEL RAW PARITY: Test pure mel spectrogram computation from raw audio clips.

        Tests mel spectrogram generation from raw 2-second clips WITHOUT padding/standardization.
        This isolates the core mel computation (downmix -> resample -> mel spec).
        """
        # Check if references have mel_raw key
        first_ref = list(audio_references.values())[0]
        if not isinstance(first_ref, dict) or "mel_raw" not in first_ref:
            pytest.skip(
                "Reference file is in old format. Regenerate with test_generate_reference_outputs."
            )

        mismatches = []
        for window_idx, (segment_path, _, _) in enumerate(
            _make_segments(sample_1min_wav, v1_config, max_segments, segments_start_s)
        ):
            # Process with model_v1 - raw mel only (no standardization)
            waveform, sr = load_processed_waveform(segment_path, v1_config["audio"])
            model_v1_spec, _, _ = featurize_waveform(waveform, sr, v1_config["spectrogram"])

            ref_key = f"segment_{window_idx}"
            if ref_key not in audio_references:
                continue

            fastai_spec = audio_references[ref_key]["mel_raw"]

            # Compare using SpecDiff (handles overlapping frames automatically)
            diff = diff_specs(model_v1_spec, fastai_spec)
            try:
                diff.assert_close(name=ref_key)
            except AssertionError as e:
                mismatches.append(str(e))

        assert len(mismatches) == 0, "mel_raw parity failures:\n" + "\n".join(mismatches)

    def test_mel_standardized_parity(self, sample_1min_wav, audio_references, v1_config, debug_dir, max_segments, segments_start_s):
        """
        MEL STANDARDIZED PARITY: Compare full fastai_audio pipeline including input standardization (padding/cropping).

        Run with --save-debug to generate debug output to tests/tmp/mel_standardized_debug/
        for detailed analysis of differences.
        """
        # Set up debug output directory if enabled
        debug_output_dir = None
        if debug_dir is not None:
            debug_output_dir = debug_dir / "mel_standardized_debug"
            debug_output_dir.mkdir(parents=True, exist_ok=True)

        mismatches = []

        for window_idx, (segment_path, _, _) in enumerate(
            _make_segments(sample_1min_wav, v1_config, max_segments, segments_start_s)
        ):
            # Process with model_v1 - full pipeline including standardization
            model_v1_spec = prepare_audio(segment_path, v1_config)

            ref_key = f"segment_{window_idx}"
            if ref_key not in audio_references:
                continue

            fastai_spec = audio_references[ref_key]["mel_standardized"]

            # Compare using SpecDiff
            diff = diff_specs(model_v1_spec, fastai_spec)

            # Save debug output for this segment (only if --save-debug flag is set)
            if debug_output_dir is not None:
                seg_dir = debug_output_dir / ref_key
                diff.save_debug(seg_dir, ref_key)

            # Check tolerance and collect failures
            try:
                diff.assert_close(name=ref_key)
            except AssertionError as e:
                mismatches.append(str(e))

        if debug_output_dir is not None:
            print(f"\nDebug output saved to: {debug_output_dir}")

        assert len(mismatches) == 0, "mel_standardized parity failures:\n" + "\n".join(mismatches)
