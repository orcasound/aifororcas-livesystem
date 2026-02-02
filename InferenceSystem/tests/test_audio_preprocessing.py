"""
Audio Preprocessing Tests - Verify model_v1 audio preprocessing matches fastai_audio

These tests are modular:
- Can generate reference outputs from fastai_audio (in inference-venv)
- Can compare model_v1 outputs against saved references (in model-v1-venv)
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from model_v1.audio_frontend import audio_segment_generator
from tests.utils import diff_specs, plot_spec_comparison


class TestAudioPreprocessingUnit:
    """Unit tests for model_v1 audio preprocessing components"""

    def test_load_audio(self, sample_1min_wav, v1_config):
        """Test that audio loads correctly with config"""
        from model_v1.audio_frontend import load_audio

        waveform, sr = load_audio(sample_1min_wav, v1_config["audio"])

        assert isinstance(waveform, torch.Tensor)
        assert waveform.ndim == 2  # (channels, samples)
        assert waveform.shape[0] == 1  # mono after downmix
        assert sr == v1_config["audio"]["resample_rate"]

    def test_featurize_waveform(self, sample_1min_wav, v1_config):
        """Test mel spectrogram feature extraction"""
        from model_v1.audio_frontend import featurize_waveform, load_audio

        waveform, sr = load_audio(sample_1min_wav, v1_config["audio"])
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
        from model_v1.audio_frontend import featurize_waveform, load_audio, standardize

        waveform, sr = load_audio(sample_1min_wav, v1_config["audio"])
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
        from model_v1.audio_frontend import prepare_audio

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

        Run this with fastai environment to create reference files.
        Saves BOTH Stage B (before standardization) and Stage C (after) outputs.
        """
        if not fastai_available:
            pytest.skip(
                "fastai_audio not available - run in fastai environment to generate references"
            )

        import random

        random.seed(42)  # Fix random seed for reproducible padding

        from model_v1.legacy_fastai_frontend import prepare_audio, prepare_audio_stage_b

        wav_name = Path(sample_1min_wav).stem
        segments = audio_segment_generator(
            sample_1min_wav,
            segment_duration_s=v1_config["inference"]["window_s"],
            segment_hop_s=v1_config["inference"]["window_hop_s"],
            max_segments=max_segments,
            start_time_s=segments_start_s or 0.0,
        )
        references = {}
        for window_idx, (segment_path, _, _) in enumerate(segments):
            # Stage B: Pure mel spectrogram (no padding)
            stage_b_spec = prepare_audio_stage_b(segment_path, v1_config)

            # Stage C: Full pipeline with standardization
            stage_c_spec = prepare_audio(segment_path, v1_config)

            references[f"segment_{window_idx}"] = {
                "stage_b": stage_b_spec,
                "stage_c": stage_c_spec,
            }

        # Save reference spectrograms
        reference_file = reference_dir / f"{wav_name}_audio_reference.pt"
        torch.save(references, reference_file)
        print(f"Saved reference outputs to {reference_file}")

        # Print shape info for debugging
        for seg_name, specs in references.items():
            print(
                f"  {seg_name}: stage_b={specs['stage_b'].shape}, stage_c={specs['stage_c'].shape}"
            )

        # Save spectrogram comparison images (stage_b vs stage_c for each segment)
        images_dir = reference_dir / f"{wav_name}_spectrograms"
        images_dir.mkdir(exist_ok=True)

        for seg_name, specs in references.items():
            img_path = images_dir / f"{seg_name}_comparison.png"
            plot_spec_comparison(
                specs["stage_b"],
                specs["stage_c"],
                img_path,
                f"{wav_name} {seg_name}",
            )

        print(f"Saved spectrogram images to {images_dir}")

    def test_stage_b_parity(self, sample_1min_wav, reference_dir, v1_config, max_segments, segments_start_s):
        """
        STAGE B PARITY: Test pure mel spectrogram computation from raw audio clips.

        Tests mel spectrogram generation from raw 2-second clips WITHOUT padding.
        This isolates the core mel computation (downmix -> resample -> mel spec).
        """
        from model_v1.audio_frontend import featurize_waveform, load_audio

        wav_name = Path(sample_1min_wav).stem
        reference_file = reference_dir / f"{wav_name}_audio_reference.pt"

        if not reference_file.exists():
            pytest.skip(
                f"Reference file not found: {reference_file}. "
                "Run test_generate_reference_outputs first."
            )

        # Load references
        references = torch.load(reference_file, weights_only=False)

        # Check if references have stage_b (new format)
        first_ref = list(references.values())[0]
        if not isinstance(first_ref, dict) or "stage_b" not in first_ref:
            pytest.skip(
                "Reference file is in old format. Regenerate with test_generate_reference_outputs."
            )

        # Create same segments as fastai test
        segments = audio_segment_generator(
            sample_1min_wav,
            segment_duration_s=v1_config["inference"]["window_s"],
            segment_hop_s=v1_config["inference"]["window_hop_s"],
            max_segments=max_segments,
            start_time_s=segments_start_s or 0.0,
        )

        mismatches = []
        for window_idx, (segment_path, _, _) in enumerate(segments):
            # Process with model_v1 - Stage B only (no standardization)
            waveform, sr = load_audio(segment_path, v1_config["audio"])
            model_v1_spec, _, _ = featurize_waveform(waveform, sr, v1_config["spectrogram"])

            # Get reference Stage B
            ref_key = f"segment_{window_idx}"
            if ref_key not in references:
                continue

            fastai_spec = references[ref_key]["stage_b"]

            # Compare using SpecDiff (handles overlapping frames automatically)
            diff = diff_specs(model_v1_spec, fastai_spec)
            try:
                diff.assert_close(name=ref_key)
            except AssertionError as e:
                mismatches.append(str(e))

        assert len(mismatches) == 0, "Stage B parity failures:\n" + "\n".join(mismatches)

    def test_stage_c_parity(self, sample_1min_wav, reference_dir, v1_config, debug_dir, max_segments, segments_start_s):
        """
        STAGE C PARITY: Compare full pipeline including standardization.

        KNOWN TO FAIL due to fastai_audio padding bug:
        - fastai pads dB-scale spectrograms with 0.0 dB (represents full power, not silence)
        - model_v1 may crop instead of pad depending on frame count

        Run with --save-debug to generate debug output to tests/tmp/stage_c_debug/
        for detailed analysis of differences.
        """
        from model_v1.audio_frontend import prepare_audio

        wav_name = Path(sample_1min_wav).stem
        reference_file = reference_dir / f"{wav_name}_audio_reference.pt"

        if not reference_file.exists():
            pytest.skip(
                f"Reference file not found: {reference_file}. "
                "Run test_generate_reference_outputs first."
            )

        # Set up debug output directory if enabled
        stage_c_debug_dir = None
        if debug_dir is not None:
            stage_c_debug_dir = debug_dir / "stage_c_debug"
            stage_c_debug_dir.mkdir(parents=True, exist_ok=True)

        # Load references
        references = torch.load(reference_file, weights_only=False)

        # Create same segments as fastai test
        segments = audio_segment_generator(
            sample_1min_wav,
            segment_duration_s=v1_config["inference"]["window_s"],
            segment_hop_s=v1_config["inference"]["window_hop_s"],
            max_segments=max_segments,
            start_time_s=segments_start_s or 0.0,
        )

        mismatches = []

        for window_idx, (segment_path, _, _) in enumerate(segments):
            # Process with model_v1 - full pipeline
            model_v1_spec = prepare_audio(segment_path, v1_config)

            # Get reference
            ref_key = f"segment_{window_idx}"
            if ref_key not in references:
                continue

            fastai_spec = references[ref_key]["stage_c"]

            # Compare using SpecDiff
            diff = diff_specs(model_v1_spec, fastai_spec)
            
            # Save debug output for this segment (only if --save-debug flag is set)
            if stage_c_debug_dir is not None:
                seg_dir = stage_c_debug_dir / ref_key
                diff.save_debug(seg_dir, ref_key)

            # Check tolerance and collect failures
            try:
                diff.assert_close(name=ref_key)
            except AssertionError as e:
                mismatches.append(str(e))

        if stage_c_debug_dir is not None:
            print(f"\nDebug output saved to: {stage_c_debug_dir}")

        assert len(mismatches) == 0, "Stage C parity failures:\n" + "\n".join(mismatches)

    @pytest.mark.requires_fastai
    def test_audio_parity_direct(
        self, sample_1min_wav, fastai_available, v1_config, max_segments, segments_start_s
    ):
        """
        Direct comparison between model_v1 and fastai_audio (requires both available).
        """
        if not fastai_available:
            pytest.skip("fastai_audio not available")

        from model_v1.audio_frontend import prepare_audio
        from model_v1.legacy_fastai_frontend import (
            prepare_audio as fastai_prepare_audio,
        )

        segments = audio_segment_generator(
            sample_1min_wav,
            segment_duration_s=v1_config["inference"]["window_s"],
            segment_hop_s=v1_config["inference"]["window_hop_s"],
            max_segments=max_segments,
            start_time_s=segments_start_s or 0.0,
        )

        mismatches = []
        for window_idx, (segment_path, _, _) in enumerate(segments):
            # model_v1 processing
            model_v1_spec = prepare_audio(segment_path, v1_config)

            # fastai_audio processing
            fastai_spec = fastai_prepare_audio(segment_path, v1_config)

            # Compare using SpecDiff
            diff = diff_specs(model_v1_spec, fastai_spec)
            try:
                diff.assert_close(name=f"segment_{window_idx}")
            except AssertionError as e:
                mismatches.append(str(e))

        assert len(mismatches) == 0, "Direct parity failures:\n" + "\n".join(mismatches)
