"""
Audio Preprocessing Tests - Verify model_v1 audio preprocessing matches fastai_audio

These tests are modular:
- Can generate reference outputs from fastai_audio (in inference-venv)
- Can compare model_v1 outputs against saved references (in model-v1-venv)
"""

import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_spectrogram(spec_tensor, output_path, title=None, sr=20000, hop_length=256):
    """Plot and save a spectrogram image."""
    spec = spec_tensor[0].numpy() if spec_tensor.ndim == 3 else spec_tensor.numpy()
    n_mels, n_frames = spec.shape
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, n_frames * hop_length / sr, 0, n_mels],
    )
    ax.set(xlabel="Time (s)", ylabel="Mel bin", title=title)
    fig.colorbar(img, ax=ax, format="%.0f", label="dB")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
        self, sample_1min_wav, reference_dir, fastai_available, v1_config
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

        from librosa import get_duration
        from model_v1.legacy_fastai_frontend import prepare_audio, prepare_audio_stage_b
        from pydub import AudioSegment

        # Get duration and create 2-second segments
        max_length = get_duration(path=sample_1min_wav)
        wav_name = Path(sample_1min_wav).stem

        # Create temp directory for segments
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = AudioSegment.from_wav(sample_1min_wav)

            # Extract first few 2-second segments for testing
            segments_to_test = min(5, int(np.floor(max_length) - 1))
            references = {}

            for i in range(segments_to_test):
                segment_path = f"{tmpdir}/{wav_name}_{i}_{i+2}.wav"
                segment = audio[i * 1000 : (i + 2) * 1000]
                segment.export(segment_path, format="wav")

                # Stage B: Pure mel spectrogram (no padding)
                stage_b_spec = prepare_audio_stage_b(segment_path, v1_config)

                # Stage C: Full pipeline with standardization
                stage_c_spec = prepare_audio(segment_path, v1_config)

                references[f"segment_{i}"] = {
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

            # Save spectrogram images
            images_dir = reference_dir / f"{wav_name}_spectrograms"
            images_dir.mkdir(exist_ok=True)

            for seg_name, specs in references.items():
                for stage, spec_tensor in specs.items():
                    img_path = images_dir / f"{seg_name}_{stage}.png"
                    plot_spectrogram(
                        spec_tensor,
                        img_path,
                        title=f"fastai_audio - {wav_name} {seg_name} ({stage})",
                    )

            print(f"Saved spectrogram images to {images_dir}")

    def test_stage_b_parity(self, sample_1min_wav, reference_dir, v1_config):
        """
        STAGE B PARITY: Test pure mel spectrogram computation from raw audio clips.

        Tests mel spectrogram generation from raw 2-second clips WITHOUT padding.
        This isolates the core mel computation (downmix -> resample -> mel spec).
        """
        from librosa import get_duration
        from model_v1.audio_frontend import featurize_waveform, load_audio
        from pydub import AudioSegment

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
        max_length = get_duration(path=sample_1min_wav)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = AudioSegment.from_wav(sample_1min_wav)

            segments_to_test = min(5, int(np.floor(max_length) - 1))
            mismatches = []

            for i in range(segments_to_test):
                segment_path = f"{tmpdir}/{wav_name}_{i}_{i+2}.wav"
                segment = audio[i * 1000 : (i + 2) * 1000]
                segment.export(segment_path, format="wav")

                # Process with model_v1 - Stage B only (no standardization)
                waveform, sr = load_audio(segment_path, v1_config["audio"])
                model_v1_spec, _, _ = featurize_waveform(waveform, sr, v1_config["spectrogram"])

                # Get reference Stage B
                ref_key = f"segment_{i}"
                if ref_key not in references:
                    continue

                fastai_spec = references[ref_key]["stage_b"]

                # Compare overlapping frames only
                n_frames = min(model_v1_spec.shape[2], fastai_spec.shape[2])

                model_v1_overlap = model_v1_spec[:, :, :n_frames]
                fastai_overlap = fastai_spec[:, :, :n_frames]

                # Strict tolerance for Stage B (core mel computation)
                try:
                    torch.testing.assert_close(
                        model_v1_overlap,
                        fastai_overlap,
                        atol=1e-4,
                        rtol=1e-4,
                    )
                except AssertionError:
                    max_diff = (model_v1_overlap - fastai_overlap).abs().max().item()
                    mean_diff = (model_v1_overlap - fastai_overlap).abs().mean().item()
                    mismatches.append(
                        f"Segment {i}: Stage B mismatch - "
                        f"model_v1={model_v1_spec.shape}, fastai={fastai_spec.shape}, "
                        f"compared {n_frames} frames, max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}"
                    )

            assert len(mismatches) == 0, f"Stage B parity failures:\n" + "\n".join(mismatches)

    @pytest.mark.xfail(reason="Known issue: fastai_audio pads dB spectrograms with 0.0 dB")
    def test_stage_c_parity(self, sample_1min_wav, reference_dir, numerical_tolerance, v1_config):
        """
        STAGE C PARITY: Compare full pipeline including standardization.

        KNOWN TO FAIL due to fastai_audio padding bug:
        - fastai pads dB-scale spectrograms with 0.0 dB (represents full power, not silence)
        - model_v1 may crop instead of pad depending on frame count

        This test is marked xfail to document the known discrepancy.
        """
        from librosa import get_duration
        from model_v1.audio_frontend import prepare_audio
        from pydub import AudioSegment

        wav_name = Path(sample_1min_wav).stem
        reference_file = reference_dir / f"{wav_name}_audio_reference.pt"

        if not reference_file.exists():
            pytest.skip(
                f"Reference file not found: {reference_file}. "
                "Run test_generate_reference_outputs first."
            )

        # Load references
        references = torch.load(reference_file, weights_only=False)

        # Check format
        first_ref = list(references.values())[0]
        if isinstance(first_ref, dict) and "stage_c" in first_ref:
            get_ref = lambda r: r["stage_c"]
        else:
            get_ref = lambda r: r  # Old format

        # Create same segments as fastai test
        max_length = get_duration(path=sample_1min_wav)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = AudioSegment.from_wav(sample_1min_wav)

            segments_to_test = min(5, int(np.floor(max_length) - 1))
            mismatches = []

            for i in range(segments_to_test):
                segment_path = f"{tmpdir}/{wav_name}_{i}_{i+2}.wav"
                segment = audio[i * 1000 : (i + 2) * 1000]
                segment.export(segment_path, format="wav")

                # Process with model_v1 - full pipeline
                model_v1_spec = prepare_audio(segment_path, v1_config)

                # Get reference
                ref_key = f"segment_{i}"
                if ref_key not in references:
                    continue

                fastai_spec = get_ref(references[ref_key])

                # Compare shapes
                if model_v1_spec.shape != fastai_spec.shape:
                    mismatches.append(
                        f"Segment {i}: shape mismatch - "
                        f"model_v1={model_v1_spec.shape}, fastai={fastai_spec.shape}"
                    )
                    continue

                # Compare values
                if not torch.allclose(
                    model_v1_spec,
                    fastai_spec,
                    atol=numerical_tolerance["atol"],
                    rtol=numerical_tolerance["rtol"],
                ):
                    max_diff = (model_v1_spec - fastai_spec).abs().max().item()
                    mismatches.append(f"Segment {i}: value mismatch - max_diff={max_diff:.6e}")

            assert len(mismatches) == 0, f"Stage C parity failures:\n" + "\n".join(mismatches)

    @pytest.mark.requires_fastai
    def test_audio_parity_direct(
        self, sample_1min_wav, fastai_available, numerical_tolerance, v1_config
    ):
        """
        Direct comparison between model_v1 and fastai_audio (requires both available).
        """
        if not fastai_available:
            pytest.skip("fastai_audio not available")

        from librosa import get_duration
        from model_v1.audio_frontend import prepare_audio
        from model_v1.legacy_fastai_frontend import prepare_audio as fastai_prepare_audio
        from pydub import AudioSegment

        wav_name = Path(sample_1min_wav).stem
        max_length = get_duration(path=sample_1min_wav)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = AudioSegment.from_wav(sample_1min_wav)

            # Test first 3 segments
            for i in range(min(3, int(np.floor(max_length) - 1))):
                segment_path = f"{tmpdir}/{wav_name}_{i}_{i+2}.wav"
                segment = audio[i * 1000 : (i + 2) * 1000]
                segment.export(segment_path, format="wav")

                # model_v1 processing
                model_v1_spec = prepare_audio(segment_path, v1_config)

                # fastai_audio processing
                fastai_spec = fastai_prepare_audio(segment_path, v1_config)

                # Compare
                assert model_v1_spec.shape == fastai_spec.shape, f"Segment {i}: shape mismatch"

                assert torch.allclose(
                    model_v1_spec,
                    fastai_spec,
                    atol=numerical_tolerance["atol"],
                    rtol=numerical_tolerance["rtol"],
                ), f"Segment {i}: value mismatch, max_diff={(model_v1_spec - fastai_spec).abs().max().item():.6e}"
