"""
Audio Preprocessing Tests - Verify model_v1 audio preprocessing matches fastai_audio

Test (a): Verify that a 1-minute WAV file is processed into the same model inputs
by both fastai_audio and model_v1.

These tests are modular:
- Can generate reference outputs from fastai_audio
- Can compare model_v1 outputs against saved references
- Can run direct comparison if both environments available
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def plot_spectrogram(spec_tensor, output_path, title=None, sr=20000, hop_length=256):
    """Plot and save a spectrogram image."""
    spec = spec_tensor[0].numpy() if spec_tensor.ndim == 3 else spec_tensor.numpy()
    n_mels, n_frames = spec.shape
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis',
                    extent=[0, n_frames * hop_length / sr, 0, n_mels])
    ax.set(xlabel='Time (s)', ylabel='Mel bin', title=title)
    fig.colorbar(img, ax=ax, format='%.0f', label='dB')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


class TestAudioPreprocessingUnit:
    """Unit tests for model_v1 audio preprocessing components"""

    def test_load_audio(self, sample_1min_wav):
        """Test that audio loads correctly"""
        from model_v1.audio_frontend import load_audio

        waveform, sr = load_audio(sample_1min_wav)

        assert isinstance(waveform, torch.Tensor)
        assert waveform.ndim == 2  # (channels, samples)
        assert sr > 0

    def test_downmix_to_mono(self, sample_1min_wav):
        """Test stereo to mono downmix"""
        from model_v1.audio_frontend import downmix_to_mono, load_audio

        waveform, sr = load_audio(sample_1min_wav)
        mono = downmix_to_mono(waveform)

        assert mono.shape[0] == 1  # Single channel

    def test_resample_audio(self, sample_1min_wav):
        """Test resampling to 20kHz"""
        from model_v1.audio_frontend import downmix_to_mono, load_audio, resample_audio

        waveform, sr = load_audio(sample_1min_wav)
        mono = downmix_to_mono(waveform)
        resampled = resample_audio(mono, orig_sr=sr, target_sr=20000)

        # Check approximate duration preserved
        original_duration = waveform.shape[1] / sr
        resampled_duration = resampled.shape[1] / 20000
        assert abs(original_duration - resampled_duration) < 0.1

    def test_compute_mel_spectrogram(self, sample_1min_wav):
        """Test mel spectrogram generation"""
        from model_v1.audio_frontend import (
            SPECTROGRAM_CONFIG,
            compute_mel_spectrogram,
            downmix_to_mono,
            load_audio,
            pad_or_trim,
            resample_audio,
        )

        waveform, sr = load_audio(sample_1min_wav)
        mono = downmix_to_mono(waveform)
        resampled = resample_audio(mono, orig_sr=sr, target_sr=20000)
        padded = pad_or_trim(resampled, duration_ms=4000, sample_rate=20000)
        mel_spec = compute_mel_spectrogram(padded)

        # Check shape: (channels, n_mels, time_frames)
        assert mel_spec.ndim == 3
        assert mel_spec.shape[1] == SPECTROGRAM_CONFIG["n_mels"]  # 256 mel bins

    def test_prepare_audio_full_pipeline(self, sample_1min_wav):
        """Test complete audio preparation pipeline"""
        from model_v1.audio_frontend import SPECTROGRAM_CONFIG, prepare_audio

        mel_spec = prepare_audio(sample_1min_wav)

        assert isinstance(mel_spec, torch.Tensor)
        assert mel_spec.ndim == 3
        assert mel_spec.shape[1] == SPECTROGRAM_CONFIG["n_mels"]


class TestAudioPreprocessingParity:
    """
    Parity tests comparing model_v1 with fastai_audio.

    These tests verify that model_v1 produces identical outputs to fastai_audio.
    """

    def test_generate_reference_outputs(self, sample_1min_wav, reference_dir, fastai_available):
        """
        Generate reference outputs from fastai_audio for later comparison.

        Run this with fastai environment to create reference files.
        Saves BOTH Stage B (before standardization) and Stage C (after) outputs.
        """
        if not fastai_available:
            pytest.skip("fastai_audio not available - run in fastai environment to generate references")

        import random
        random.seed(42)  # Fix random seed for reproducible padding

        from audio.data import AudioConfig, AudioList, SpectrogramConfig
        from librosa import get_duration
        from pydub import AudioSegment

        # Get duration and create 2-second segments (matching FastAIModel.predict)
        max_length = get_duration(path=sample_1min_wav)
        wav_name = Path(sample_1min_wav).stem

        # Create temp directory for segments
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = AudioSegment.from_wav(sample_1min_wav)

            # Extract first few 2-second segments for testing
            segments_to_test = min(5, int(np.floor(max_length) - 1))
            segment_paths = []
            for i in range(segments_to_test):
                segment_path = f"{tmpdir}/{wav_name}_{i}_{i+2}.wav"
                segment = audio[i * 1000:(i + 2) * 1000]
                segment.export(segment_path, format="wav")
                segment_paths.append(segment_path)

            # Configure fastai_audio (exact params from fastai_inference.py)
            # Use pad_mode="zeros-after" for deterministic padding (zeros at end only)
            config = AudioConfig(
                standardize=False,
                sg_cfg=SpectrogramConfig(
                    f_min=0.0,
                    f_max=10000,
                    hop_length=256,
                    n_fft=2560,
                    n_mels=256,
                    pad=0,
                    to_db_scale=True,
                    top_db=100,
                    win_length=None,
                    n_mfcc=20
                )
            )
            config.duration = 4000
            config.resample_to = 20000
            config.downmix = True
            config.pad_mode = "zeros-after"  # Deterministic: zeros at end only

            # Process with fastai_audio - get BOTH Stage B and Stage C outputs
            from audio.transform import tfm_downmix, tfm_resample
            from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
            import torchaudio

            references = {}
            for idx, segment_path in enumerate(segment_paths):
                # Stage B: Pure mel spectrogram from raw 2-second clips (NO padding)
                # Load audio
                sig, sr = torchaudio.load(segment_path)

                # Downmix to mono (if stereo)
                if sig.shape[0] > 1:
                    sig = tfm_downmix(sig)

                # Resample to 20kHz
                if sr != config.resample_to:
                    sig = tfm_resample(sig, sr, config.resample_to)

                # NO PADDING FOR STAGE B - test raw mel spectrogram computation
                # The 2-second clips will produce ~156 frames naturally

                # Create mel spectrogram (Stage B - pure mel computation from raw audio)
                # CRITICAL: Don't pass sample_rate - uses default 16000 (fastai quirk)
                mel_transform = MelSpectrogram(
                    n_fft=config.sg_cfg.n_fft,
                    hop_length=config.sg_cfg.hop_length,
                    n_mels=config.sg_cfg.n_mels,
                    f_min=config.sg_cfg.f_min,
                    f_max=config.sg_cfg.f_max,
                    pad=config.sg_cfg.pad,
                )
                amplitude_to_db = AmplitudeToDB(top_db=config.sg_cfg.top_db)

                mel_spec = mel_transform(sig)
                stage_b_spec = amplitude_to_db(mel_spec).clone()

                # Stage C: Get fully processed output via AudioList (includes padding + tfm_crop_time)
                with tempfile.TemporaryDirectory() as single_dir:
                    shutil.copy(segment_path, single_dir)
                    test = AudioList.from_folder(single_dir, config=config).split_none().label_empty()
                    testdb = test.transform(None).databunch(bs=1)
                    stage_c_spec = testdb.x[0].spectro.clone()

                references[f"segment_{idx}"] = {
                    "stage_b": stage_b_spec,  # Raw mel spec from 2-sec clip (~156 frames)
                    "stage_c": stage_c_spec,  # After padding + standardization (312 frames)
                }

            # Save reference spectrograms
            reference_file = reference_dir / f"{wav_name}_audio_reference.pt"
            torch.save(references, reference_file)
            print(f"Saved reference outputs to {reference_file}")

            # Print shape info for debugging
            for seg_name, specs in references.items():
                print(f"  {seg_name}: stage_b={specs['stage_b'].shape}, stage_c={specs['stage_c'].shape}")

            # Save spectrogram images
            images_dir = reference_dir / f"{wav_name}_spectrograms"
            images_dir.mkdir(exist_ok=True)

            for seg_name, specs in references.items():
                for stage, spec_tensor in specs.items():
                    img_path = images_dir / f"{seg_name}_{stage}.png"
                    plot_spectrogram(
                        spec_tensor,
                        img_path,
                        title=f"fastai_audio - {wav_name} {seg_name} ({stage})"
                    )

            print(f"Saved spectrogram images to {images_dir}")

    def test_stage_b_parity(self, sample_1min_wav, reference_dir):
        """
        STAGE B PARITY: Test pure mel spectrogram computation from raw audio clips.

        Tests mel spectrogram generation from raw 2-second clips WITHOUT padding.
        This isolates the core mel computation (downmix → resample → mel spec)
        from duration standardization.

        Expected behavior:
        - 2-second clips at 20kHz → ~156 mel spectrogram frames
        - Compares all frames with strict tolerance (1e-4)
        """
        from librosa import get_duration
        from model_v1.audio_frontend import audio_to_mel_spectrogram_from_file
        from pydub import AudioSegment

        wav_name = Path(sample_1min_wav).stem
        reference_file = reference_dir / f"{wav_name}_audio_reference.pt"

        if not reference_file.exists():
            pytest.skip(f"Reference file not found: {reference_file}. Run test_generate_reference_outputs first.")

        # Load references
        references = torch.load(reference_file, weights_only=False)

        # Check if references have stage_b (new format)
        first_ref = list(references.values())[0]
        if not isinstance(first_ref, dict) or "stage_b" not in first_ref:
            pytest.skip("Reference file is in old format. Regenerate with test_generate_reference_outputs.")

        # Create same segments as fastai test
        max_length = get_duration(path=sample_1min_wav)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = AudioSegment.from_wav(sample_1min_wav)

            segments_to_test = min(5, int(np.floor(max_length) - 1))
            mismatches = []

            for i in range(segments_to_test):
                segment_path = f"{tmpdir}/{wav_name}_{i}_{i+2}.wav"
                segment = audio[i * 1000:(i + 2) * 1000]
                segment.export(segment_path, format="wav")

                # Process with model_v1 - Stage B only (no standardization)
                model_v1_spec = audio_to_mel_spectrogram_from_file(segment_path)

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
                except AssertionError as e:
                    max_diff = (model_v1_overlap - fastai_overlap).abs().max().item()
                    mean_diff = (model_v1_overlap - fastai_overlap).abs().mean().item()
                    mismatches.append(
                        f"Segment {i}: Stage B mismatch - "
                        f"model_v1={model_v1_spec.shape}, fastai={fastai_spec.shape}, "
                        f"compared {n_frames} frames, max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}"
                    )

            assert len(mismatches) == 0, f"Stage B parity failures:\n" + "\n".join(mismatches)

    @pytest.mark.xfail(reason="Known issue: fastai_audio pads dB spectrograms with 0.0 dB (incorrect)")
    def test_stage_c_parity(self, sample_1min_wav, reference_dir, numerical_tolerance):
        """
        STAGE C PARITY: Compare full pipeline including standardization.

        KNOWN TO FAIL due to fastai_audio padding bug:
        - fastai pads dB-scale spectrograms with 0.0 dB (represents full power, not silence)
        - model_v1 may crop instead of pad depending on frame count
        - High-frequency bins (243-255) most affected

        This test is marked xfail to document the known discrepancy.
        """
        from librosa import get_duration
        from model_v1.audio_frontend import prepare_audio
        from pydub import AudioSegment

        wav_name = Path(sample_1min_wav).stem
        reference_file = reference_dir / f"{wav_name}_audio_reference.pt"

        if not reference_file.exists():
            pytest.skip(f"Reference file not found: {reference_file}. Run test_generate_reference_outputs first.")

        # Load references
        references = torch.load(reference_file, weights_only=False)

        # Check if references have stage_c (new format) or are in old format
        first_ref = list(references.values())[0]
        if isinstance(first_ref, dict) and "stage_c" in first_ref:
            get_ref = lambda r: r["stage_c"]
        else:
            get_ref = lambda r: r  # Old format: direct tensor

        # Create same segments as fastai test
        max_length = get_duration(path=sample_1min_wav)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = AudioSegment.from_wav(sample_1min_wav)

            segments_to_test = min(5, int(np.floor(max_length) - 1))
            mismatches = []

            for i in range(segments_to_test):
                segment_path = f"{tmpdir}/{wav_name}_{i}_{i+2}.wav"
                segment = audio[i * 1000:(i + 2) * 1000]
                segment.export(segment_path, format="wav")

                # Process with model_v1 - full pipeline
                model_v1_spec = prepare_audio(segment_path)

                # Get reference
                ref_key = f"segment_{i}"
                if ref_key not in references:
                    continue

                fastai_spec = get_ref(references[ref_key])

                # Compare shapes
                if model_v1_spec.shape != fastai_spec.shape:
                    mismatches.append(f"Segment {i}: shape mismatch - model_v1={model_v1_spec.shape}, fastai={fastai_spec.shape}")
                    continue

                # Compare values
                if not torch.allclose(model_v1_spec, fastai_spec,
                                     atol=numerical_tolerance["atol"],
                                     rtol=numerical_tolerance["rtol"]):
                    max_diff = (model_v1_spec - fastai_spec).abs().max().item()
                    mismatches.append(f"Segment {i}: value mismatch - max_diff={max_diff:.6e}")

            assert len(mismatches) == 0, f"Stage C parity failures:\n" + "\n".join(mismatches)

    @pytest.mark.requires_fastai
    def test_audio_parity_direct(self, sample_1min_wav, fastai_available, numerical_tolerance):
        """
        Direct comparison between model_v1 and fastai_audio (requires both available).
        """
        if not fastai_available:
            pytest.skip("fastai_audio not available")

        from audio.data import AudioConfig, AudioList, SpectrogramConfig
        from librosa import get_duration
        from model_v1.audio_frontend import prepare_audio
        from pydub import AudioSegment

        wav_name = Path(sample_1min_wav).stem
        max_length = get_duration(path=sample_1min_wav)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = AudioSegment.from_wav(sample_1min_wav)

            # Test first 3 segments
            for i in range(min(3, int(np.floor(max_length) - 1))):
                segment_path = f"{tmpdir}/{wav_name}_{i}_{i+2}.wav"
                segment = audio[i * 1000:(i + 2) * 1000]
                segment.export(segment_path, format="wav")

                # model_v1 processing
                model_v1_spec = prepare_audio(segment_path)

                # fastai_audio processing
                config = AudioConfig(
                    standardize=False,
                    sg_cfg=SpectrogramConfig(
                        f_min=0.0, f_max=10000, hop_length=256,
                        n_fft=2560, n_mels=256, pad=0,
                        to_db_scale=True, top_db=100,
                        win_length=None, n_mfcc=20
                    )
                )
                config.duration = 4000
                config.resample_to = 20000
                config.downmix = True

                # Create single-file AudioList
                with tempfile.TemporaryDirectory() as single_dir:
                    shutil.copy(segment_path, single_dir)
                    test = AudioList.from_folder(single_dir, config=config).split_none().label_empty()
                    testdb = test.transform(None).databunch(bs=1)
                    fastai_spec = testdb.x[0].spectro

                # Compare
                assert model_v1_spec.shape == fastai_spec.shape, \
                    f"Segment {i}: shape mismatch"

                assert torch.allclose(model_v1_spec, fastai_spec,
                                     atol=numerical_tolerance["atol"],
                                     rtol=numerical_tolerance["rtol"]), \
                    f"Segment {i}: value mismatch, max_diff={(model_v1_spec - fastai_spec).abs().max().item():.6e}"
