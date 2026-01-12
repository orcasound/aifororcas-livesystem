"""
Audio Frontend for model_v1 - Replaces fastai_audio for audio preprocessing

This module provides audio loading and mel spectrogram generation that matches
the fastai_audio implementation exactly for inference parity.

Interfaces:
- load_audio(file_path, audio_config) -> (waveform, sample_rate)
- featurize_waveform(waveform, sample_rate, spectrogram_config) -> (features, times, freqs)
- standardize(spectrogram, model_config, spectrogram_config) -> spectrogram
- prepare_audio(file_path, config) -> spectrogram

CRITICAL: The original fastai_audio uses sample_rate=16000 (torchaudio default)
for MelSpectrogram even though audio is resampled to 20kHz. This is replicated
here for exact parity with the trained model.
"""

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


# =============================================================================
# Private Helper Functions
# =============================================================================


def _downmix_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Downmix multi-channel audio to mono.

    Args:
        waveform: Audio tensor of shape (channels, samples)

    Returns:
        Mono audio tensor of shape (1, samples)
    """
    if waveform.shape[0] > 1:
        return waveform.mean(dim=0, keepdim=True)
    return waveform


def _resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """
    Resample audio to target sample rate using scipy's polyphase resampling.

    Args:
        waveform: Audio tensor of shape (channels, samples)
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio tensor
    """
    if orig_sr == target_sr:
        return waveform

    sr_gcd = math.gcd(orig_sr, target_sr)
    sig_np = waveform.numpy()
    resampled = resample_poly(sig_np, int(target_sr / sr_gcd), int(orig_sr / sr_gcd), axis=-1)
    return torch.from_numpy(resampled.astype(np.float32))


def _compute_mel_spectrogram(
    waveform: torch.Tensor,
    spectrogram_config: Dict,
) -> torch.Tensor:
    """
    Compute mel spectrogram from waveform.

    Args:
        waveform: Audio tensor of shape (channels, samples)
        spectrogram_config: Dict with keys: sample_rate, n_fft, hop_length,
            mel_n_filters, mel_f_min, mel_f_max, mel_f_pad, convert_to_db, top_db

    Returns:
        Mel spectrogram tensor in dB scale (if convert_to_db=True)
    """
    mel_transform = MelSpectrogram(
        sample_rate=spectrogram_config["sample_rate"],
        n_fft=spectrogram_config["n_fft"],
        hop_length=spectrogram_config["hop_length"],
        n_mels=spectrogram_config["mel_n_filters"],
        f_min=spectrogram_config["mel_f_min"],
        f_max=spectrogram_config["mel_f_max"],
        pad=spectrogram_config["mel_f_pad"],
    )

    mel_spec = mel_transform(waveform)

    if spectrogram_config.get("convert_to_db", True):
        amplitude_to_db = AmplitudeToDB(top_db=spectrogram_config["top_db"])
        mel_spec = amplitude_to_db(mel_spec)

    return mel_spec.detach()


# =============================================================================
# Public API
# =============================================================================


def load_audio(file_path: str, audio_config: Dict) -> Tuple[torch.Tensor, int]:
    """
    Load audio file with preprocessing (downmix, resample).

    Args:
        file_path: Path to audio file
        audio_config: Dict with keys:
            - downmix_mono: bool - whether to downmix to mono
            - resample_rate: int - target sample rate (e.g., 20000)

    Returns:
        Tuple of (waveform tensor [1, samples], sample_rate)
        Sample rate is always the target resample_rate.
    """
    # Load audio using soundfile
    data, orig_sr = sf.read(str(file_path), dtype="float32")

    # Convert to torch tensor with shape (channels, samples)
    if data.ndim == 1:
        waveform = torch.from_numpy(data.reshape(1, -1))
    else:
        waveform = torch.from_numpy(data.T)

    # Downmix to mono if requested
    if audio_config.get("downmix_mono", True):
        waveform = _downmix_to_mono(waveform)

    # Resample to target rate
    target_sr = audio_config["resample_rate"]
    waveform = _resample_audio(waveform, orig_sr, target_sr)

    return waveform, target_sr


def featurize_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    spectrogram_config: Dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract mel spectrogram features from audio waveform.

    Args:
        waveform: Audio tensor of shape (channels, samples)
        sample_rate: Sample rate of the waveform (used for time axis calculation)
        spectrogram_config: Dict with keys:
            - sample_rate: int - sample rate for mel filterbank (16000 for fastai parity)
            - n_fft: int - FFT size (2560)
            - hop_length: int - hop between frames (256)
            - mel_n_filters: int - number of mel bins (256)
            - mel_f_min: float - minimum frequency (0.0)
            - mel_f_max: float - maximum frequency (10000.0)
            - mel_f_pad: int - padding (0)
            - convert_to_db: bool - convert to dB scale (True)
            - top_db: float - top dB value (100)

    Returns:
        Tuple of (features, times, freqs):
            - features: Mel spectrogram tensor, shape (1, n_mels, n_frames)
            - times: 1D tensor of time values for each frame center (seconds)
            - freqs: 1D tensor of mel bin center frequencies (Hz)
    """
    features = _compute_mel_spectrogram(waveform, spectrogram_config)

    # Compute time axis (frame centers in seconds)
    n_frames = features.shape[-1]
    hop_length = spectrogram_config["hop_length"]
    times = torch.arange(n_frames) * hop_length / sample_rate

    # Compute frequency axis (mel bin centers)
    # Using linear spacing in mel scale between f_min and f_max
    n_mels = spectrogram_config["mel_n_filters"]
    f_min = spectrogram_config["mel_f_min"]
    f_max = spectrogram_config["mel_f_max"]

    # Convert to mel scale, create linear spacing, convert back
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels)
    freqs = torch.from_numpy(mel_to_hz(mel_points).astype(np.float32))

    return features, times, freqs


def standardize(
    spectrogram: torch.Tensor,
    model_config: Dict,
    spectrogram_config: Dict,
) -> torch.Tensor:
    """
    Pad or crop spectrogram to fixed frame count for model input.

    Args:
        spectrogram: Tensor of shape (channels, n_mels, n_frames)
        model_config: Dict with key:
            - input_pad_s: float - target duration in seconds (e.g., 4.0)
        spectrogram_config: Dict with keys:
            - hop_length: int - hop between frames (256)
            Also needs audio resample_rate from audio_config (passed via spectrogram_config
            or model_config should include it)

    Returns:
        Spectrogram padded/cropped to exact frame count

    Note:
        KNOWN ISSUE: fastai_audio pads dB-scale spectrograms with 0.0 dB, which
        represents full-scale signal (power=1.0), not silence. This is preserved
        for exact parity with the trained model.
    """
    # Calculate target frames from duration
    # target_frames = input_pad_s * resample_rate / hop_length
    input_pad_s = model_config["input_pad_s"]
    hop_length = spectrogram_config["hop_length"]

    # Get resample_rate - it may be in model_config or we use standard 20000
    resample_rate = model_config.get("resample_rate", 20000)

    target_frames = int(input_pad_s * resample_rate / hop_length)
    current_frames = spectrogram.shape[-1]

    if current_frames > target_frames:
        # Crop from start
        spectrogram = spectrogram[..., :target_frames]
    elif current_frames < target_frames:
        # Pad with zeros at end (matches fastai_audio pad_mode="zeros-after")
        padding = target_frames - current_frames
        spectrogram = torch.nn.functional.pad(
            spectrogram, (0, padding), mode="constant", value=0
        )

    return spectrogram


def prepare_audio(file_path: str, config: Dict) -> torch.Tensor:
    """
    Complete audio preprocessing pipeline.

    Args:
        file_path: Path to audio file
        config: Full config dict with sections:
            - audio: {downmix_mono, resample_rate}
            - spectrogram: {sample_rate, n_fft, hop_length, mel_n_filters,
                           mel_f_min, mel_f_max, mel_f_pad, convert_to_db, top_db}
            - model: {input_pad_s}

    Returns:
        Mel spectrogram tensor ready for model inference, shape (1, n_mels, target_frames)
    """
    audio_config = config["audio"]
    spectrogram_config = config["spectrogram"]
    model_config = config["model"]

    # Add resample_rate to model_config for standardize()
    model_config_with_sr = {**model_config, "resample_rate": audio_config["resample_rate"]}

    # Load and preprocess audio
    waveform, sample_rate = load_audio(file_path, audio_config)

    # Extract mel spectrogram features
    features, times, freqs = featurize_waveform(waveform, sample_rate, spectrogram_config)

    # Standardize to fixed frame count
    features = standardize(features, model_config_with_sr, spectrogram_config)

    return features
