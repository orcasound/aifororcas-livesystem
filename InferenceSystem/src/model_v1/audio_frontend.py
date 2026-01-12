"""
Audio Frontend - Replaces fastai_audio for audio preprocessing

This module provides audio loading and mel spectrogram generation that matches
the fastai_audio implementation exactly for inference parity.

CRITICAL: The original fastai_audio uses sample_rate=16000 (torchaudio default)
for MelSpectrogram even though audio is resampled to 20kHz. This is replicated
here for exact parity with the trained model.
"""

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample
from pathlib import Path
from typing import Tuple, Optional
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
import math


# Audio configuration matching fastai_inference.py
AUDIO_CONFIG = {
    "target_sample_rate": 20000,  # Resample all audio to this rate
    "duration_ms": 4000,          # Pad/trim to 4 seconds
}

# Spectrogram configuration matching fastai_audio
# CRITICAL: sample_rate=16000 is the torchaudio default, NOT 20000
# This is a quirk in the original code that must be preserved for parity
SPECTROGRAM_CONFIG = {
    "sample_rate": 16000,  # torchaudio default (NOT the actual audio rate!)
    "n_fft": 2560,
    "hop_length": 256,
    "n_mels": 256,
    "f_min": 0.0,
    "f_max": 10000.0,
    "pad": 0,
    "top_db": 100,
}


def load_audio(file_path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio file using soundfile (same backend as patched torchaudio in fastai_inference).

    Args:
        file_path: Path to WAV file

    Returns:
        Tuple of (waveform tensor [channels, samples], sample_rate)
    """
    # Use soundfile directly (matches patched torchaudio.load in fastai_inference.py)
    data, sample_rate = sf.read(str(file_path), dtype='float32')

    # Convert to torch tensor with shape (channels, samples)
    if data.ndim == 1:
        waveform = torch.from_numpy(data.reshape(1, -1))
    else:
        waveform = torch.from_numpy(data.T)

    return waveform, sample_rate


def downmix_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Downmix multi-channel audio to mono.

    Matches fastai_audio's tfm_downmix behavior.

    Args:
        waveform: Audio tensor of shape (channels, samples)

    Returns:
        Mono audio tensor of shape (1, samples)
    """
    if waveform.shape[0] > 1:
        # Average all channels
        return waveform.mean(dim=0, keepdim=True)
    return waveform


def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int = AUDIO_CONFIG["target_sample_rate"]
) -> torch.Tensor:
    """
    Resample audio to target sample rate using scipy's polyphase resampling.

    Uses the same resampling method as fastai_audio (scipy.signal.resample_poly)
    for exact parity.

    Args:
        waveform: Audio tensor of shape (channels, samples)
        orig_sr: Original sample rate
        target_sr: Target sample rate (default: 20000)

    Returns:
        Resampled audio tensor
    """
    if orig_sr == target_sr:
        return waveform

    # Use scipy's polyphase resampling (same as fastai_audio's tfm_resample)
    sr_gcd = math.gcd(orig_sr, target_sr)
    sig_np = waveform.numpy()
    resampled = resample_poly(sig_np, int(target_sr/sr_gcd), int(orig_sr/sr_gcd), axis=-1)
    return torch.from_numpy(resampled)


def pad_or_trim(
    waveform: torch.Tensor,
    duration_ms: int = AUDIO_CONFIG["duration_ms"],
    sample_rate: int = AUDIO_CONFIG["target_sample_rate"],
    pad_mode: str = "zeros"
) -> torch.Tensor:
    """
    Pad or trim audio to exact duration.

    Matches fastai_audio's tfm_padtrim_signal and tfm_crop_time behavior.

    Args:
        waveform: Audio tensor of shape (channels, samples)
        duration_ms: Target duration in milliseconds
        sample_rate: Sample rate of the audio
        pad_mode: Padding mode ("zeros" for zero-padding)

    Returns:
        Audio tensor padded/trimmed to exact duration
    """
    target_samples = int(sample_rate * duration_ms / 1000)
    current_samples = waveform.shape[-1]

    if current_samples < target_samples:
        # Pad with zeros
        padding = target_samples - current_samples
        if pad_mode == "zeros":
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)
        else:
            # Replicate padding as fallback
            waveform = torch.nn.functional.pad(waveform, (0, padding), mode='replicate')
    elif current_samples > target_samples:
        # Trim to target length
        waveform = waveform[..., :target_samples]

    return waveform


def compute_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """
    Compute mel spectrogram matching fastai_audio's create_spectro method.

    CRITICAL: Uses sample_rate=16000 (torchaudio default) NOT the actual 20kHz
    audio rate. This is a quirk in the original fastai_audio code that must be
    preserved for exact parity with the trained model.

    Args:
        waveform: Audio tensor of shape (channels, samples)

    Returns:
        Mel spectrogram tensor in dB scale
    """
    # Create MelSpectrogram transform with exact parameters
    # Note: sample_rate uses the default 16000, NOT the actual audio rate
    mel_transform = MelSpectrogram(
        sample_rate=SPECTROGRAM_CONFIG["sample_rate"],
        n_fft=SPECTROGRAM_CONFIG["n_fft"],
        hop_length=SPECTROGRAM_CONFIG["hop_length"],
        n_mels=SPECTROGRAM_CONFIG["n_mels"],
        f_min=SPECTROGRAM_CONFIG["f_min"],
        f_max=SPECTROGRAM_CONFIG["f_max"],
        pad=SPECTROGRAM_CONFIG["pad"],
    )

    # Convert to dB scale
    amplitude_to_db = AmplitudeToDB(top_db=SPECTROGRAM_CONFIG["top_db"])

    # Apply transforms
    mel_spec = mel_transform(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)

    return mel_spec_db.detach()


# =============================================================================
# STAGE B: Audio to Mel Spectrogram (before standardization)
# =============================================================================

def audio_to_mel_spectrogram(
    audio: torch.Tensor,
    orig_sr: int,
    target_sr: int = AUDIO_CONFIG["target_sample_rate"],
) -> torch.Tensor:
    """
    STAGE B: Convert raw audio to mel spectrogram (pure mel computation).

    This stage performs:
    1. Downmix to mono (if needed)
    2. Resample to target sample rate (20kHz)
    3. Compute mel spectrogram
    4. Convert to dB scale

    The output has VARIABLE frame count depending on input audio length.
    For 2-second audio at 20kHz: ~156 frames
    For 4-second audio at 20kHz: ~313 frames

    Use standardize_spectrogram() (Stage C) to get fixed 312 frames.

    Args:
        audio: Raw audio tensor of shape (channels, samples)
        orig_sr: Original sample rate of the audio
        target_sr: Target sample rate (default: 20000)

    Returns:
        Mel spectrogram tensor in dB scale, shape (1, 256, variable_frames)
    """
    # Step 1: Downmix to mono
    audio = downmix_to_mono(audio)

    # Step 2: Resample to target rate
    audio = resample_audio(audio, orig_sr, target_sr)

    # Step 3 & 4: Compute mel spectrogram (includes dB conversion)
    mel_spec = compute_mel_spectrogram(audio)

    return mel_spec


def audio_to_mel_spectrogram_from_file(
    file_path: str,
    target_sr: int = AUDIO_CONFIG["target_sample_rate"],
) -> torch.Tensor:
    """
    STAGE B convenience wrapper: Load file and convert to mel spectrogram.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 20000)

    Returns:
        Mel spectrogram tensor in dB scale, shape (1, 256, variable_frames)
    """
    audio, orig_sr = load_audio(file_path)
    return audio_to_mel_spectrogram(audio, orig_sr, target_sr)


# =============================================================================
# STAGE C: Spectrogram Standardization
# =============================================================================

def standardize_spectrogram(
    spectrogram: torch.Tensor,
    target_frames: int = 312,
    pad_mode: str = "zeros-after",
) -> torch.Tensor:
    """
    STAGE C: Crop or pad spectrogram to exact frame count for model input.

    This stage standardizes the spectrogram dimensions for the model.

    KNOWN ISSUE: fastai_audio pads dB-scale spectrograms with 0.0 dB, which
    represents full-scale signal (power=1.0), not silence. This is technically
    incorrect but must be preserved for exact parity with the trained model.

    For 4000ms at 20kHz with hop=256: int(20000 * 4 / 256) = 312 frames

    Args:
        spectrogram: Tensor of shape (channels, n_mels, time_frames)
        target_frames: Target number of time frames (default: 312)
        pad_mode: "zeros-after" (deterministic) or "zeros" (random, for training)

    Returns:
        Spectrogram cropped/padded to exact frame count
    """
    current_frames = spectrogram.shape[-1]

    if current_frames > target_frames:
        # Crop to target (take from start, matching tfm_crop_time with crop_start=0)
        spectrogram = spectrogram[..., :target_frames]
    elif current_frames < target_frames:
        # Pad with zeros - matches tfm_pad_spectro with pad_mode="zeros-after"
        # NOTE: Padding with 0.0 dB is incorrect but matches fastai_audio behavior
        padding = target_frames - current_frames
        if pad_mode == "zeros-after":
            # Zeros at end only (deterministic)
            spectrogram = torch.nn.functional.pad(spectrogram, (0, padding), mode='constant', value=0)
        else:
            # Random placement (for training compatibility)
            import random
            zeros_front = random.randint(0, padding)
            zeros_back = padding - zeros_front
            spectrogram = torch.nn.functional.pad(spectrogram, (zeros_front, zeros_back), mode='constant', value=0)

    return spectrogram


# =============================================================================
# FULL PIPELINE: Combines Stage B + Stage C
# =============================================================================

def prepare_audio(file_path: str, target_frames: int = 312) -> torch.Tensor:
    """
    Complete audio preprocessing pipeline matching fastai_audio.

    Combines audio loading, padding, Stage B (mel spectrogram), and Stage C (standardization).

    Args:
        file_path: Path to WAV file
        target_frames: Target frame count for Stage C (default: 312)

    Returns:
        Mel spectrogram tensor ready for model inference, shape (1, 256, target_frames)
    """
    # Load and pad audio to 4 seconds
    audio, orig_sr = load_audio(file_path)
    audio = downmix_to_mono(audio)
    audio = resample_audio(audio, orig_sr, AUDIO_CONFIG["target_sample_rate"])
    audio = pad_or_trim(audio, AUDIO_CONFIG["duration_ms"], AUDIO_CONFIG["target_sample_rate"])

    # Stage B: Audio to mel spectrogram
    mel_spec = compute_mel_spectrogram(audio)

    # Stage C: Standardize to fixed frame count
    mel_spec = standardize_spectrogram(mel_spec, target_frames)

    return mel_spec


class AudioItem:
    """
    Container for processed audio data, similar to fastai_audio's AudioItem.

    Attributes:
        path: Path to the source audio file
        spectro: Mel spectrogram tensor
        sig: Raw audio signal (optional)
        sr: Sample rate
    """

    def __init__(
        self,
        path: str,
        spectro: Optional[torch.Tensor] = None,
        sig: Optional[torch.Tensor] = None,
        sr: int = AUDIO_CONFIG["target_sample_rate"]
    ):
        self.path = Path(path)
        self.spectro = spectro
        self.sig = sig
        self.sr = sr

    @classmethod
    def from_file(cls, file_path: str) -> "AudioItem":
        """
        Create AudioItem from a WAV file with full preprocessing.

        Args:
            file_path: Path to WAV file

        Returns:
            AudioItem with mel spectrogram
        """
        spectro = prepare_audio(file_path)
        return cls(path=file_path, spectro=spectro)

    def __repr__(self) -> str:
        shape = self.spectro.shape if self.spectro is not None else None
        return f"AudioItem(path={self.path.name}, spectro_shape={shape})"


def process_audio_folder(folder_path: str) -> list:
    """
    Process all WAV files in a folder, similar to AudioList.from_folder.

    Args:
        folder_path: Path to folder containing WAV files

    Returns:
        List of AudioItem objects
    """
    folder = Path(folder_path)
    audio_files = list(folder.glob("*.wav")) + list(folder.glob("*.WAV"))

    items = []
    for audio_file in sorted(audio_files):
        try:
            item = AudioItem.from_file(str(audio_file))
            items.append(item)
        except Exception as e:
            print(f"Warning: Failed to process {audio_file}: {e}")
            continue

    return items
