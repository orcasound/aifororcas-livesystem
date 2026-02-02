"""
Audio Frontend for model_v1 - Replaces fastai_audio for audio preprocessing

This module provides audio loading and mel spectrogram generation that matches
the fastai_audio implementation exactly for inference parity.

Interfaces:
- load_audio(file_path, audio_config) -> (waveform, sample_rate)
- featurize_waveform(waveform, sample_rate, spectrogram_config) -> (features, times, freqs)
- standardize(spectrogram, model_config, spectrogram_config) -> spectrogram
- prepare_audio(file_path, config) -> spectrogram
- audio_segment_generator(audio_file_path, segment_duration_s, segment_hop_s, ...) -> Generator

CRITICAL: The original fastai_audio uses sample_rate=16000 (torchaudio default)
for MelSpectrogram even though audio is resampled to 20kHz. This is replicated
here for exact parity with the trained model.
"""

import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from librosa import get_duration
from pydub import AudioSegment
from scipy.signal import resample_poly
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from .types import DetectorInferenceConfig


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


@contextmanager
def _temp_segment_dir(output_dir: Optional[str] = None):
    """Context manager for temporary segment directory."""
    if output_dir is not None:
        # Use provided directory
        yield output_dir
    else:
        # Create and cleanup temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir


def audio_segment_generator(
    audio_file_path: str,
    segment_duration_s: float,
    segment_hop_s: float,
    output_dir: Optional[str] = None,
    max_segments: Optional[int] = None,
    start_time_s: float = 0.0,
    strict_segments: bool = True,
) -> Generator[Tuple[str, float, float], None, None]:
    """
    Generate overlapping audio segments from an audio file.

    Yields generator of segments as they are created to handle large audio files.
    Automatically creates and cleans up a temporary directory unless output_dir is explicitly provided.

    Args:
        audio_file_path: Path to input audio file
        segment_duration_s: Duration of each segment in seconds (e.g., 2.0)
        segment_hop_s: Hop/stride between segment starts in seconds (e.g., 1.0)
        output_dir: Optional directory to save segments. If None, creates temporary
                    directory that is cleaned up after generation completes.
        max_segments: Optional limit on number of segments (useful for testing)
        start_time_s: Start time offset in seconds (default: 0.0)
        strict_segments: If True (default), only generate segments that fit completely
                        within audio duration. If False, allow final partial segment
                        that may extend beyond audio duration (matches fastai behavior).

    Yields:
        Tuple of (segment_path, start_s, end_s):
            - segment_path: Path to the generated audio segment file
            - start_s: Start time of the segment in seconds (float)
            - end_s: End time of the segment in seconds (float)

    Example:
        >>> for segment_path, start_s, end_s in audio_segment_generator("audio.wav", 2.0, 1.0):
        ...     mel_spec = prepare_audio(segment_path, config)
        ...     print(f"Segment {start_s:.1f}-{end_s:.1f}s")
    """
    with _temp_segment_dir(output_dir) as segment_dir:
        # Get audio duration
        audio_duration = get_duration(path=audio_file_path)
        wav_name = Path(audio_file_path).stem

        # Load audio
        audio = AudioSegment.from_wav(audio_file_path)

        # Calculate number of segments
        num_segments = int(np.floor((audio_duration - start_time_s) / segment_hop_s))
        if max_segments is not None:
            num_segments = min(max_segments, num_segments)

        # Generate segments
        for i in range(num_segments):
            start_s = i * segment_hop_s + start_time_s
            end_s = start_s + segment_duration_s

            # In strict mode, stop if segment extends beyond audio
            # In non-strict mode, allow partial segments (pydub handles gracefully)
            if strict_segments and end_s > audio_duration:
                break

            # Export segment (pydub needs milliseconds as integers)
            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            segment_path = f"{segment_dir}/{wav_name}_{start_ms:06d}_{end_ms:06d}.wav"
            segment = audio[start_ms:end_ms]
            segment.export(segment_path, format="wav")

            yield segment_path, start_s, end_s


class AudioPreprocessor:
    """
    Config-driven wrapper around audio_segment_generator + prepare_audio.

    Args:
        config: DetectorInferenceConfig or dict with audio/spectrogram/model/inference sections
    """

    def __init__(self, config):
        if isinstance(config, DetectorInferenceConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = DetectorInferenceConfig.from_dict(config)
        else:
            raise TypeError(
                f"config must be DetectorInferenceConfig or dict, got {type(config)}"
            )

    def process_segments(
        self,
        audio_file_path: str,
    ) -> Generator[Tuple[torch.Tensor, float, float], None, None]:
        """
        Generate preprocessed mel spectrogram segments from an audio file.

        Wraps audio_segment_generator() + prepare_audio() into a single
        generator that yields ready-to-infer tensors.

        Args:
            audio_file_path: Path to input audio file

        Yields:
            Tuple of (mel_spectrogram, start_time_s, duration_s):
                - mel_spectrogram: tensor of shape (1, n_mels, target_frames)
                - start_time_s: start time of this segment in seconds
                - duration_s: duration of this segment in seconds
        """
        inf = self.config.inference
        config_dict = self.config.as_dict()

        for segment_path, start_s, end_s in audio_segment_generator(
            audio_file_path,
            segment_duration_s=inf.window_s,
            segment_hop_s=inf.window_hop_s,
            strict_segments=inf.strict_segments,
        ):
            mel_spec = prepare_audio(segment_path, config_dict)
            yield mel_spec, start_s, inf.window_s
