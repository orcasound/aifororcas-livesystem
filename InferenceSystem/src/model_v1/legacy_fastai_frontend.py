"""
Legacy FastAI Audio Frontend - For generating reference outputs in tests

This module wraps fastai_audio functionality for generating reference spectrograms
used in parity testing. It isolates all fastai_audio imports to this file.

Only use in inference-venv (legacy environment with fastai_audio installed).
"""

import shutil
import tempfile
from pathlib import Path
from typing import Dict

import torch
import torchaudio

# Monkey-patch torchaudio to avoid torchcodec dependency
_original_torchaudio_load = torchaudio.load


def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Wrapper for torchaudio.load that uses soundfile directly"""
    import soundfile as sf

    data, samplerate = sf.read(str(filepath), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    return waveform, samplerate


torchaudio.load = _patched_torchaudio_load


def prepare_audio(file_path: str, config: Dict) -> torch.Tensor:
    """
    Process audio file using fastai_audio pipeline.

    This matches the processing in fastai_inference.py FastAIModel.predict()
    but for a single file rather than batch processing.

    Args:
        file_path: Path to audio file (typically a 2-second segment)
        config: Full config dict with sections:
            - audio: {downmix_mono, resample_rate}
            - spectrogram: {sample_rate, n_fft, hop_length, mel_n_filters,
                           mel_f_min, mel_f_max, mel_f_pad, convert_to_db, top_db}
            - model: {input_pad_s}

    Returns:
        Mel spectrogram tensor from fastai_audio, shape (1, n_mels, n_frames)
    """
    # Import fastai_audio components (only available in inference-venv)
    from audio.data import AudioConfig, AudioList, SpectrogramConfig

    audio_config = config["audio"]
    spectrogram_config = config["spectrogram"]
    model_config = config["model"]

    # Build SpectrogramConfig from our config dict
    sg_cfg = SpectrogramConfig(
        f_min=spectrogram_config["mel_f_min"],
        f_max=spectrogram_config["mel_f_max"],
        hop_length=spectrogram_config["hop_length"],
        n_fft=spectrogram_config["n_fft"],
        n_mels=spectrogram_config["mel_n_filters"],
        pad=spectrogram_config["mel_f_pad"],
        to_db_scale=spectrogram_config.get("convert_to_db", True),
        top_db=spectrogram_config["top_db"],
        win_length=None,
        n_mfcc=20,
    )

    # Build AudioConfig
    fastai_config = AudioConfig(standardize=False, sg_cfg=sg_cfg)
    fastai_config.duration = int(model_config["input_pad_s"] * 1000)  # seconds to ms
    fastai_config.resample_to = audio_config["resample_rate"]
    fastai_config.downmix = audio_config.get("downmix_mono", True)
    fastai_config.pad_mode = "zeros-after"  # Deterministic padding

    # Process single file via AudioList
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy file to temp directory (AudioList.from_folder needs a directory)
        shutil.copy(file_path, tmpdir)

        # Create AudioList and process
        test = AudioList.from_folder(tmpdir, config=fastai_config).split_none().label_empty()
        testdb = test.transform(None).databunch(bs=1)

        # Get spectrogram from first (only) item
        spectrogram = testdb.x[0].spectro.clone()

    return spectrogram


def prepare_audio_stage_b(file_path: str, config: Dict) -> torch.Tensor:
    """
    Process audio file to Stage B (mel spectrogram before standardization).

    This generates a pure mel spectrogram from the audio without padding/cropping.
    Used for Stage B parity testing.

    Args:
        file_path: Path to audio file (typically a 2-second segment)
        config: Full config dict with sections:
            - audio: {downmix_mono, resample_rate}
            - spectrogram: {sample_rate, n_fft, hop_length, mel_n_filters,
                           mel_f_min, mel_f_max, mel_f_pad, convert_to_db, top_db}

    Returns:
        Mel spectrogram tensor, shape (1, n_mels, variable_frames)
    """
    from audio.transform import tfm_downmix, tfm_resample
    from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

    audio_config = config["audio"]
    spectrogram_config = config["spectrogram"]

    # Load audio
    sig, sr = torchaudio.load(file_path)

    # Downmix to mono (if stereo)
    if audio_config.get("downmix_mono", True) and sig.shape[0] > 1:
        sig = tfm_downmix(sig)

    # Resample to target rate
    target_sr = audio_config["resample_rate"]
    if sr != target_sr:
        sig = tfm_resample(sig, sr, target_sr)

    # Create mel spectrogram (Stage B - pure mel computation)
    # CRITICAL: Don't pass sample_rate - uses default 16000 (fastai quirk)
    mel_transform = MelSpectrogram(
        n_fft=spectrogram_config["n_fft"],
        hop_length=spectrogram_config["hop_length"],
        n_mels=spectrogram_config["mel_n_filters"],
        f_min=spectrogram_config["mel_f_min"],
        f_max=spectrogram_config["mel_f_max"],
        pad=spectrogram_config["mel_f_pad"],
    )

    mel_spec = mel_transform(sig)

    if spectrogram_config.get("convert_to_db", True):
        amplitude_to_db = AmplitudeToDB(top_db=spectrogram_config["top_db"])
        mel_spec = amplitude_to_db(mel_spec)

    return mel_spec.clone()
