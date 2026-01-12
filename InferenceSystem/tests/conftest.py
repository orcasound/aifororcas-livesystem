"""
Pytest Configuration and Shared Fixtures for InferenceSystem tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

# Add src to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# Monkey-patch torchaudio to avoid torchcodec dependency
# (Same patch as in fastai_inference.py)
# =============================================================================

_original_torchaudio_load = torchaudio.load

def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Wrapper for torchaudio.load that uses soundfile directly"""
    import soundfile as sf
    data, samplerate = sf.read(str(filepath), dtype='float32')
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    return waveform, samplerate

torchaudio.load = _patched_torchaudio_load

_original_torchaudio_save = torchaudio.save

def _patched_torchaudio_save(filepath, src, sample_rate, *args, **kwargs):
    """Wrapper for torchaudio.save that uses soundfile directly"""
    import soundfile as sf
    audio_data = src.numpy().T if src.ndim > 1 else src.numpy().reshape(-1, 1)
    sf.write(str(filepath), audio_data, sample_rate)

torchaudio.save = _patched_torchaudio_save

# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "claude-scratch" / "test_data"

# Reference outputs directory (for parity testing)
REFERENCE_DIR = Path(__file__).parent / "reference_outputs"


@pytest.fixture
def test_data_dir():
    """Return path to test data directory"""
    assert TEST_DATA_DIR.exists(), f"Test data directory not found: {TEST_DATA_DIR}"
    return TEST_DATA_DIR


@pytest.fixture
def reference_dir():
    """Return path to reference outputs directory"""
    REFERENCE_DIR.mkdir(exist_ok=True)
    return REFERENCE_DIR


@pytest.fixture
def sample_1min_wav(test_data_dir):
    """
    Return path to a 1-minute test WAV file.
    """
    wav_files = sorted(list(test_data_dir.glob("*.wav")))
    assert len(wav_files) > 0, "No WAV files found in test_data directory"
    return str(wav_files[0])


@pytest.fixture
def all_test_wavs(test_data_dir):
    """Return paths to all test WAV files"""
    wav_files = list(test_data_dir.glob("*.wav"))
    return [str(f) for f in wav_files]


@pytest.fixture
def numerical_tolerance():
    """Return numerical tolerance settings for comparisons"""
    return {
        "atol": 1e-5,  # Absolute tolerance
        "rtol": 1e-5,  # Relative tolerance
    }


@pytest.fixture
def model_dir():
    """Return path to model directory"""
    model_path = Path(__file__).parent.parent / "model"
    if model_path.exists():
        return model_path
    pytest.skip("Model directory not found")


@pytest.fixture
def fastai_available():
    """Check if fastai environment is available"""
    try:
        from audio.data import AudioConfig, AudioList, SpectrogramConfig
        from fastai.basic_train import load_learner
        return True
    except ImportError:
        return False


def pytest_configure(config):
    """Pytest configuration hook"""
    config.addinivalue_line(
        "markers", "parity: marks tests that compare fastai vs model_v1"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that take a long time"
    )
    config.addinivalue_line(
        "markers", "requires_fastai: marks tests that need fastai environment"
    )
