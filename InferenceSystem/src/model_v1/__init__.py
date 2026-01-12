"""
model_v1 - Orca Detection Model without fastai dependencies

This module provides a fastai-free implementation of the OrcaHello SRKW detection model.
It uses pure PyTorch and torchaudio for audio processing and model inference.

Usage:
    from model_v1.inference import OrcaDetectionModel

    model = OrcaDetectionModel(model_path="./model", threshold=0.5)
    result = model.predict("path/to/audio.wav")
"""

__version__ = "1.0.0"
