import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torchvision.models import resnet50
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from torch import Tensor

from .audio_frontend import audio_segment_generator, prepare_audio


@dataclass
class SegmentPrediction:
    start_time_s: float
    duration_s: float
    confidence: float

@dataclass
class DetectionResult:
    """
    Detection result matching the fastai inference output format.
    
    Attributes:
        local_predictions: List of binary predictions (0/1) for each time segment
        local_confidences: List of confidence scores for each time segment
        global_prediction: Binary prediction (0/1) indicating if orca calls were detected
        global_confidence: Mean confidence score across positive detections (0-100)
    """
    local_predictions: List[int]
    local_confidences: List[float]
    global_prediction: int
    global_confidence: float
    segment_predictions: List[SegmentPrediction]
    wav_file_path: str



class AdaptiveConcatPool2d(nn.Module):
    """
    Adaptive pooling layer that concatenates max and average pooling.

    Matches fastai's AdaptiveConcatPool2d which outputs 2x the input channels.
    NOTE: FastAI concatenates [max_pool, avg_pool] in that order.
    """
    def __init__(self, output_size=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        # FastAI concatenates max first, then avg
        return torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)


class OrcaHelloSRKWDetector(nn.Module):
    """
    Wrapper of ResNet50 model for binary detection of individual orca calls
    from featurized audio segments (mel spectrograms).

    Architecture matches the production fastai model which uses:
    - ResNet50 backbone (3,4,6,3 Bottleneck blocks)
    - Single-channel input (grayscale spectrogram)
    - Custom head: AdaptiveConcatPool2d -> Linear(4096, 512) -> Linear(512, 2)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        # FastAI trained model has classes ['negative', 'positive']
        # So class index 1 is the "positive" (call detected) class
        self.call_class_index = 1
        self.model = resnet50()
        # Modify first conv layer for single-channel input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace avgpool and fc with fastai-style head
        # FastAI head structure:
        # [0] AdaptiveConcatPool2d - outputs 2048*2 = 4096 features
        # [1] Flatten
        # [2] BatchNorm1d(4096)
        # [3] Dropout(0.25)
        # [4] Linear(4096, 512)
        # [5] ReLU
        # [6] BatchNorm1d(512)
        # [7] Dropout(0.5)
        # [8] Linear(512, num_classes)
        self.model.avgpool = AdaptiveConcatPool2d(1)
        self.model.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.25),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor  # shape: (batch, 1, n_mels, time_frames)
    ) -> Tensor:                 # shape: (batch, num_classes)
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (batch, 1, n_mels, time_frames)

        Returns:
            Tensor: Logits tensor of shape (batch, num_classes)
        """
        return self.model(x)

    def predict_call(self, x: Tensor  # shape: (batch, 1, n_mels, time_frames)
    ) -> Tensor:                      # shape: (batch,)
        """
        Get class probabilities using softmax.

        Args:
            x (Tensor): Input tensor of shape (batch, 1, n_mels, time_frames)

        Returns:
            Tensor: Probability tensor of shape (batch,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)[:, self.call_class_index].squeeze()
    
    def detect_srkw_from_file(self, wav_file_path: str) -> DetectionResult:
        """
        TODO: Internally makes use of audio frontend functions and self.predict_call
        Skip the smoothed dataframe `submission` part in fastai_inference.py for now.
        """
        pass

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "OrcaHelloSRKWDetector":
        """
        Load model from PyTorch checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Device to load model onto ("cpu" or "cuda")

        Returns:
            OrcaHelloSRKWDetector with loaded weights in eval mode
        """
        model = cls()
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

