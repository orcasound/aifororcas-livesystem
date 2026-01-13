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

    def __init__(self, config: Optional[Dict] = None, num_classes: Optional[int] = None):
        super().__init__()

        # Load config values (prioritize explicit args over config dict, then defaults)
        # Extract from config if provided
        if config is not None:
            model_config = config.get('model', {})
            config_num_classes = model_config.get('num_classes', 2)
            config_call_class_index = model_config.get('call_class_index', 1)
        else:
            config_num_classes = 2
            config_call_class_index = 1

        # Explicit args override config, config overrides defaults
        self.num_classes = num_classes if num_classes is not None else config_num_classes
        # FastAI trained model has classes ['negative', 'positive']
        # So class index 1 is the "positive" (call detected) class
        self.call_class_index = config_call_class_index

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
            nn.Linear(512, self.num_classes),
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
    
    def detect_srkw_from_file(self, wav_file_path: str, config: Dict) -> DetectionResult:
        """
        Detect SRKW calls from a WAV file.

        This method:
        1. Segments the audio file using configured window and hop parameters
        2. Generates mel spectrograms for each segment
        3. Runs inference using predict_call() on each segment
        4. Aggregates results into local and global predictions

        Args:
            wav_file_path: Path to the WAV file
            config: Configuration dict with audio, spectrogram, and inference parameters

        Returns:
            DetectionResult with local predictions, confidences, and global prediction
        """
        # Extract config params
        inference_config = config.get('inference', {})
        local_conf_threshold = inference_config.get('local_conf_threshold', 0.5)
        global_pred_threshold = inference_config.get('global_pred_threshold', 3)
        segment_duration_s = inference_config.get('window_s', 2.0)
        segment_hop_s = inference_config.get('window_hop_s', 1.0)

        # Generate segments and process into spectrograms
        spectrograms = []
        segment_info = []  # Track start times and durations

        for segment_path in audio_segment_generator(
            wav_file_path,
            segment_duration_s=segment_duration_s,
            segment_hop_s=segment_hop_s
        ):
            # Extract start time from filename: "basename_start_end.wav"
            filename = Path(segment_path).stem
            parts = filename.split('_')
            start_s = int(parts[-2])

            # Convert segment to spectrogram
            mel_spec = prepare_audio(segment_path, config)
            spectrograms.append(mel_spec)
            segment_info.append({
                'start_time_s': float(start_s),
                'duration_s': segment_duration_s
            })

        if len(spectrograms) == 0:
            # No segments generated - return empty result
            return DetectionResult(
                local_predictions=[],
                local_confidences=[],
                global_prediction=0,
                global_confidence=0.0,
                segment_predictions=[],
                wav_file_path=wav_file_path
            )

        # Stack spectrograms into batch: (num_segments, 1, n_mels, time_frames)
        spectrograms_batch = torch.stack(spectrograms)

        # Run inference on all segments
        confidences = self.predict_call(spectrograms_batch)

        # Convert to lists for output
        local_confidences = confidences.cpu().numpy().tolist()
        local_predictions = [1 if conf > local_conf_threshold else 0 for conf in local_confidences]

        # Calculate global prediction
        num_positive = sum(local_predictions)
        global_prediction = 1 if num_positive >= global_pred_threshold else 0

        # Calculate global confidence (mean of positive detections, scaled 0-100)
        if num_positive > 0:
            positive_confs = [conf for conf, pred in zip(local_confidences, local_predictions) if pred == 1]
            global_confidence = sum(positive_confs) / len(positive_confs) * 100
        else:
            global_confidence = 0.0

        # Build segment predictions
        segment_preds = []
        for seg_info, conf in zip(segment_info, local_confidences):
            segment_preds.append(
                SegmentPrediction(
                    start_time_s=seg_info['start_time_s'],
                    duration_s=seg_info['duration_s'],
                    confidence=conf
                )
            )

        return DetectionResult(
            local_predictions=local_predictions,
            local_confidences=local_confidences,
            global_prediction=global_prediction,
            global_confidence=global_confidence,
            segment_predictions=segment_preds,
            wav_file_path=wav_file_path
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: Optional[Dict] = None, device: str = "cpu") -> "OrcaHelloSRKWDetector":
        """
        Load model from PyTorch checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            config: Optional config dict with model parameters
            device: Device to load model onto ("cpu" or "cuda")

        Returns:
            OrcaHelloSRKWDetector with loaded weights in eval mode
        """
        model = cls(config=config)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

