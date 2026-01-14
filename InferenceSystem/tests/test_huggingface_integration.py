"""
Tests for HuggingFace Hub Integration

Tests the PyTorchModelHubMixin integration for OrcaHelloSRKWDetectorV1.
Verifies save_pretrained, from_pretrained, and model compatibility.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model_v1.inference import OrcaHelloSRKWDetectorV1


class TestHuggingFaceIntegration:
    """Tests for PyTorchModelHubMixin integration"""

    def test_save_pretrained_local(self, v1_config, model_dir, tmp_path):
        """Test save_pretrained creates expected files"""
        model_path = model_dir / "model_v1.pt"
        if not model_path.exists():
            pytest.skip("Checkpoint not found")

        model = OrcaHelloSRKWDetectorV1.from_checkpoint(str(model_path), v1_config)
        save_dir = tmp_path / "test_model"
        model.save_pretrained(save_dir)

        # Verify expected files exist
        assert (save_dir / "config.json").exists(), "config.json not created"

        # Check for either safetensors or pytorch_model.bin
        has_safetensors = (save_dir / "model.safetensors").exists()
        has_pytorch_bin = (save_dir / "pytorch_model.bin").exists()
        assert has_safetensors or has_pytorch_bin, "No model weights file created"

    def test_from_pretrained_local(self, v1_config, model_dir, tmp_path):
        """Test from_pretrained reloads model correctly"""
        model_path = model_dir / "model_v1.pt"
        if not model_path.exists():
            pytest.skip("Checkpoint not found")

        # Save model
        original = OrcaHelloSRKWDetectorV1.from_checkpoint(str(model_path), v1_config)
        save_dir = tmp_path / "test_model"
        original.save_pretrained(save_dir)

        # Load from saved directory
        loaded = OrcaHelloSRKWDetectorV1.from_pretrained(str(save_dir))

        # Verify predictions match
        x = torch.randn(1, 1, 256, 313)
        original_pred = original.predict_call(x).item()
        loaded_pred = loaded.predict_call(x).item()

        assert abs(original_pred - loaded_pred) < 1e-5, \
            f"Predictions differ: {original_pred} vs {loaded_pred}"

    def test_from_checkpoint_still_works(self, v1_config, model_dir):
        """Backward compatibility: from_checkpoint still works"""
        model_path = model_dir / "model_v1.pt"
        if not model_path.exists():
            pytest.skip("Checkpoint not found")

        model = OrcaHelloSRKWDetectorV1.from_checkpoint(str(model_path), v1_config)
        assert model is not None
        assert isinstance(model, OrcaHelloSRKWDetectorV1)

    def test_config_serialization(self, v1_config, model_dir, tmp_path):
        """Test that config is properly serialized and deserialized"""
        model_path = model_dir / "model_v1.pt"
        if not model_path.exists():
            pytest.skip("Checkpoint not found")

        # Save model
        original = OrcaHelloSRKWDetectorV1.from_checkpoint(str(model_path), v1_config)
        save_dir = tmp_path / "test_model"
        original.save_pretrained(save_dir)

        # Load and verify config exists
        import json
        config_file = save_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            saved_config = json.load(f)

        # Verify it's a valid JSON dict
        assert isinstance(saved_config, dict)
        # The exact structure depends on PyTorchModelHubMixin's serialization
        # At minimum, verify it's not empty
        assert len(saved_config) > 0

    def test_predict_call_after_reload(self, v1_config, model_dir, tmp_path):
        """Test that predict_call works correctly after save/load"""
        model_path = model_dir / "model_v1.pt"
        if not model_path.exists():
            pytest.skip("Checkpoint not found")

        # Create test input
        test_input = torch.randn(2, 1, 256, 313)  # Batch of 2

        # Get original predictions
        original = OrcaHelloSRKWDetectorV1.from_checkpoint(str(model_path), v1_config)
        original_preds = original.predict_call(test_input)

        # Save and reload
        save_dir = tmp_path / "test_model"
        original.save_pretrained(save_dir)
        loaded = OrcaHelloSRKWDetectorV1.from_pretrained(str(save_dir))
        loaded_preds = loaded.predict_call(test_input)

        # Compare predictions
        assert original_preds.shape == loaded_preds.shape
        assert torch.allclose(original_preds, loaded_preds, atol=1e-5), \
            "Predictions differ after reload"

    def test_model_architecture_preserved(self, v1_config, model_dir, tmp_path):
        """Test that model architecture is preserved after save/load"""
        model_path = model_dir / "model_v1.pt"
        if not model_path.exists():
            pytest.skip("Checkpoint not found")

        # Save and reload
        original = OrcaHelloSRKWDetectorV1.from_checkpoint(str(model_path), v1_config)
        save_dir = tmp_path / "test_model"
        original.save_pretrained(save_dir)
        loaded = OrcaHelloSRKWDetectorV1.from_pretrained(str(save_dir))

        # Verify key architectural properties
        assert loaded.num_classes == original.num_classes
        assert loaded.call_class_index == original.call_class_index

        # Verify model structure
        assert hasattr(loaded, 'model')
        assert hasattr(loaded.model, 'conv1')
        assert hasattr(loaded.model, 'fc')


@pytest.mark.slow
class TestHuggingFaceDetectSRKW:
    """Test detect_srkw_from_file works after HF save/load"""

    def test_detect_srkw_after_reload(self, v1_config, model_dir, sample_1min_wav, tmp_path):
        """Test that detect_srkw_from_file works after save/load"""
        model_path = model_dir / "model_v1.pt"
        if not model_path.exists():
            pytest.skip("Checkpoint not found")

        # Get original detection result
        original = OrcaHelloSRKWDetectorV1.from_checkpoint(str(model_path), v1_config)
        original_result = original.detect_srkw_from_file(sample_1min_wav, v1_config)

        # Save and reload
        save_dir = tmp_path / "test_model"
        original.save_pretrained(save_dir)
        loaded = OrcaHelloSRKWDetectorV1.from_pretrained(str(save_dir))
        loaded_result = loaded.detect_srkw_from_file(sample_1min_wav, v1_config)

        # Compare results
        assert original_result.global_prediction == loaded_result.global_prediction, \
            "Global predictions differ"
        assert abs(original_result.global_confidence - loaded_result.global_confidence) < 0.1, \
            f"Global confidence differs: {original_result.global_confidence} vs {loaded_result.global_confidence}"
        assert len(original_result.local_predictions) == len(loaded_result.local_predictions), \
            "Different number of segments"
