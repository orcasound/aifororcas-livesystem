"""
Model Inference Tests - Verify OrcaHelloSRKWDetectorV1 matches fastai

Test structure:
- TestOrcaHelloSRKWDetectorUnit: Unit tests for model class
- TestPredictCallParity: Parity tests comparing predict_call() to fastai
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestOrcaHelloSRKWDetectorUnit:
    """Unit tests for OrcaHelloSRKWDetectorV1 model class"""

    def test_model_creation(self, v1_config):
        """Test model instantiation"""
        from model_v1.inference import OrcaHelloSRKWDetectorV1

        model = OrcaHelloSRKWDetectorV1(v1_config)
        assert model is not None
        assert model.num_classes == 2
        # FastAI classes are ['negative', 'positive'], so class 1 is "positive"
        assert model.call_class_index == 1

    def test_model_creation_defaults(self):
        """Test model with empty config uses defaults"""
        from model_v1.inference import OrcaHelloSRKWDetectorV1

        model = OrcaHelloSRKWDetectorV1({})
        assert model.num_classes == 2
        assert model.call_class_index == 1

    def test_forward_pass_shape(self, v1_config):
        """Test forward() returns (batch, num_classes) logits"""
        from model_v1.inference import OrcaHelloSRKWDetectorV1

        model = OrcaHelloSRKWDetectorV1(v1_config)
        model.eval()

        # Standard input shape: (batch, 1, n_mels=256, time_frames=313)
        x = torch.randn(1, 1, 256, 313)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"

    def test_forward_pass_batch(self, v1_config):
        """Test forward() with batch > 1"""
        from model_v1.inference import OrcaHelloSRKWDetectorV1

        model = OrcaHelloSRKWDetectorV1(v1_config)
        model.eval()

        x = torch.randn(4, 1, 256, 313)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"

    def test_predict_call_shape(self, v1_config):
        """Test predict_call() returns (batch,) probabilities"""
        from model_v1.inference import OrcaHelloSRKWDetectorV1

        model = OrcaHelloSRKWDetectorV1(v1_config)
        model.eval()

        x = torch.randn(1, 1, 256, 313)
        prob = model.predict_call(x)

        # predict_call returns scalar for batch=1 due to squeeze()
        assert prob.dim() == 0 or prob.shape == (1,), f"Expected scalar or (1,), got {prob.shape}"

    def test_predict_call_batch(self, v1_config):
        """Test predict_call() with batch > 1"""
        from model_v1.inference import OrcaHelloSRKWDetectorV1

        model = OrcaHelloSRKWDetectorV1(v1_config)
        model.eval()

        x = torch.randn(4, 1, 256, 313)
        probs = model.predict_call(x)

        assert probs.shape == (4,), f"Expected (4,), got {probs.shape}"

    def test_predict_call_range(self, v1_config):
        """Test predict_call() returns values in [0, 1]"""
        from model_v1.inference import OrcaHelloSRKWDetectorV1

        model = OrcaHelloSRKWDetectorV1(v1_config)
        model.eval()

        # Run multiple random inputs
        for _ in range(5):
            x = torch.randn(4, 1, 256, 313)
            probs = model.predict_call(x)

            assert (probs >= 0).all(), f"Found negative probability: {probs}"
            assert (probs <= 1).all(), f"Found probability > 1: {probs}"

    def test_from_checkpoint_missing_file(self, v1_config):
        """Test from_checkpoint() raises error for missing file"""
        from model_v1.inference import OrcaHelloSRKWDetectorV1

        with pytest.raises(FileNotFoundError):
            OrcaHelloSRKWDetectorV1.from_checkpoint("/nonexistent/path/model.pt", v1_config)

    def test_from_checkpoint(self, model_dir, v1_config):
        """Test loading from checkpoint file"""
        from model_v1.inference import OrcaHelloSRKWDetectorV1

        model_path = model_dir / "model_v1.pt"
        if not model_path.exists():
            pytest.skip(f"Checkpoint not found: {model_path}. Run extraction script first.")

        model = OrcaHelloSRKWDetectorV1.from_checkpoint(str(model_path), v1_config)

        assert model is not None
        # Verify model is in eval mode
        assert not model.training

        # Verify forward pass works
        x = torch.randn(1, 1, 256, 313)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 2)


class TestPredictCallParity:
    """
    Parity tests comparing predict_call() output between fastai and model_v1.

    Uses fixed spectrograms as input to isolate model behavior from audio preprocessing.
    """

    def test_generate_reference(self, model_dir, reference_dir, fastai_available):
        """
        Generate reference outputs from fastai model for parity testing.

        Run this test in inference-venv (which has fastai) to create reference files.
        The reference file contains:
        - spectrograms: dict of synthetic spectrogram tensors
        - predictions: dict of fastai prediction values for each spectrogram
        """
        if not fastai_available:
            pytest.skip("fastai not available - run in inference-venv to generate references")

        from fastai.basic_train import load_learner
        import torch.nn.functional as F

        # Load fastai model
        learner = load_learner(str(model_dir), "model.pkl")

        # Create synthetic spectrograms (same shape as real inputs)
        # Shape: (1, n_mels=256, time_frames=313) - what fastai expects
        torch.manual_seed(42)  # Reproducible
        spectrograms = {
            f"synthetic_{i}": torch.randn(1, 256, 313)
            for i in range(5)
        }

        references = {
            "spectrograms": {},
            "predictions": {},
        }

        print("\nGenerating fastai reference predictions:")
        for name, spectro in spectrograms.items():
            # Save spectrogram
            references["spectrograms"][name] = spectro.clone()

            # Get fastai prediction
            # FastAI learner.model expects (batch, channels, height, width)
            # and returns logits
            learner.model.eval()
            with torch.no_grad():
                logits = learner.model(spectro.unsqueeze(0))  # Add batch dim
                probs = F.softmax(logits, dim=1)
                # FastAI classes are ['negative', 'positive']
                # Class index 1 is "positive" (call detected)
                call_prob = probs[0, 1].item()

            references["predictions"][name] = call_prob
            print(f"  {name}: {call_prob:.6f}")

        # Also store the class order for verification
        if hasattr(learner.data, 'classes'):
            references["classes"] = learner.data.classes
            print(f"\nFastAI classes: {learner.data.classes}")

        reference_file = reference_dir / "model_parity_reference.pt"
        torch.save(references, reference_file)
        print(f"\nSaved reference to: {reference_file}")

    def test_predict_call_matches_fastai(self, model_dir, reference_dir, numerical_tolerance, v1_config):
        """
        Compare predict_call() against saved fastai reference.

        Uses saved spectrograms to ensure we test model parity, not audio preprocessing.
        """
        reference_file = reference_dir / "model_parity_reference.pt"

        if not reference_file.exists():
            pytest.skip(f"Reference file not found: {reference_file}. Run test_generate_reference first.")

        model_path = model_dir / "model_v1.pt"
        if not model_path.exists():
            pytest.skip(f"Checkpoint not found: {model_path}. Run extraction script first.")

        from model_v1.inference import OrcaHelloSRKWDetectorV1

        # Load references
        references = torch.load(reference_file, weights_only=False)

        # Load model_v1
        model = OrcaHelloSRKWDetectorV1.from_checkpoint(str(model_path), v1_config)

        mismatches = []

        print("\nComparing predict_call() to fastai reference:")
        for name in references["spectrograms"]:
            # Get spectrogram
            spectro = references["spectrograms"][name]

            # Add batch dimension if needed: (1, 256, 313) -> (1, 1, 256, 313)
            if spectro.dim() == 3:
                spectro = spectro.unsqueeze(0)

            # Get model_v1 prediction
            model_v1_prob = model.predict_call(spectro).item()

            # Get fastai reference
            fastai_prob = references["predictions"][name]

            diff = abs(model_v1_prob - fastai_prob)
            status = "PASS" if diff <= numerical_tolerance["atol"] else "FAIL"

            print(f"  {name}: model_v1={model_v1_prob:.6f}, fastai={fastai_prob:.6f}, diff={diff:.2e} [{status}]")

            if diff > numerical_tolerance["atol"]:
                mismatches.append(
                    f"{name}: model_v1={model_v1_prob:.6f}, fastai={fastai_prob:.6f}, diff={diff:.2e}"
                )

        if "classes" in references:
            print(f"\nNote: FastAI classes were {references['classes']}")

        assert len(mismatches) == 0, f"Parity failures:\n" + "\n".join(mismatches)
