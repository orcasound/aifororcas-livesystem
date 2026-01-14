#!/usr/bin/env python3
"""Upload OrcaHelloSRKWDetectorV1 to HuggingFace Hub."""

import argparse
from pathlib import Path
import sys
import yaml
import os

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from model_v1.inference import OrcaHelloSRKWDetectorV1

DEFAULT_REPO_ID = "orcasound/orcahello-srkw-detector-v1"

def main():
    # Check for HF_TOKEN environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️  Warning: HF_TOKEN environment variable not set.")
        print("   You need a HuggingFace token with WRITE permissions to upload.")
        print("   Get one at: https://huggingface.co/settings/tokens")
        print("   Then set it: export HF_TOKEN=hf_xxx...")
        print()
    
    parser = argparse.ArgumentParser(description="Upload OrcaHelloSRKWDetectorV1 to HuggingFace Hub")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"HuggingFace repo ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--checkpoint", type=Path, default=ROOT_DIR / "model/model_v1.pt", help="Path to model checkpoint")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "tests/test_config.yaml", help="Path to config YAML")
    parser.add_argument("--commit-message", default="Upload OrcaHelloSRKWDetectorV1", help="Commit message for upload")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    # Validate paths
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: Config not found at {args.config}")
        sys.exit(1)

    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = OrcaHelloSRKWDetectorV1.from_checkpoint(str(args.checkpoint), config)

    # Upload to hub
    print(f"Uploading to {args.repo_id}...")
    print(f"Commit message: {args.commit_message}")
    model.push_to_hub(
        args.repo_id,
        commit_message=args.commit_message,
        private=args.private,
        token=hf_token
    )

    print(f"\n✓ Successfully uploaded to: https://huggingface.co/{args.repo_id}")
    print(f"\nTo load the model:")
    print(f"  from model_v1.inference import OrcaHelloSRKWDetectorV1")
    print(f"  model = OrcaHelloSRKWDetectorV1.from_pretrained('{args.repo_id}')")

if __name__ == "__main__":
    main()
