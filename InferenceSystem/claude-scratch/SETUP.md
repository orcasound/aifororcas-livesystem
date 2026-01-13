# Reproducible InferenceSystem Setup

This directory contains modified configuration files that work around compatibility issues on Python 3.11 and macOS ARM64.

## Quick Start with uv (Recommended)

```bash
# Navigate to InferenceSystem directory
cd InferenceSystem

# 1. Extract model (if not done)
unzip -o model/model.zip -d model/
mv model/model/model.pkl model/
rmdir model/model

# 2. Remove old environment (if exists)
rm -rf inference-venv

# 3. Create fresh environment with Python 3.11
uv venv inference-venv --python 3.11
source inference-venv/bin/activate

# 4. Install fastai 1.0.61 without dependencies (avoids pynvx ARM64 issue)
uv pip install --no-deps fastai==1.0.61

# 5. Install remaining dependencies
# Note: This will install fastai-audio which pulls in fastai 2.x,
# but we'll reinstall fastai 1.0.61 in the next step
uv pip install -r claude-scratch/requirements.txt

# 6. Reinstall fastai 1.0.61 (needed after fastai-audio overwrites it)
uv pip install --no-deps fastai==1.0.61

# 7. Apply fastai_audio patch for Python 3.11
python3 << 'EOF'
import sys
import os

site_packages = [p for p in sys.path if 'site-packages' in p][0]
audio_data_file = os.path.join(site_packages, 'audio', 'data.py')

with open(audio_data_file, 'r') as f:
    content = f.read()

content = content.replace(
    'from dataclasses import dataclass, asdict',
    'from dataclasses import dataclass, asdict, field'
)
content = content.replace(
    'sg_cfg: SpectrogramConfig = SpectrogramConfig()',
    'sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)'
)

with open(audio_data_file, 'w') as f:
    f.write(content)

print(f"Patched: {audio_data_file}")
EOF

# 8. Test the environment
python claude-scratch/test_local_wav.py
```

## Docker Build

```bash
cd InferenceSystem

# Build using the fixed Dockerfile
docker build -f claude-scratch/Dockerfile -t live-inference-system:fixed .

# Run with test config
docker run --rm live-inference-system:fixed

# Run with local WAV file
docker run --rm \
  -v /path/to/audio.wav:/audio.wav \
  live-inference-system:fixed \
  python -u ./src/model/fastai_inference.py /audio.wav
```

## Key Differences from Original

### requirements.txt
- **Removed**: Explicit `pynvx` dependency (causes ARM64 issues)
- **Removed**: `fastai==1.0.61` from requirements.txt (must be installed separately with `--no-deps`)
- **Added**: FastAI dependencies manually (beautifulsoup4, bottleneck, fastprogress, nvidia-ml-py3, packaging, Pillow, pyyaml, scipy)
- **Added**: `soundfile` for audio I/O
- **Workflow**: Install fastai 1.0.61 with `--no-deps` → Install requirements.txt → Reinstall fastai 1.0.61 (because fastai-audio overwrites it)

### Dockerfile
- **Updated**: Python 3.11 compatibility patch applied inline
- **Fixed**: Model extraction steps
- **Added**: Inline Python script for patching instead of shell script (portable)

## Issues Addressed

1. **pynvx ARM64**: fastai 1.0.61 depends on pynvx which has no ARM64 wheels. We install fastai with `--no-deps` to skip it.

2. **fastai version conflict**: The audio package pulls in fastai 2.x. We explicitly install 1.0.61 first without deps.

3. **Python 3.11 dataclass**: fastai_audio uses mutable default argument in dataclass, which is an error in Python 3.11+. We patch it to use `field(default_factory=...)`.

## Validation

Environment setup verified on macOS ARM64 with Python 3.11.5:

- **Environment**: Successfully created with uv
- **FastAI version**: 1.0.61 (verified after installation)
- **Python 3.11 patch**: Applied successfully to fastai_audio
- **Test result**: `test_local_wav.py` runs successfully
  - Processed 59 segments from test WAV file
  - Detected 2 positive segments (below threshold of 3)
  - `global_prediction: 0`, `global_confidence: 58.750`
  - All dependencies working correctly

The environment is ready for inference tasks.
