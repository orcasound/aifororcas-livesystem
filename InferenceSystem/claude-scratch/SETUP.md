# Reproducible InferenceSystem Setup

This directory contains modified configuration files that work around compatibility issues on Python 3.11 and macOS ARM64.

## Quick Start with uv (Recommended)

```bash
# 1. Extract model (if not done)
cd InferenceSystem
unzip -o model/model.zip -d model/
mv model/model/model.pkl model/
rmdir model/model

# 2. Create environment
uv venv inference-venv --python 3.11
source inference-venv/bin/activate

# 3. Install fastai without dependencies (avoids pynvx ARM64 issue)
uv pip install --no-deps fastai==1.0.61

# 4. Install remaining dependencies
uv pip install -r claude-scratch/requirements.txt

# 5. Apply fastai_audio patch for Python 3.11
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

# 6. Test
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
- **Added**: `soundfile` for audio I/O
- **Clarified**: Install fastai 1.0.61 without deps, then install other packages

### Dockerfile
- **Updated**: Python 3.11 compatibility patch applied inline
- **Fixed**: Model extraction steps
- **Added**: Inline Python script for patching instead of shell script (portable)

## Issues Addressed

1. **pynvx ARM64**: fastai 1.0.61 depends on pynvx which has no ARM64 wheels. We install fastai with `--no-deps` to skip it.

2. **fastai version conflict**: The audio package pulls in fastai 2.x. We explicitly install 1.0.61 first without deps.

3. **Python 3.11 dataclass**: fastai_audio uses mutable default argument in dataclass, which is an error in Python 3.11+. We patch it to use `field(default_factory=...)`.

## Validation

Both setups were tested on macOS ARM64 with Python 3.11.5 and produced:

- **CI Test**: `global_prediction: 1`, confidence 66.28%
- **Local WAV Test**: `global_prediction: 1`, confidence 63.91%
