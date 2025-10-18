# InferenceSystem Dependency Management

## Version Constraints Strategy

The InferenceSystem uses Python 3.11+ and is no longer constrained by Python 3.6 compatibility. We use **version ranges** for key dependencies to allow updates within safe bounds while ensuring compatibility with Python 3.11.3 and later.

### Key Dependencies and Their Constraints

| Package | Constraint | Reasoning |
|---------|-----------|-----------|
| `numba` | `>=0.57.0` | Versions 0.57.0+ are compatible with Python 3.11. No upper bound to allow updates. |
| `numpy` | `>=1.21.0` | numpy 1.21+ is compatible with Python 3.9+ and works well with Python 3.11. |
| `spacy` | `>=3.5.4` | spacy 3.5.4+ supports Python 3.11. No upper bound to allow updates. |
| `librosa` | `>=0.10.0` | Version 0.10.0+ is compatible with Python 3.11 and modern numpy/numba. |
| `pandas` | `>=1.1.0,<2.0` | Constrained to maintain compatibility with existing code. |
| `torchaudio` | `>=2.1.0` | Version 2.1.0+ is compatible with Python 3.11. |

## Dependabot Configuration

The `.github/dependabot.yml` file includes `versioning-strategy: increase` for the pip ecosystem. This tells Dependabot to:
- Increase the lower bound of version ranges (e.g., `>=0.48` → `>=0.58.1`)
- Rather than replace ranges with pinned versions
- This prevents Dependabot from suggesting versions that don't exist for our Python version

## PyTorch Index Configuration

The `requirements.txt` uses `--extra-index-url https://download.pytorch.org/whl/torch_stable.html` to access PyTorch wheels. This is specified as an extra index (not `--find-links`) to ensure pip can still access PyPI for all other packages while using the PyTorch-specific index for torch, torchvision, and torchaudio.

**Note**: Using `--extra-index-url` instead of `-f` (or `--find-links`) is critical because:
- `--find-links` can cause pip to look ONLY at the specified URL first, potentially breaking resolution for packages not found there
- `--extra-index-url` adds the URL as an additional package index alongside PyPI, ensuring all packages can be found

## Why These Constraints Matter

Dependabot doesn't validate whether proposed versions:
- Are available on PyPI for your platform
- Are compatible with your other dependencies
- Are installable in your CI environment (Python version mismatch)

By using version ranges with minimum requirements (and upper bounds where necessary), we ensure compatibility while allowing safe updates.

## Testing Dependency Updates

When testing dependency updates locally or in CI:
1. Ensure you're using Python 3.11 or later
2. Test on both Ubuntu and Windows (as both are used in CI)
3. Verify the package version exists on PyPI for Python 3.11+:
   ```bash
   pip index versions <package> --python-version 3.11
   ```
4. Check compatibility with other dependencies (especially librosa/numba/numpy)

## Verified Working Versions

Key versions confirmed working with Python 3.11+:
- `numba>=0.57.0` (for Python 3.11+)
- `numpy>=1.21.0` (for Python 3.9+ compatibility)
- `spacy>=3.5.4` (for Python 3.11+ compatibility)
- `librosa>=0.10.0` (for Python 3.11+)
- `torchaudio>=2.1.0` (for Python 3.11+)

## Migration from Python 3.8

This version of the InferenceSystem has been upgraded from Python 3.8 to Python 3.11.3. Key changes:
1. Removed Python 3.6 compatibility constraints
2. Updated numpy from pinned 1.19.5 to >=1.21.0
3. Removed spacy upper bound constraint (<3.7.0)
4. Updated librosa from <0.11.0 to >=0.10.0
5. Updated torchaudio from <0.14.0 to >=2.1.0
6. Updated numba from <0.59.0 to >=0.57.0
