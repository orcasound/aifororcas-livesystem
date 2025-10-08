# InferenceSystem Dependency Management

## Version Constraints Strategy

The InferenceSystem uses Python 3.8 and runs on Ubuntu 18.04 in production (Docker). Because of this, we use **version ranges** rather than pinned versions for key dependencies to allow Dependabot to update within safe bounds while preventing updates to versions that don't exist for Python 3.8.

### Key Dependencies and Their Constraints

| Package | Constraint | Reasoning |
|---------|-----------|-----------|
| `numba` | `>=0.51.0,<0.59.0` | Versions 0.59.0+ require Python 3.9+. Max version 0.58.1 is confirmed working. Minimum 0.51.0 required by librosa 0.10.x. |
| `numpy` | `==1.19.5` | Pinned to 1.19.5 for Python 3.6 compatibility (Docker uses Ubuntu 18.04). numpy 1.20+ requires Python 3.7+. This is the only stable version that works across Python 3.6-3.8. |
| `spacy` | `>=3.5.4,<3.8.3` | Version 3.8.7 doesn't have wheels for Python 3.8. Max version 3.8.2 is confirmed available. |
| `librosa` | `>=0.8.0,<0.11.0` | Version 0.11.0+ requires numba 0.51.0+, which may have compatibility issues. Version 0.10.0 is confirmed working. |
| `pandas` | `>=1.1.0,<2.0` | Constrained to maintain compatibility with numpy 1.x and Python 3.6+. |

## Dependabot Configuration

The `.github/dependabot.yml` file includes `versioning-strategy: increase` for the pip ecosystem. This tells Dependabot to:
- Increase the lower bound of version ranges (e.g., `>=0.48` → `>=0.58.1`)
- Rather than replace ranges with pinned versions
- This prevents Dependabot from suggesting versions that don't exist for our Python version

## Why These Constraints Matter

Dependabot doesn't validate whether proposed versions:
- Are available on PyPI for your platform (many packages skip Python 3.8 builds for newer versions)
- Are compatible with your other dependencies
- Are installable in your CI environment (Python version mismatch)

For example, Dependabot previously suggested:
- `numba==0.60.0` - doesn't exist for Python 3.8 (only goes up to 0.58.1)
- `spacy==3.8.7` - doesn't have wheels for Python 3.8 (latest is 3.8.2)
- `numpy==1.26.4` - doesn't support Python 3.6 (Docker constraint) or Python 3.8

**Note on numpy**: Due to Docker using Python 3.6, numpy must be pinned to 1.19.5 (the last version supporting Python 3.6 before numpy 1.20+ required Python 3.7+). While CI uses Python 3.8, we must satisfy the lowest common denominator.

By using version ranges with upper bounds (and pinned versions where necessary), we prevent these invalid updates while still allowing Dependabot to suggest updates within the safe range.

## Testing Dependency Updates

When testing dependency updates locally or in CI:
1. Ensure you're using Python 3.8 (not 3.9+)
2. Test on both Ubuntu and Windows (as both are used in CI)
3. Verify the package version exists on PyPI for Python 3.8:
   ```bash
   pip index versions <package> --python-version 3.8
   ```
4. Check compatibility with other dependencies (especially librosa/numba/numpy)

## Verified Working Versions

See `ModelTraining/requirements.lock.txt` for a full list of verified working versions that pass tests with Python 3.8+.

Key versions confirmed working:
- `numba==0.58.1` (for Python 3.8+)
- `numpy==1.19.5` (for Python 3.6-3.8 compatibility)
- `spacy==3.7.5` (for Python 3.8+)
- `librosa==0.10.0` (for Python 3.8+)

**Note**: The InferenceSystem Docker build uses Ubuntu 18.04 with Python 3.6, which constrains numpy to 1.19.x. The CI tests use Python 3.8 and can support newer versions, but we constrain to the lowest common denominator for compatibility.

## Future Migration Path

When the InferenceSystem Docker is upgraded to Python 3.7 or later:
1. Update the Docker base image from Ubuntu 18.04 to Ubuntu 20.04+ (which has Python 3.8+)
2. Relax numpy constraint to `>=1.19.5,<1.25.0` to allow versions up to 1.24.4
3. Consider updating other constraints as well
4. Test thoroughly before deploying to production

When upgrading to Python 3.9+:
1. Complete the Python 3.7+ migration first
2. Update CI workflows to use Python 3.9+
3. Further relax version constraints to allow newer package versions
4. Test thoroughly before deploying to production
