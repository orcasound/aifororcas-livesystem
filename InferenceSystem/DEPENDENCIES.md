# InferenceSystem Dependency Management

## Version Constraints Strategy

The InferenceSystem uses Python 3.8 and runs on Ubuntu 18.04 in production (Docker). Because of this, we use **version ranges** rather than pinned versions for key dependencies to allow Dependabot to update within safe bounds while preventing updates to versions that don't exist for Python 3.8.

### Key Dependencies and Their Constraints

| Package | Constraint | Reasoning |
|---------|-----------|-----------|
| `numba` | `>=0.51.0,<0.59.0` | Versions 0.59.0+ require Python 3.9+. Max version 0.58.1 is confirmed working. Minimum 0.51.0 required by librosa 0.10.x. |
| `numpy` | `>=1.19.5,<1.25.0` | Versions 1.25.0+ require Python 3.9+. Max version 1.24.4 is confirmed working. |
| `spacy` | `>=3.5.4,<3.8.3` | Version 3.8.7 doesn't have wheels for Python 3.8. Max version 3.8.2 is confirmed available. |
| `librosa` | `>=0.8.0,<0.11.0` | Version 0.11.0+ requires numba 0.51.0+, which may have compatibility issues. Version 0.10.0 is confirmed working. |
| `pandas` | `>=1.1.0,<2.0` | Constrained to maintain compatibility with numpy 1.x and Python 3.8. |

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
- `numpy==1.26.4` - doesn't support Python 3.8 (max is 1.24.4)

By using version ranges with upper bounds, we prevent these invalid updates while still allowing Dependabot to suggest updates within the safe range.

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

See `ModelTraining/requirements.lock.txt` for a full list of verified working versions that pass tests.

Key versions confirmed working:
- `numba==0.58.1`
- `numpy==1.24.4`
- `spacy==3.7.5`
- `librosa==0.10.0`

## Future Migration Path

When the InferenceSystem is upgraded to Python 3.9 or later:
1. Update the Docker base image from Ubuntu 18.04
2. Update CI workflows to use Python 3.9+
3. Relax version constraints to allow newer versions
4. Test thoroughly before deploying to production
