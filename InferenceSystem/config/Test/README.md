# Test Configuration Files

This directory contains test configuration files for the InferenceSystem used in CI/CD workflows.

## Configuration Types

### LiveHLS Configs
These configs test live streaming from hydrophones:
- `FastAI_LiveHLS_OrcasoundLab.yml` - Tests live streaming from Orcasound Lab hydrophone
- `AudioSet_LiveHLS_OrcasoundLab.yml` - Tests live streaming using AudioSet model

### DateRangeHLS Configs
These configs test inference on historical audio data from specific date/time ranges:

#### Positive Detection Test
- `FastAI_DateRangeHLS_OrcasoundLab.yml` - **Expected: global_prediction: 1**
  - Hydrophone: rpi_orcasound_lab
  - Time: 2020-09-01 15:13 to 16:45 PST
  - This is a known positive detection case used to verify inference is working correctly

#### Hydrophone Coverage Tests
The following configs test inference across all hydrophones. These may produce either positive (1) or negative (0) predictions depending on whether orca calls were present at the specified time:

- `FastAI_DateRangeHLS_AndrewsBay.yml` - rpi_andrews_bay at 2025-12-01 15:13-16:45 PST
- `FastAI_DateRangeHLS_BushPoint.yml` - rpi_bush_point at 2020-09-01 15:13-16:45 PST
- `FastAI_DateRangeHLS_MastCenter.yml` - rpi_mast_center at 2020-09-01 15:13-16:45 PST
- `FastAI_DateRangeHLS_NorthSJC.yml` - rpi_north_sjc at 2020-09-01 15:13-16:45 PST
- `FastAI_DateRangeHLS_PointRobinson.yml` - rpi_point_robinson at 2020-09-01 15:13-16:45 PST
- `FastAI_DateRangeHLS_PortTownsend.yml` - rpi_port_townsend at 2020-09-01 15:13-16:45 PST
- `FastAI_DateRangeHLS_SunsetBay.yml` - rpi_sunset_bay at 2020-09-01 15:13-16:45 PST

#### Edge Case Tests
These configs test system behavior in edge cases:

- `FastAI_DateRangeHLS_NoAudio.yml` - Tests when no audio files exist for the specified time
  - Hydrophone: rpi_andrews_bay at 2020-09-01 15:13-16:45 PST
  - Expected: System should handle gracefully without crashing with an informative warning message

- `FastAI_DateRangeHLS_IncompleteMinute.yml` - Tests when audio stream doesn't last a full minute
  - Hydrophone: rpi_orcasound_lab at 2018-11-01 21:28 PST
  - Expected: System should handle gracefully without crashing

## CI/CD Testing

The GitHub Actions workflow (`.github/workflows/InferenceSystem.yaml`) runs these tests on:
- Windows (test-windows job)
- Ubuntu (test-ubuntu job)
- Docker (test-docker job)

Only the `FastAI_DateRangeHLS_OrcasoundLab.yml` test explicitly verifies `global_prediction: 1` output to confirm inference is working correctly. Other tests verify the system runs without errors but may produce either positive or negative predictions.
