# Test Configuration Files

This directory contains test configuration files for the InferenceSystem used in CI/CD workflows.

## Directory Structure

Test configurations are organized into subdirectories based on expected outcomes:
- `Positive/` - Tests expected to produce `global_prediction: 1` (positive detections)
- `Negative/` - Tests expected to produce `global_prediction: 0` (negative detections)
- `Fail/` - Edge case tests for error handling (no audio, incomplete streams)

## Configuration Types

### LiveHLS Configs
These configs test live streaming from hydrophones:
- `FastAI_LiveHLS_OrcasoundLab.yml` - Tests live streaming from Orcasound Lab hydrophone
- `AudioSet_LiveHLS_OrcasoundLab.yml` - Tests live streaming using AudioSet model

### DateRangeHLS Configs

#### Positive Detection Tests (Positive/)
These are known positive detection cases used to verify inference is working correctly.
Tests expected to produce `global_prediction: 1`:

- `FastAI_DateRangeHLS_OrcasoundLab.yml`
  - Hydrophone: rpi_orcasound_lab
  - Time: 2020-09-01 15:13 to 16:45 PST

- `FastAI_DateRangeHLS_AndrewsBay.yml`
  - Hydrophone: rpi_andrews_bay
  - Time: 2025-12-16 23:16 to 23:17 PST

- `FastAI_DateRangeHLS_BushPoint.yml`
  - Hydrophone: rpi_bush_point
  - Time: 2024-11-02 09:52 to 09:53 PST

- `FastAI_DateRangeHLS_NorthSJC.yml`
  - Hydrophone: rpi_north_sjc
  - Time: 2025-12-29 21:58 to 21:59 PST

- `FastAI_DateRangeHLS_PortTownsend.yml`
  - Hydrophone: rpi_port_townsend
  - Time: 2025-12-26 14:08 to 14:09 PST

- `FastAI_DateRangeHLS_SunsetBay.yml`
  - Hydrophone: rpi_sunset_bay
  - Time: 2025-12-11 15:03 to 15:04 PST

#### Negative Detection Tests (Negative/)
Tests expected to produce `global_prediction: 0`:

- `FastAI_DateRangeHLS_MastCenter.yml`
  - Hydrophone: rpi_mast_center
  - Time: 2023-08-05 12:32 to 14:00 PST

- `FastAI_DateRangeHLS_PointRobinson.yml`
  - Hydrophone: rpi_point_robinson
  - Time: 2025-11-01 15:13 to 16:45 PST

#### Edge Case Tests (Fail/)
These configs test system behavior in edge cases:

- `FastAI_DateRangeHLS_NoAudio.yml` - Tests when no audio files exist for the specified time
  - Hydrophone: rpi_andrews_bay at 2020-09-01 15:13 to 16:45 PST
  - Expected: System should handle gracefully without crashing with an informative warning message

- `FastAI_DateRangeHLS_NoAudio2.yml` - Tests when no audio files exist for the specified time (second case)
  - Hydrophone: rpi_mast_center at 2020-09-01 15:13 to 16:45 PST
  - Expected: System should handle gracefully without crashing with an informative warning message

- `FastAI_DateRangeHLS_IncompleteMinute.yml` - Tests when audio stream doesn't last a full minute
  - Hydrophone: rpi_orcasound_lab at 2018-11-01 21:28 to 21:29 PST
  - Expected: System should handle gracefully without crashing

## CI/CD Testing

The GitHub Actions workflow (`.github/workflows/InferenceSystem.yaml`) runs these tests on:
- Windows (test-windows job)
- Ubuntu (test-ubuntu job)
- Docker (test-docker job)

Tests from the `Positive/` directory verify `global_prediction: 1` output to confirm inference is working correctly for positive detections. Tests from the `Negative/` directory verify `global_prediction: 0` to confirm negative detection handling. Tests from the `Fail/` directory verify the system handles edge cases gracefully without crashing.
