#!/usr/bin/env python3
"""
Script to process Orcasound listener reports JSON files.

This script:
1. Validates JSON data into Pydantic objects
2. Flattens OrcasoundDetection entries with detection prefix
3. Creates DataFrame with important fields and saves to CSV
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytz
from detection_types import OrcasoundListenerReport


def load_and_validate_json(file_path: Path) -> List[OrcasoundListenerReport]:
    """Load JSON file and validate into Pydantic objects."""
    print(f"Loading and validating: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract the candidates results
    candidates_results = data.get("data", {}).get("candidates", {}).get("results", [])

    # Validate each result into OrcasoundListenerReport
    validated_reports = []
    for i, result in enumerate(candidates_results):
        try:
            report = OrcasoundListenerReport(**result)
            validated_reports.append(report)
        except Exception as e:
            print(f"Warning: Failed to validate result {i}: {e}")
            continue

    print(f"Successfully validated {len(validated_reports)} reports")
    return validated_reports


def flatten_detections(reports: List[OrcasoundListenerReport]) -> List[Dict[str, Any]]:
    """Flatten OrcasoundDetection entries into individual records."""
    flattened = []

    for report in reports:
        # Get all fields from report except category and detections
        report_fields = report.model_dump()
        report_fields.pop("category", None)  # Remove category to avoid clash
        report_fields.pop("detections", None)  # Remove detections as we'll flatten them

        for detection in report.detections:
            # Create flattened entry
            entry = {}

            # Add detection fields with detection prefix
            detection_dict = detection.model_dump()
            for key, value in detection_dict.items():
                entry[f"detection.{key}"] = value

            # Add report fields (except category and detections)
            for key, value in report_fields.items():
                if key == "feed":
                    # Flatten feed fields
                    for feed_key, feed_value in value.items():
                        entry[f"feed.{feed_key}"] = feed_value
                else:
                    entry[f"report.{key}"] = value

            flattened.append(entry)

    print(f"Flattened into {len(flattened)} detection entries")
    return flattened


def parse_timestamp_to_pst(timestamp_str: str) -> Dict[str, str]:
    """Parse UTC timestamp and extract PST date/time components."""
    pst_fields = {}

    if timestamp_str:
        try:
            # Handle timestamp format (remove fractional seconds if present)
            # e.g. '2025-08-14T01:55:02.48881Z' -> '2025-08-14T01:55:02+00:00'
            clean_timestamp = (
                timestamp_str.rsplit(".", 1)[0] + "Z"
                if "." in timestamp_str
                else timestamp_str
            )
            utc_dt = datetime.fromisoformat(clean_timestamp.replace("Z", "+00:00"))

            # Convert to Pacific time (handles PST/PDT automatically)
            pacific_tz = pytz.timezone("America/Los_Angeles")
            pst_dt = utc_dt.astimezone(pacific_tz)

            # Extract PST components
            pst_fields["detection.date_pst"] = pst_dt.strftime("%Y-%m-%d")
            pst_fields["detection.time_pst"] = pst_dt.strftime("%H:%M")
            pst_fields["detection.year_month_pst"] = pst_dt.strftime("%Y-%m")
            pst_fields["detection.year_pst"] = pst_dt.strftime("%Y")
            pst_fields["detection.month_pst"] = pst_dt.strftime("%m")
            pst_fields["detection.hour_pst"] = pst_dt.strftime("%H")

        except Exception as e:
            print(f"Warning: Failed to parse timestamp '{timestamp_str}': {e}")
            # Set empty values for failed parsing
            for field in [
                "detection.date_pst",
                "detection.time_pst",
                "detection.year_month_pst",
                "detection.year_pst",
                "detection.month_pst",
                "detection.hour_pst",
            ]:
                pst_fields[field] = None

    return pst_fields


def create_dataframe(flattened_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create DataFrame with important fields."""

    # Define the important fields we want to extract
    important_fields = [
        "detection.timestamp",
        "detection.playlistTimestamp",
        "detection.description",
        "detection.source",
        "detection.id",
        "detection.category",
        "feed.name",
        "feed.id",
        "report.detectionCount",
        "report.id",
        # PST timestamp fields
        "detection.date_pst",
        "detection.time_pst",
        "detection.year_month_pst",
        "detection.year_pst",
        "detection.month_pst",
        "detection.hour_pst",
    ]

    # Extract the important fields
    df_data = []
    for entry in flattened_data:
        row = {}
        for field in important_fields:
            row[field] = entry.get(field)

        # Parse timestamp and add PST fields
        timestamp_str = entry.get("detection.timestamp", "")
        pst_fields = parse_timestamp_to_pst(timestamp_str)
        row.update(pst_fields)

        df_data.append(row)

    df = pd.DataFrame(df_data)
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    return df


def process_file(input_file: Path) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Process a single JSON file and return the results."""
    print(f"\n=== Processing {input_file} ===")

    # Create output directory at the same level as the parent directory
    parent_dir = input_file.parent
    output_dir = parent_dir.parent / f"{parent_dir.name}_parsed"
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load and validate
    try:
        reports = load_and_validate_json(input_file)
    except Exception as e:
        print(f"Error loading/validating {input_file}: {e}")
        return [], pd.DataFrame()

    if not reports:
        print(f"No valid reports found in {input_file}")
        return [], pd.DataFrame()

    # Step 2: Flatten detections
    flattened_data = flatten_detections(reports)

    # Save flattened JSON
    flattened_json_path = output_dir / f"{input_file.stem}_flattened.json"
    with open(flattened_json_path, "w") as f:
        json.dump(flattened_data, f, indent=2)
    print(f"Saved flattened JSON to: {flattened_json_path}")

    # Step 3: Create DataFrame and save CSV
    df = create_dataframe(flattened_data)
    csv_path = output_dir / f"{input_file.stem}_parsed.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")

    # Print summary stats
    print(f"\nSummary for {input_file.name}:")
    print(f"  Reports: {len(reports)}")
    print(f"  Total detections: {len(flattened_data)}")
    print(
        f"  Date range: {df['detection.timestamp'].min()} to {df['detection.timestamp'].max()}"
    )
    print(f"  Unique feeds: {df['feed.name'].nunique()}")

    return flattened_data, df


def main():
    parser = argparse.ArgumentParser(description="Process Orcasound listener reports")
    parser.add_argument("files", nargs="+", help="JSON files to process")

    args = parser.parse_args()

    files_to_process = [Path(f) for f in args.files]

    # Track all results for aggregation
    all_flattened_data = []
    all_dataframes = []
    processed_files = []

    # Process each file
    for file_path in files_to_process:
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        try:
            flattened_data, df = process_file(file_path)
            if flattened_data and len(df) > 0:
                all_flattened_data.extend(flattened_data)
                all_dataframes.append(df)
                processed_files.append(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback

            traceback.print_exc()

    # Aggregate results if we have processed files
    if len(processed_files) > 1:
        print(f"\n=== Aggregating results from {len(processed_files)} files ===")

        # Determine output directory at the same level as the parent directory
        parent_dir = processed_files[0].parent
        output_dir = parent_dir.parent / f"{parent_dir.name}_parsed"

        # Save combined flattened JSON
        combined_json_path = output_dir / "reports_combined.json"
        with open(combined_json_path, "w") as f:
            json.dump(all_flattened_data, f, indent=2)
        print(f"Saved combined flattened JSON to: {combined_json_path}")

        # Combine all DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Save combined CSV
        combined_csv_path = output_dir / "reports_combined.csv"
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Saved combined CSV to: {combined_csv_path}")

        # Print combined summary stats
        print("\nCombined Summary:")
        print(f"  Total files processed: {len(processed_files)}")
        print(f"  Total detections: {len(all_flattened_data)}")
        if len(combined_df) > 0:
            print(
                f"  Date range: {combined_df['detection.timestamp'].min()} to {combined_df['detection.timestamp'].max()}"
            )
            print(f"  Unique feeds: {combined_df['feed.name'].nunique()}")
            print(f"  Files processed: {[f.name for f in processed_files]}")

    elif len(processed_files) == 1:
        print(f"\nProcessed single file: {processed_files[0].name}")
    else:
        print("\nNo files were successfully processed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
