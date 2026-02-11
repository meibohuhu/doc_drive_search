#!/usr/bin/env python3
"""
Test script to correct measurements for a specific route.
python3 scripts/test_correct_measurements.py 2>&1 | head -150

"""

import json
import gzip
from pathlib import Path
from typing import Dict, Any, Tuple

# Import the correction functions from the main script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from correct_measurements import apply_opposite_vehicle_rules, should_process_route

# Test route path
TEST_ROUTE = Path("/shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/simlingo_extracted_test/data/simlingo/lb1_split/routes_training/OppositeVehicleRunningRedLight/Town10_Rep0_Town10HD_Scenario8_45_route0_01_11_16_07_53")
# Town10_Rep0_Town10HD_Scenario8_45_route0_01_11_16_07_53
# Town03_Rep0_Town03_Scenario8_27_route0_01_11_17_33_54

def process_test_route(route_path: Path):
    """Process a single test route."""
    print(f"Testing route: {route_path.name}")
    print(f"Full path: {route_path}\n")
    
    # Check if route should be processed (for testing, we'll process anyway)
    should_process = should_process_route(route_path)
    if not should_process:
        print("⚠️  This route does not normally meet the processing criteria")
        print("   (It will be processed anyway for testing purposes)\n")
    else:
        print("✅ This route meets the processing criteria\n")
    
    measurements_dir = route_path / "measurements"
    if not measurements_dir.exists():
        print(f"❌ Error: measurements directory not found: {measurements_dir}")
        return
    
    # Get all measurement files sorted by name
    measurement_files = sorted(measurements_dir.glob("*.json.gz"))
    
    if not measurement_files:
        print(f"❌ Error: No measurement files found in {measurements_dir}")
        return
    
    print(f"Found {len(measurement_files)} measurement files\n")
    
    # Create measurements_new directory
    measurements_new_dir = route_path / "measurements_new"
    measurements_new_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {measurements_new_dir}\n")
    
    # Process a few sample files first
    sample_files = measurement_files[:5]  # Process first 5 files as samples
    print("=" * 80)
    print("Processing sample files (first 5):")
    print("=" * 80)
    
    for i, src_file in enumerate(sample_files):
        # Get next file for next_command updates
        next_file = measurement_files[i + 1] if i + 1 < len(measurement_files) else None
        
        # Read the source file
        with gzip.open(src_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        # Read next file if available
        next_data = None
        if next_file and next_file.exists():
            with gzip.open(next_file, 'rt', encoding='utf-8') as f:
                next_data = json.load(f)
        
        # Show original values
        print(f"\n📄 File: {src_file.name}")
        print(f"   Original: command={data.get('command', 'N/A')}, next_command={data.get('next_command', 'N/A')}")
        print(f"   target_point: {data.get('target_point', 'N/A')}")
        print(f"   target_point_next: {data.get('target_point_next', 'N/A')}")
        print(f"   junction: {data.get('junction', 'N/A')}")
        
        # Apply corrections
        command, next_command = apply_opposite_vehicle_rules(data, next_data)
        data['command'] = command
        data['next_command'] = next_command
        
        print(f"   Corrected: command={command}, next_command={next_command}")
        
        # Write corrected file
        output_filename = src_file.stem  # removes .gz, keeps .json
        dest_file = measurements_new_dir / output_filename
        with open(dest_file, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        print(f"   ✅ Saved to: {dest_file.name}")
    
    # Process all remaining files
    print("\n" + "=" * 80)
    print(f"Processing remaining {len(measurement_files) - len(sample_files)} files...")
    print("=" * 80)
    
    for i, src_file in enumerate(measurement_files[len(sample_files):], start=len(sample_files)):
        # Get next file for next_command updates
        next_file = measurement_files[i + 1] if i + 1 < len(measurement_files) else None
        
        # Read the source file
        with gzip.open(src_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        # Read next file if available
        next_data = None
        if next_file and next_file.exists():
            with gzip.open(next_file, 'rt', encoding='utf-8') as f:
                next_data = json.load(f)
        
        # Apply corrections
        command, next_command = apply_opposite_vehicle_rules(data, next_data)
        data['command'] = command
        data['next_command'] = next_command
        
        # Write corrected file
        output_filename = src_file.stem
        dest_file = measurements_new_dir / output_filename
        with open(dest_file, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(measurement_files)} files...")
    
    print(f"\n✅ Complete! Processed {len(measurement_files)} files")
    print(f"   Output directory: {measurements_new_dir}")


if __name__ == "__main__":
    if not TEST_ROUTE.exists():
        print(f"❌ Error: Test route does not exist: {TEST_ROUTE}")
        sys.exit(1)
    
    process_test_route(TEST_ROUTE)

