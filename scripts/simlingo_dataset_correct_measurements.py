#!/usr/bin/env python3
"""
Script to correct measurements in route files based on specific rules.

##### FOR SIMLINGO mhu 20260210 ###############
"""

import json
import gzip
from pathlib import Path
from typing import Dict, Any, Tuple

# Source directory
SOURCE_DIR = "/shared/rc/llm-gen-agent/mhu/simlingo_dataset/database/simlingo_extracted_test/data/simlingo/lb1_split/routes_training"

# Folders that should be copied without changes
NO_CHANGE_FOLDERS = ["ControlLoss", "DynamicObjectCrossing"]

# Folder that needs rule-based corrections
CORRECT_FOLDER = "OppositeVehicleRunningRedLight"


def apply_opposite_vehicle_rules(data: Dict[str, Any], next_data: Dict[str, Any] = None) -> Tuple[int, int]:
    """
    Apply rules for OppositeVehicleRunningRedLight scenario.
    
    Returns: (command, next_command)
    
    Rules:
    1. target_point |y|<4, next_target_point |y|>=4, and junction=false => command=4, next_command=1(y<0) or 2(y>0)
    2. target_point |y|>4, next_target_point |y|>4, and |target_point y - next_target_point y|>3 => 
       command=1(if y<0) / command=2(if y>0)
    3. target_point |y|>4, next_target_point |y|>4, but |target_point y - next_target_point y|<3, and junction=true => 
       command=4, next_command=4
    4. target_point |y|<4, next_target_point |y|<4 => command=4
    5. target_point |y|>4, next_target_point |y|>4, but |target_point y - next_target_point y|<1, junction=false => 
       change lane to left(if y>0) / change lane to right(if y<0)
    """
    target_point = data['target_point']
    target_point_next = data['target_point_next']
    junction = data.get('junction', False)
    
    tp_y = target_point[1]
    tp_next_y = target_point_next[1]
    
    tp_y_abs = abs(tp_y)
    tp_next_y_abs = abs(tp_next_y)
    y_diff = abs(tp_y - tp_next_y)
    
    # Rule 1: target_point |y|<4, next_target_point |y|>=4, and junction=false => command=4, next_command=1(y<0) or 2(y>0)
    if tp_y_abs < 4 and tp_next_y_abs >= 4 and not junction:
        next_cmd = 1 if tp_next_y < 0 else 2
        return (4, next_cmd)
    
    # Rule 4: target_point |y|<4, next_target_point |y|<4 => command=4
    if tp_y_abs < 4 and tp_next_y_abs < 4:
        return (4, data.get('next_command', 4))
    
    # Rules 2, 3, 5: target_point |y|>4, next_target_point |y|>4
    if tp_y_abs >= 4 and tp_next_y_abs >= 4:
        # Rule 5: difference < 1, junction=false => lane change (check first as more specific)
        if y_diff < 0.1 and not junction:
            # change lane to left(if y>0) / change lane to right(if y<0)
            cmd = 5 if tp_y > 0 else 6  # 5=left, 6=right
            return (cmd, data.get('next_command', 4))
        
        # Rule 3: difference < 3, junction=true => command=4, next_command=4
        if y_diff < 3 and junction:
            return (4, 4)
        
        # Rule 2: difference > 3 => command=1(if y<0) / command=2(if y>0)
        if y_diff >= 3:
            cmd = 1 if tp_y < 0 else 2
            return (cmd, data.get('next_command', 4))
    
    # Default: keep original values
    return (data.get('command', 4), data.get('next_command', 4))


def process_measurement_file(src_path: Path, dest_path: Path, scenario_folder: str, next_file_path: Path = None):
    """Process a single measurement file."""
    # Read the source file
    with gzip.open(src_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    # Read next file if available (for next_command updates)
    next_data = None
    if next_file_path and next_file_path.exists():
        with gzip.open(next_file_path, 'rt', encoding='utf-8') as f:
            next_data = json.load(f)
    
    # Apply corrections based on scenario
    if scenario_folder == CORRECT_FOLDER:
        command, next_command = apply_opposite_vehicle_rules(data, next_data)
        data['command'] = command
        data['next_command'] = next_command
    # For NO_CHANGE_FOLDERS, data remains unchanged
    
    # Ensure destination directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write corrected file (as .json, not .json.gz based on user's request)
    with open(dest_path, 'wt', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def process_route_folder(route_path: Path, scenario_folder: str):
    """Process all measurement files in a route folder and create measurements_new."""
    measurements_dir = route_path / "measurements"
    if not measurements_dir.exists():
        print(f"Warning: measurements directory not found: {measurements_dir}")
        return
    
    # Get all measurement files sorted by name
    measurement_files = sorted(measurements_dir.glob("*.json.gz"))
    
    if not measurement_files:
        print(f"Warning: No measurement files found in {measurements_dir}")
        return
    
    # Create measurements_new directory in the same route folder
    measurements_new_dir = route_path / "measurements_new"
    measurements_new_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(measurement_files)} files in {route_path.name}")
    
    # Process each file
    for i, src_file in enumerate(measurement_files):
        # Get next file for next_command updates
        next_file = measurement_files[i + 1] if i + 1 < len(measurement_files) else None
        
        # Create output file with .json extension (remove .gz)
        output_filename = src_file.stem  # removes .gz, keeps .json
        dest_file = measurements_new_dir / output_filename
        process_measurement_file(src_file, dest_file, scenario_folder, next_file)


def should_process_route(route_path: Path) -> bool:
    """
    Check if a route should be processed based on results.json.gz criteria.
    
    Returns True if the route matches the condition:
    - score_composed < 100.0 AND
    - (score_route <= 94.0 OR num_infractions != (len(min_speed_infractions) + len(outside_route_lanes)))
    
    This matches the condition in dataset_base.py lines 297-306.
    """
    results_file = route_path / "results.json.gz"
    
    if not results_file.exists():
        # If no results file, skip this route
        return False
    
    try:
        with gzip.open(results_file, 'rt', encoding='utf-8') as f:
            results_route = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read results.json.gz for {route_path}: {e}")
        return False
    
    # Check condition: score_composed < 100.0
    if results_route['scores']['score_composed'] >= 100.0:
        return False
    
    # Check conditions cond1 and cond2
    cond1 = results_route['scores']['score_route'] > 94.0  # we allow 6% of the route score to be missing
    cond2 = results_route['num_infractions'] == (
        len(results_route['infractions']['min_speed_infractions']) + 
        len(results_route['infractions']['outside_route_lanes'])
    )
    
    # Process if NOT (cond1 AND cond2), i.e., if the route crashed or has other issues
    return not (cond1 and cond2)


def main():
    """Main processing function."""
    source_path = Path(SOURCE_DIR)
    
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {source_path}")
        return
    
    # Process each scenario folder
    for scenario_folder in source_path.iterdir():
        if not scenario_folder.is_dir():
            continue
        
        scenario_name = scenario_folder.name
        print(f"\nProcessing scenario: {scenario_name}")
        
        # Skip folders that should not be processed
        if scenario_name in NO_CHANGE_FOLDERS:
            print(f"  Skipping {scenario_name} (no changes needed)")
            continue
        
        # Handle OppositeVehicleRunningRedLight with rule-based corrections
        if scenario_name == CORRECT_FOLDER:
            print(f"  Applying rule-based corrections to {scenario_name}...")
            
            # Process each route folder
            route_folders = sorted([d for d in scenario_folder.iterdir() if d.is_dir()])
            processed_count = 0
            skipped_count = 0
            
            for route_folder in route_folders:
                # Check if this route should be processed
                if not should_process_route(route_folder):
                    skipped_count += 1
                    continue
                
                processed_count += 1
                
                # Process measurement files with corrections and create measurements_new
                process_route_folder(route_folder, scenario_name)
            
            print(f"  Processed {processed_count} routes, skipped {skipped_count} routes")
        
        # Other folders: skip for now
        else:
            print(f"  Skipping {scenario_name} (not in scope)")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

