#!/usr/bin/env python3
"""
Extract eval step information from txt log files and generate CSV files
"""

import os
import re
import csv
from pathlib import Path
from typing import List, Dict, Optional

# Folders to process
FOLDERS = [
    "data/pending_process/simlingo_bench2drive220_newcommand_part1",
    "data/pending_process/simlingo_bench2drive220_newcommand_part2",
    "data/pending_process/simlingo_bench2drive220_newcommand_part2_2",
    "data/pending_process/simlingo_bench2drive220_newcommand_part2_3",
    "data/pending_process/simlingo_bench2drive220_newcommand_part2_4",
]

# CSV columns
CSV_COLUMNS = [
    "route",
    "step",
    "far_command",
    "next_far_command",
    "last_command",
    "use_last_command",
    "actual_command",
    "command",
    "next_command",
    "dist_to_command",
    "command_replaced"
]


def parse_debug_line(line: str) -> Optional[Dict]:
    """
    Parse a DEBUG line to extract step information
    Example: [DEBUG] Step 0: far_command=4, next_far_command=4, last_command=-1, use_last_command=False, actual_command=4, command='follow the road', next_command='follow the road', dist_to_command=51, command_replaced=False
    """
    # Match the DEBUG line pattern
    pattern = r'\[DEBUG\] Step (\d+): far_command=(-?\d+), next_far_command=(-?\d+), last_command=(-?\d+), use_last_command=(True|False), actual_command=(-?\d+), command=\'([^\']+)\', next_command=\'([^\']+)\', dist_to_command=(\d+), command_replaced=(True|False)'
    
    match = re.search(pattern, line)
    if not match:
        return None
    
    step = int(match.group(1))
    far_command = int(match.group(2))
    next_far_command = int(match.group(3))
    last_command = int(match.group(4))
    use_last_command = match.group(5) == 'True'
    actual_command = int(match.group(6))
    command = match.group(7)
    next_command = match.group(8)
    dist_to_command = int(match.group(9))
    command_replaced = match.group(10) == 'True'
    
    return {
        "step": step,
        "far_command": far_command,
        "next_far_command": next_far_command,
        "last_command": last_command,
        "use_last_command": use_last_command,
        "actual_command": actual_command,
        "command": command,
        "next_command": next_command,
        "dist_to_command": dist_to_command,
        "command_replaced": command_replaced
    }


def extract_route_id(line: str) -> Optional[str]:
    """
    Extract route ID from "Preparing RouteScenario_XXX" line
    Example: [1m========= Preparing RouteScenario_1792 (repetition 0) =========[0m
    """
    pattern = r'Preparing RouteScenario_(\d+)'
    match = re.search(pattern, line)
    if match:
        return f"RouteScenario_{match.group(1)}"
    return None


def process_txt_file(txt_path: str) -> List[Dict]:
    """
    Process a single txt file and extract all step information
    Returns a list of dictionaries with step data
    """
    results = []
    current_route = None
    
    print(f"  Processing: {os.path.basename(txt_path)}")
    
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Check for route start
            route_id = extract_route_id(line)
            if route_id:
                current_route = route_id
                continue
            
            # Check for DEBUG step line
            if '[DEBUG] Step' in line:
                step_data = parse_debug_line(line)
                if step_data and current_route:
                    step_data["route"] = current_route
                    results.append(step_data)
    
    print(f"    Extracted {len(results)} steps from {len(set(r['route'] for r in results))} routes")
    return results


def process_folder(folder_path: str) -> str:
    """
    Process all txt files in a folder and generate a CSV file
    Returns the path to the generated CSV file
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Warning: Folder does not exist: {folder_path}")
        return None
    
    # Find all txt files
    txt_files = sorted(folder_path.glob("*.txt"))
    if not txt_files:
        print(f"Warning: No txt files found in {folder_path}")
        return None
    
    print(f"\nProcessing folder: {folder_path}")
    print(f"Found {len(txt_files)} txt file(s)")
    
    # Process all txt files
    all_results = []
    for txt_file in txt_files:
        results = process_txt_file(str(txt_file))
        all_results.extend(results)
    
    # Generate CSV filename
    csv_filename = folder_path.name + "_eval_steps_extracted.csv"
    csv_path = folder_path / csv_filename
    
    # Write CSV file
    print(f"\nWriting CSV: {csv_path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        
        # Sort by route and step
        all_results.sort(key=lambda x: (x['route'], x['step']))
        
        for row in all_results:
            # Convert to CSV row format
            csv_row = {
                "route": row["route"],
                "step": row["step"],
                "far_command": row["far_command"],
                "next_far_command": row["next_far_command"],
                "last_command": row["last_command"],
                "use_last_command": row["use_last_command"],
                "actual_command": row["actual_command"],
                "command": row["command"],
                "next_command": row["next_command"],
                "dist_to_command": row["dist_to_command"],
                "command_replaced": row["command_replaced"]
            }
            writer.writerow(csv_row)
    
    print(f"✓ Generated CSV with {len(all_results)} rows")
    print(f"  Routes: {len(set(r['route'] for r in all_results))}")
    
    return str(csv_path)


def main():
    """Main function"""
    print("=" * 70)
    print("Extracting eval steps from txt files")
    print("=" * 70)
    
    csv_files = []
    for folder in FOLDERS:
        csv_path = process_folder(folder)
        if csv_path:
            csv_files.append(csv_path)
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Processed {len(csv_files)} folder(s):")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    print("\n✓ All CSV files generated successfully!")


if __name__ == '__main__':
    main()

