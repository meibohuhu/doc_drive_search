#!/usr/bin/env python3
"""
Script to merge multiple JSON result files and compute global statistics
70.91%
87.102
RouteScenario_17569 重复了 eval_20260205_064857.txt， eval_20260205_052309.txt
"""

import json
import math
from collections import defaultdict
from typing import Dict, List, Any

# File paths to merge
FILE_PATHS = [
    "data/pending_process/simlingo_bench2drive220_newcommand_part1.json",
    "data/pending_process/simlingo_bench2drive220_newcommand_part2_1_done.json",
    "data/pending_process/simlingo_bench2drive220_newcommand_part2_1.json",
    "data/pending_process/simlingo_bench2drive220_newcommand_part2_2_done.json",
    "data/pending_process/simlingo_bench2drive220_newcommand_part2_2.json",
    "data/pending_process/simlingo_bench2drive220_newcommand_part2_3.json",
    "data/pending_process/simlingo_bench2drive220_newcommand_part2_4.json",
    "data/pending_process/simlingo_bench2drive220_newcommand_part2.json",
    "data/pending_process/simlingo_bench2drive220_newcommand_part3.json",
]

OUTPUT_FILE = "data/pending_process/simlingo_bench2drive220_newcommand_merged.json"

ROUND_DIGITS = 3

# Infraction type mapping
INFRACTION_KEYS = [
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "min_speed_infractions",
    "yield_emergency_vehicle_infractions",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout"
]


def get_infractions_value(record: Dict, key: str) -> float:
    """Get the count of infractions for a given key"""
    infractions = record.get("infractions", {})
    if key == "outside_route_lanes":
        # Special handling for outside_route_lanes (percentage based)
        outside_list = infractions.get(key, [])
        if outside_list:
            # Extract percentage from the string
            for item in outside_list:
                if "%" in item:
                    try:
                        return float(item.split("%")[0])
                    except:
                        pass
        return 0.0
    else:
        # Count the number of infractions
        infraction_list = infractions.get(key, [])
        return len(infraction_list) if isinstance(infraction_list, list) else 0


def check_duplicates(route_ids: List[str]):
    """Checks that all route ids are present only once in the files"""
    seen = {}
    for route_id in route_ids:
        if route_id in seen:
            raise ValueError(f"Stopping. Found that the route {route_id} has more than one record (first seen at index {seen[route_id]})")
        seen[route_id] = len(seen)


def load_and_merge_records() -> tuple:
    """Load all files and merge records, handling duplicates by keeping the best record"""
    records_dict = {}  # route_id -> record
    sensors = None
    total_progress = 0
    duplicate_count = 0
    
    for file_path in FILE_PATHS:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if not data or '_checkpoint' not in data:
                print(f"Warning: {file_path} has no valid checkpoint data")
                continue
            
            records = data.get('_checkpoint', {}).get('records', [])
            total_progress += data.get('_checkpoint', {}).get('progress', [0, 0])[1]
            
            # Check sensors consistency
            file_sensors = data.get('sensors', [])
            if file_sensors:
                if sensors is None:
                    sensors = file_sensors
                elif file_sensors != sensors:
                    print(f"Warning: {file_path} has different sensors: {file_sensors} vs {sensors}")
            
            # Add records, handling duplicates
            for record in records:
                route_id = record.get('route_id')
                if not route_id:
                    continue
                
                if route_id in records_dict:
                    # Duplicate found - keep the better record
                    existing = records_dict[route_id]
                    # Prefer Completed status, or record with more data
                    existing_status = existing.get('status', '')
                    new_status = record.get('status', '')
                    
                    if new_status == 'Completed' and existing_status != 'Completed':
                        records_dict[route_id] = record
                        duplicate_count += 1
                        print(f"  Replaced {route_id} (kept Completed version)")
                    elif existing_status == 'Completed' and new_status != 'Completed':
                        # Keep existing
                        duplicate_count += 1
                        print(f"  Skipped duplicate {route_id} (kept Completed version)")
                    else:
                        # Both same status, keep the one with more infractions data (more complete)
                        existing_infractions = sum(len(v) if isinstance(v, list) else 0 
                                                  for v in existing.get('infractions', {}).values())
                        new_infractions = sum(len(v) if isinstance(v, list) else 0 
                                             for v in record.get('infractions', {}).values())
                        if new_infractions > existing_infractions:
                            records_dict[route_id] = record
                            duplicate_count += 1
                            print(f"  Replaced {route_id} (kept more complete version)")
                        else:
                            duplicate_count += 1
                            print(f"  Skipped duplicate {route_id} (kept existing version)")
                else:
                    records_dict[route_id] = record
        
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Convert to list and sort
    all_records = list(records_dict.values())
    all_records.sort(key=lambda x: (
        int(x['route_id'].split('_')[1]),
        int(x['route_id'].split('_rep')[-1])
    ))
    
    all_route_ids = [r['route_id'] for r in all_records]
    
    if duplicate_count > 0:
        print(f"\nHandled {duplicate_count} duplicate route(s)")
    
    return all_records, all_route_ids, sensors, total_progress


def compute_global_statistics(records: List[Dict]) -> Dict:
    """Compute global statistics from all records"""
    if not records:
        return {}
    
    total_routes = len(records)
    
    # Initialize global record
    global_record = {
        "index": -1,
        "route_id": -1,
        "status": "Failed",  # Will be updated based on results
        "infractions": {key: 0.0 for key in INFRACTION_KEYS},
        "scores_mean": {
            "score_composed": 0.0,
            "score_route": 0.0,
            "score_penalty": 0.0
        },
        "scores_std_dev": {
            "score_composed": 0.0,
            "score_route": 0.0,
            "score_penalty": 0.0
        },
        "meta": {
            "total_length": 0.0,
            "duration_game": 0.0,
            "duration_system": 0.0,
            "exceptions": []
        }
    }
    
    # Calculate mean scores
    total_composed = 0.0
    total_route = 0.0
    total_penalty = 0.0
    completed_count = 0  # Routes with score_composed == 100
    
    for record in records:
        scores = record.get("scores", {})
        score_composed = scores.get("score_composed", 0.0)
        total_composed += score_composed
        total_route += scores.get("score_route", 0.0)
        total_penalty += scores.get("score_penalty", 0.0)
        
        # Count successful routes: score_composed == 100
        if score_composed == 100.0:
            completed_count += 1
        
        # Accumulate meta data
        meta = record.get("meta", {})
        global_record["meta"]["total_length"] += meta.get("route_length", 0.0)
        global_record["meta"]["duration_game"] += meta.get("duration_game", 0.0)
        global_record["meta"]["duration_system"] += meta.get("duration_system", 0.0)
        
        # Collect exceptions (routes with score_composed != 100)
        score_composed = scores.get("score_composed", 0.0)
        if score_composed != 100.0:
            route_id = record.get("route_id", "Unknown")
            status = record.get("status", "Failed")
            # Use status from record, or create a message based on score
            if status == "Completed" and score_composed < 100:
                status = f"Completed with score {score_composed}"
            global_record["meta"]["exceptions"].append([
                route_id,
                record.get("index", -1),
                status
            ])
    
    # Calculate means
    global_record["scores_mean"]["score_composed"] = round(total_composed / total_routes, ROUND_DIGITS)
    global_record["scores_mean"]["score_route"] = round(total_route / total_routes, ROUND_DIGITS)
    global_record["scores_mean"]["score_penalty"] = round(total_penalty / total_routes, ROUND_DIGITS)
    
    # Calculate standard deviations
    if total_routes > 1:
        for record in records:
            scores = record.get("scores", {})
            for key in global_record["scores_std_dev"]:
                diff = scores.get(key, 0.0) - global_record["scores_mean"][key]
                global_record["scores_std_dev"][key] += diff * diff
        
        for key in global_record["scores_std_dev"]:
            variance = global_record["scores_std_dev"][key] / (total_routes - 1)
            global_record["scores_std_dev"][key] = round(math.sqrt(variance), ROUND_DIGITS)
    else:
        for key in global_record["scores_std_dev"]:
            global_record["scores_std_dev"][key] = 0.0
    
    # Calculate infractions per km
    km_driven = 0.0
    for record in records:
        scores = record.get("scores", {})
        meta = record.get("meta", {})
        route_length = meta.get("route_length", 0.0)
        score_route = scores.get("score_route", 0.0)
        km_driven += (route_length / 1000.0) * (score_route / 100.0)
        
        # Accumulate infractions
        for key in INFRACTION_KEYS:
            global_record["infractions"][key] += get_infractions_value(record, key)
    
    km_driven = max(km_driven, 0.001)
    
    # Normalize infractions per km (except outside_route_lanes which is percentage)
    for key in INFRACTION_KEYS:
        if key != "outside_route_lanes":
            global_record["infractions"][key] /= km_driven
        global_record["infractions"][key] = round(global_record["infractions"][key], ROUND_DIGITS)
    
    # Determine global status based on success rate (score_composed == 100)
    success_rate = completed_count / total_routes if total_routes > 0 else 0.0
    if success_rate >= 0.95:
        global_record["status"] = "Completed"
    elif success_rate >= 0.5:
        global_record["status"] = "Failed"
    else:
        global_record["status"] = "Failed"
    
    return global_record


def create_output_json(records: List[Dict], global_record: Dict, sensors: List[str], 
                      total_routes: int, total_progress: int) -> Dict:
    """Create the final merged JSON structure"""
    
    # Prepare values and labels for the leaderboard format
    values = [
        str(global_record["scores_mean"]["score_composed"]),
        str(global_record["scores_mean"]["score_route"]),
        str(global_record["scores_mean"]["score_penalty"]),
        str(global_record["infractions"]["collisions_pedestrian"]),
        str(global_record["infractions"]["collisions_vehicle"]),
        str(global_record["infractions"]["collisions_layout"]),
        str(global_record["infractions"]["red_light"]),
        str(global_record["infractions"]["stop_infraction"]),
        str(global_record["infractions"]["outside_route_lanes"]),
        str(global_record["infractions"]["route_dev"]),
        str(global_record["infractions"]["route_timeout"]),
        str(global_record["infractions"]["vehicle_blocked"]),
        str(global_record["infractions"]["yield_emergency_vehicle_infractions"]),
        str(global_record["infractions"]["scenario_timeouts"]),
        str(global_record["infractions"]["min_speed_infractions"])
    ]
    
    labels = [
        "Avg. driving score",
        "Avg. route completion",
        "Avg. infraction penalty",
        "Collisions with pedestrians",
        "Collisions with vehicles",
        "Collisions with layout",
        "Red lights infractions",
        "Stop sign infractions",
        "Off-road infractions",
        "Route deviations",
        "Route timeouts",
        "Agent blocked",
        "Yield emergency vehicles infractions",
        "Scenario timeouts",
        "Min speed infractions"
    ]
    
    # Update record indices
    for idx, record in enumerate(records):
        record["index"] = idx
    
    output = {
        "_checkpoint": {
            "global_record": global_record,
            "progress": [total_routes, total_progress],
            "records": records
        },
        "entry_status": "Finished" if total_routes == total_progress else "Started",
        "eligible": total_routes == total_progress,
        "sensors": sensors or [],
        "values": values,
        "labels": labels
    }
    
    return output


def main():
    """Main function to merge files and compute statistics"""
    print("Loading and merging records from all files...")
    records, route_ids, sensors, total_progress = load_and_merge_records()
    
    print(f"Total records loaded: {len(records)}")
    print(f"Total progress: {total_progress}")
    
    # Check for duplicates (should be none after handling)
    print("Checking for duplicate route IDs...")
    try:
        check_duplicates(route_ids)
        print("✓ No duplicates found")
    except ValueError as e:
        print(f"✗ {e}")
        print("This should not happen after duplicate handling. Please check the code.")
        return
    
    # Compute global statistics
    print("Computing global statistics...")
    global_record = compute_global_statistics(records)
    
    # Calculate success rate: score_composed == 100
    completed = sum(1 for r in records if r.get("scores", {}).get("score_composed", 0) == 100.0)
    success_rate = (completed / len(records) * 100) if records else 0.0
    
    print(f"\n=== Global Statistics ===")
    print(f"Total routes: {len(records)}")
    print(f"Completed routes: {completed}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Avg. driving score: {global_record['scores_mean']['score_composed']:.3f}")
    print(f"Avg. route completion: {global_record['scores_mean']['score_route']:.3f}")
    print(f"Avg. infraction penalty: {global_record['scores_mean']['score_penalty']:.3f}")
    
    # Create output JSON
    print("\nCreating merged JSON file...")
    output = create_output_json(records, global_record, sensors, len(records), total_progress)
    
    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"✓ Merged results saved to: {OUTPUT_FILE}")
    print(f"\nSummary:")
    print(f"  - Total routes: {len(records)}")
    print(f"  - Success rate: {success_rate:.2f}%")
    print(f"  - Driving score: {global_record['scores_mean']['score_composed']:.3f}")


if __name__ == '__main__':
    main()

