#!/usr/bin/env python3
"""
Merge records from multiple result folders into a single JSON file.
"""
import json
from pathlib import Path

def merge_records(folder_paths, output_path):
    """
    Merge all records from multiple folders' res/*_res.json files.

    Args:
        folder_paths: List of folder paths to merge
        output_path: Path to output merged JSON file
    """
    merged_records = []
    source_folders = []

    # Process each folder
    for folder_path in folder_paths:
        folder_res = Path(folder_path) / "res"
        if not folder_res.exists():
            print(f"Warning: {folder_res} does not exist, skipping...")
            continue

        print(f"\nProcessing folder: {folder_path}")
        source_folders.append(str(folder_path))
        folder_record_count = 0

        for json_file in sorted(folder_res.glob("*_res.json")):
            with open(json_file, 'r') as f:
                data = json.load(f)
                if "_checkpoint" in data and "records" in data["_checkpoint"]:
                    records = data["_checkpoint"]["records"]
                    merged_records.extend(records)
                    folder_record_count += len(records)

        print(f"  Added {folder_record_count} record(s) from this folder")

    # Write merged records
    output_data = {
        "merged_records": merged_records,
        "total_count": len(merged_records),
        "source_folders": source_folders
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nMerge complete!")
    print(f"Total records merged: {len(merged_records)}")
    print(f"Output file: {output_path}")

    return len(merged_records)

if __name__ == "__main__":
    # Define the folders to merge
    base_path = "/code/doc_drive_search/eval_results/agent_simlingo_cluster_withlmdrive_usingspeed/simlingo/bench2drive"
    folders = [
        f"{base_path}/1",
        f"{base_path}/3"
    ]

    output = "/code/doc_drive_search/eval_results/agent_simlingo_cluster_withlmdrive_usingspeed/simlingo/bench2drive/merged_records.json"

    merge_records(folders, output)
