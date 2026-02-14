#!/usr/bin/env python3
"""
Upload packaged evaluation results to HuggingFace Hub
python scripts_adobe/upload_to_hf.py PhoenixHu/agent_simlingo_cluster_targetpoint

"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, login
from tqdm import tqdm

# Configuration
TEMP_DIR = "/tmp/eval_results_upload"
DEFAULT_REPO = "your-username/your-repo-name"

def get_file_size(filepath):
    """Get human-readable file size"""
    size = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <hf-repo-name> [hf-token]")
        print(f"Example: {sys.argv[0]} username/repo-name")
        print(f"         {sys.argv[0]} username/repo-name hf_xxxxx")
        sys.exit(1)

    repo_id = sys.argv[1]
    hf_token = sys.argv[2] if len(sys.argv) > 2 else None

    # Validate repo name
    if repo_id == DEFAULT_REPO:
        print("Error: Please provide a valid HuggingFace repository name")
        sys.exit(1)

    print("=" * 60)
    print("Upload Evaluation Results to HuggingFace")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Source: {TEMP_DIR}")
    print()

    # Login to HuggingFace
    try:
        if hf_token:
            print("Logging in with provided token...")
            login(token=hf_token)
        else:
            print("Checking HuggingFace authentication...")
            # Will use existing token from cache
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        print("\nPlease either:")
        print("  1. Provide token: python upload_to_hf.py <repo> <token>")
        print("  2. Login first: huggingface-cli login")
        sys.exit(1)

    print()

    # Check if repo exists
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"✓ Repository exists: {repo_id}")
    except:
        print(f"Repository '{repo_id}' not found.")
        create = input("Create it now? (y/n): ").strip().lower()
        if create == 'y':
            try:
                api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)
                print(f"✓ Created repository: {repo_id}")
            except Exception as e:
                print(f"✗ Failed to create repository: {e}")
                sys.exit(1)
        else:
            print("Upload cancelled.")
            sys.exit(0)

    print()

    # Get all tar.gz files
    archive_files = sorted(Path(TEMP_DIR).glob("*.tar.gz"))

    if not archive_files:
        print(f"✗ No .tar.gz files found in {TEMP_DIR}")
        sys.exit(1)

    print(f"Found {len(archive_files)} files to upload:")
    total_size = 0
    for f in archive_files:
        size = get_file_size(f)
        total_size += f.stat().st_size
        print(f"  - {f.name} ({size})")

    print(f"\nTotal size: {get_file_size_bytes(total_size)}")
    print()

    # Confirm upload
    confirm = input("Proceed with upload? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Upload cancelled.")
        sys.exit(0)

    print()
    print("=" * 60)
    print("Starting upload...")
    print("=" * 60)
    print()

    # Upload files
    success_count = 0
    fail_count = 0

    for archive_file in archive_files:
        filename = archive_file.name
        remote_path = f"eval_results/{filename}"

        print(f"Uploading: {filename}")
        print(f"  Size: {get_file_size(archive_file)}")

        try:
            # Upload with progress bar
            api.upload_file(
                path_or_fileobj=str(archive_file),
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  ✓ Successfully uploaded to {remote_path}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ Upload failed: {e}")
            fail_count += 1

        print()

    # Summary
    print("=" * 60)
    print("Upload Summary")
    print("=" * 60)
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print()

    if fail_count == 0:
        print("✓ All files uploaded successfully!")
        print(f"\nView at: https://huggingface.co/datasets/{repo_id}")

        # Ask to clean up
        cleanup = input("\nDelete local archives? (y/n): ").strip().lower()
        if cleanup == 'y':
            for f in archive_files:
                f.unlink()
            print(f"✓ Deleted local archives from {TEMP_DIR}")
    else:
        print(f"✗ Some uploads failed. Local archives kept in: {TEMP_DIR}")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)

def get_file_size_bytes(size_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

if __name__ == "__main__":
    main()
