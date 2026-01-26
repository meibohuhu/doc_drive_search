#!/usr/bin/env python3
"""
Submit SimLingo evaluation jobs to SLURM cluster.
This script generates and submits SLURM batch scripts for evaluating SimLingo on Bench2Drive.
"""

import os
import subprocess
import glob
from pathlib import Path

# Configuration - modify these paths
CONFIG = {
    "checkpoint": "/home/mh2803/projects/simlingo/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",
    "route_path": "/home/mh2803/projects/simlingo/Bench2Drive/leaderboard/data/bench2drive_split",
    "seeds": [1],  # Evaluation seeds
    "out_root": "/home/mh2803/projects/simlingo/eval_results/Bench2Drive",
    "carla_root": "/home/mh2803/softwar/carla0915",  # Adjust if CARLA is installed elsewhere
    "repo_root": "/home/mh2803/projects/simlingo",
    "partition": "tier3",
    "max_parallel_jobs": 1,  # Maximum number of parallel jobs
}

# Port ranges (adjust if needed to avoid conflicts)
PORT_START = 20000
TM_PORT_START = 30000
PORT_STEP = 50


def get_route_files(route_path):
    """Get all route XML files from the route path."""
    route_files = glob.glob(os.path.join(route_path, "route_*.xml"))
    route_files.sort()
    return route_files


def create_eval_script(route_file, route_id, seed, port, tm_port, config):
    """Create a SLURM evaluation script for a single route."""
    script_content = f"""#!/bin/bash -l
#SBATCH --job-name=simlingo_eval_{route_id}_s{seed}
#SBATCH --error={config['out_root']}/simlingo/bench2drive/{seed}/err/{route_id}_err.log
#SBATCH --output={config['out_root']}/simlingo/bench2drive/{seed}/out/{route_id}_out.log
#SBATCH --account=llm-gen-agent
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition={config['partition']}
#SBATCH --mem=40G

echo "JOB ID $SLURM_JOB_ID"
echo "Route: {route_file}"
echo "Route ID: {route_id}"
echo "Seed: {seed}"

# Load spack modules
spack load /lhqcen5
spack load cuda@12.4.0/obxqih4

# Set up environment paths
export PATH="/home/mh2803/miniconda3/envs/simlingo/bin:$PATH"
export PYTHONPATH="{config['repo_root']}:${{PYTHONPATH}}"

# Set up CARLA paths
export CARLA_ROOT="{config['carla_root']}"
export PYTHONPATH="${{PYTHONPATH}}:${{CARLA_ROOT}}/PythonAPI/carla"
export PYTHONPATH="${{PYTHONPATH}}:${{CARLA_ROOT}}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
export PYTHONPATH="${{PYTHONPATH}}:{config['repo_root']}/Bench2Drive/leaderboard"
export PYTHONPATH="${{PYTHONPATH}}:{config['repo_root']}/Bench2Drive/scenario_runner"
export SCENARIO_RUNNER_ROOT="{config['repo_root']}/Bench2Drive/scenario_runner"

# Create output directories
VIZ_PATH="{config['out_root']}/simlingo/bench2drive/{seed}/viz/{route_id}"
RESULT_FILE="{config['out_root']}/simlingo/bench2drive/{seed}/res/{route_id}_res.json"
mkdir -p "${{VIZ_PATH}}"
mkdir -p "$(dirname ${{RESULT_FILE}})"

export SAVE_PATH="${{VIZ_PATH}}"

# Change to project directory
cd {config['repo_root']}

# Run evaluation
python -u {config['repo_root']}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py \\
    --routes={route_file} \\
    --repetitions=1 \\
    --track=SENSORS \\
    --checkpoint=${{RESULT_FILE}} \\
    --timeout=600 \\
    --agent={config['repo_root']}/team_code/agent_simlingo.py \\
    --agent-config={config['checkpoint']} \\
    --traffic-manager-seed={seed} \\
    --port={port} \\
    --traffic-manager-port={tm_port}

echo ""
echo "✅ Evaluation completed!"
"""
    return script_content


def submit_jobs(config):
    """Submit evaluation jobs for all routes and seeds."""
    route_files = get_route_files(config['route_path'])
    
    if not route_files:
        print(f"Error: No route files found in {config['route_path']}")
        return
    
    print(f"Found {len(route_files)} route files")
    print(f"Will submit jobs for {len(config['seeds'])} seeds")
    print(f"Total jobs: {len(route_files) * len(config['seeds'])}")
    
    # Create output directories
    for seed in config['seeds']:
        for subdir in ['out', 'err', 'res', 'viz']:
            os.makedirs(
                os.path.join(config['out_root'], 'simlingo', 'bench2drive', str(seed), subdir),
                exist_ok=True
            )
    
    # Create scripts directory
    scripts_dir = os.path.join(config['out_root'], 'simlingo', 'bench2drive', 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)
    
    port_counter = 0
    submitted_jobs = []
    
    for seed in config['seeds']:
        for route_file in route_files:
            # Extract route ID from filename
            route_id = os.path.basename(route_file).replace('route_', '').replace('.xml', '')
            
            # Calculate ports
            port = PORT_START + (port_counter % 200) * PORT_STEP
            tm_port = TM_PORT_START + (port_counter % 200) * PORT_STEP
            port_counter += 1
            
            # Create script
            script_content = create_eval_script(
                route_file, route_id, seed, port, tm_port, config
            )
            
            script_path = os.path.join(scripts_dir, f'eval_{route_id}_s{seed}.sh')
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(script_path, 0o755)
            
            # Submit job
            try:
                result = subprocess.run(
                    ['sbatch', script_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                job_id = result.stdout.strip().split()[-1]
                submitted_jobs.append({
                    'job_id': job_id,
                    'route_id': route_id,
                    'seed': seed,
                    'script': script_path
                })
                print(f"Submitted job {job_id}: route {route_id}, seed {seed}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job for route {route_id}, seed {seed}: {e}")
    
    print(f"\n✅ Submitted {len(submitted_jobs)} jobs")
    print(f"Scripts saved in: {scripts_dir}")
    print(f"\nMonitor jobs with: squeue -u $USER")
    print(f"Cancel all jobs with: scancel -u $USER")


if __name__ == "__main__":
    import sys
    
    # Allow overriding config from command line
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("Usage: python submit_eval_simlingo_slurm.py [--checkpoint PATH] [--route-path PATH] [--seeds 1,2,3]")
            print("\nModify CONFIG dictionary in the script to change default settings.")
            sys.exit(0)
    
    submit_jobs(CONFIG)

