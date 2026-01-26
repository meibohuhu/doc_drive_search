# SimLingo Evaluation Setup Guide

This guide explains how to set up CARLA locally and submit evaluation jobs to SLURM.

## 1. Setup CARLA Locally

Run the setup script to download and install CARLA 0.9.15:

```bash
cd /home/mh2803/projects/simlingo/scripts_mh
./setup_carla_local.sh
```

This will:
- Download CARLA 0.9.15 to `~/software/carla0915`
- Extract and set up CARLA
- Download and import additional maps

**Note**: Adjust the `CARLA_INSTALL_DIR` variable in the script if you want to install CARLA elsewhere.

After setup, add these to your environment (or add to `~/.bashrc`):

```bash
export CARLA_ROOT=~/software/carla0915
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
```

## 2. Submit Single Evaluation Job

To submit a single evaluation job manually:

```bash
sbatch scripts_mh/eval_simlingo_cluster.sh [route_file] [route_id] [seed] [checkpoint] [port] [tm_port]
```

Example:
```bash
sbatch scripts_mh/eval_simlingo_cluster.sh \
    /home/mh2803/projects/simlingo/Bench2Drive/leaderboard/data/bench2drive_split/route_000.xml \
    000 \
    1 \
    /home/mh2803/projects/simlingo/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt \
    20000 \
    30000
```

## 3. Submit Multiple Evaluation Jobs (Recommended)

Use the Python script to automatically submit all routes:

```bash
cd /home/mh2803/projects/simlingo/scripts_mh
python submit_eval_simlingo_slurm.py
```

Before running, modify the `CONFIG` dictionary in `submit_eval_simlingo_slurm.py`:

```python
CONFIG = {
    "checkpoint": "/path/to/your/checkpoint.pt",
    "route_path": "/path/to/route/files",
    "seeds": [1, 2, 3],
    "out_root": "/path/to/output",
    "carla_root": "/path/to/carla",  # Should match your CARLA installation
    "repo_root": "/home/mh2803/projects/simlingo",
    "partition": "tier3",
    "max_parallel_jobs": 10,
}
```

The script will:
- Find all route files in the specified path
- Create SLURM batch scripts for each route × seed combination
- Submit all jobs to the cluster
- Save scripts in `{out_root}/simlingo/bench2drive/scripts/`

## 4. Monitor Jobs

Check job status:
```bash
squeue -u $USER
```

Check specific job:
```bash
squeue -j <JOB_ID>
```

View job output:
```bash
tail -f /home/mh2803/projects/simlingo/scripts/cluster_logs/eval_out_<JOB_ID>.txt
```

View job errors:
```bash
tail -f /home/mh2803/projects/simlingo/scripts/cluster_logs/eval_err_<JOB_ID>.txt
```

## 5. Cancel Jobs

Cancel a specific job:
```bash
scancel <JOB_ID>
```

Cancel all your jobs:
```bash
scancel -u $USER
```

Cancel jobs by name pattern:
```bash
scancel -n simlingo_eval
```

## 6. Check Results

Results will be saved in:
- **JSON results**: `{out_root}/simlingo/bench2drive/{seed}/res/{route_id}_res.json`
- **Visualizations**: `{out_root}/simlingo/bench2drive/{seed}/viz/{route_id}/`
- **Logs**: `{out_root}/simlingo/bench2drive/{seed}/out/{route_id}_out.log`
- **Errors**: `{out_root}/simlingo/bench2drive/{seed}/err/{route_id}_err.log`

## Troubleshooting

### CARLA not found
- Make sure `CARLA_ROOT` is set correctly
- Check that CARLA is installed at the specified path
- Verify the Python egg file exists: `${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg`

### Port conflicts
- Each job needs unique ports
- The submission script automatically assigns ports, but if you have many jobs, adjust `PORT_START` and `TM_PORT_START` in the script

### Job fails immediately
- Check the error log: `cat {out_root}/simlingo/bench2drive/{seed}/err/{route_id}_err.log`
- Verify checkpoint path is correct
- Ensure route files exist
- Check that conda environment is activated correctly

### Out of memory
- Increase `--mem` in the SLURM script (currently 40G)
- Reduce batch size if applicable

## Notes

- Evaluation jobs typically take 10-30 minutes per route
- Make sure you have enough SLURM credits/allocations
- The `max_parallel_jobs` setting in the Python script is informational only - actual limits depend on your SLURM account settings

