#!/usr/bin/env python3
"""
Calculate driving score and success rate from Bench2Drive evaluation results.
Works with partial results (temporary route results).
"""

import json
import sys
from pathlib import Path


def calculate_metrics(json_file_path):
    """
    Calculate driving score and success rate from evaluation results.
    
    Args:
        json_file_path: Path to the JSON results file
        
    Returns:
        dict: Dictionary containing metrics
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    records = data['_checkpoint']['records']
    entry_status = data.get('entry_status', 'Unknown')
    progress = data['_checkpoint'].get('progress', [0, 0])
    
    # Filter out failed/crashed routes for driving score calculation
    # But include them in total count for success rate calculation
    driving_scores = []
    completed_routes = []
    started_routes = []
    failed_routes = []
    total_routes = len(records)
    
    success_count = 0
    
    for record in records:
        status = record.get('status', '')
        
        # Track different route statuses
        if status == 'Started':
            started_routes.append(record)
            # Started routes can be resumed, but don't have scores yet
            continue
        elif status == 'Failed - Simulation crashed' and record.get('scores', {}).get('score_composed', 0) == 0:
            failed_routes.append(record)
            continue
        
        # Collect driving scores from all routes (including failed ones with scores)
        score_composed = record.get('scores', {}).get('score_composed', 0)
        if score_composed is not None:
            driving_scores.append(score_composed)
        
        # Check if route is completed or perfect
        if status == 'Completed' or status == 'Perfect':
            completed_routes.append(record)
            
            # Check for success: no infractions except min_speed_infractions
            infractions = record.get('infractions', {})
            success_flag = True
            
            for infraction_type, infraction_list in infractions.items():
                if infraction_type != 'min_speed_infractions' and len(infraction_list) > 0:
                    success_flag = False
                    break
            
            if success_flag:
                success_count += 1
    
    # Calculate metrics
    if len(driving_scores) > 0:
        avg_driving_score = sum(driving_scores) / len(driving_scores)
    else:
        avg_driving_score = 0.0
    
    if len(driving_scores) > 0:
        success_rate = success_count / len(driving_scores)
    else:
        success_rate = 0.0
    
    # Additional statistics
    completion_rate = len(completed_routes) / total_routes if total_routes > 0 else 0.0
    
    # Check if evaluation can be resumed
    can_resume = False
    resume_reason = ""
    if entry_status == 'Started':
        can_resume = True
        resume_reason = "Evaluation was started but interrupted (can resume)"
    elif entry_status == 'Crashed':
        can_resume = True
        resume_reason = "Evaluation crashed (can resume)"
    elif entry_status in ['Completed', '']:
        if progress and len(progress) >= 2 and progress[0] < progress[1]:
            can_resume = True
            resume_reason = f"Not all routes completed (progress: {progress[0]}/{progress[1]})"
        else:
            resume_reason = "Evaluation appears complete"
    else:
        can_resume = True
        resume_reason = f"Unknown status '{entry_status}' (can try to resume)"
    
    return {
        'driving_score': avg_driving_score,
        'success_rate': success_rate,
        'total_routes': total_routes,
        'completed_routes': len(completed_routes),
        'started_routes': len(started_routes),
        'failed_routes': len(failed_routes),
        'successful_routes': success_count,
        'routes_with_scores': len(driving_scores),
        'completion_rate': completion_rate,
        'progress': progress,
        'entry_status': entry_status,
        'can_resume': can_resume,
        'resume_reason': resume_reason
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_metrics.py <path_to_json_file>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    if not Path(json_file_path).exists():
        print(f"Error: File not found: {json_file_path}")
        sys.exit(1)
    
    metrics = calculate_metrics(json_file_path)
    
    print("\n" + "="*60)
    print("Bench2Drive Evaluation Metrics (Temporary Results)")
    print("="*60)
    print(f"\nDriving Score:     {metrics['driving_score']:.4f}")
    print(f"Success Rate:      {metrics['success_rate']:.4f} ({metrics['success_rate']*100:.2f}%)")
    print(f"\nRoute Statistics:")
    print(f"  Total Routes:     {metrics['total_routes']}")
    print(f"  Completed Routes: {metrics['completed_routes']}")
    if metrics['started_routes'] > 0:
        print(f"  Started Routes:   {metrics['started_routes']} (can be resumed)")
    if metrics['failed_routes'] > 0:
        print(f"  Failed Routes:    {metrics['failed_routes']}")
    print(f"  Successful Routes: {metrics['successful_routes']}")
    print(f"  Routes with Scores: {metrics['routes_with_scores']}")
    print(f"  Completion Rate:  {metrics['completion_rate']:.4f} ({metrics['completion_rate']*100:.2f}%)")
    
    if metrics['progress']:
        print(f"\nProgress: {metrics['progress']}")
    
    print(f"\nEntry Status: {metrics['entry_status']}")
    if metrics['can_resume']:
        print(f"🔄 Resume: YES - {metrics['resume_reason']}")
    else:
        print(f"✅ Resume: NO - {metrics['resume_reason']}")
    print("="*60)
    
    # Also output as JSON for easy parsing
    print("\nJSON Output:")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
