"""
Image Editing Evaluation Metrics Calculator
License: MIT
Description: Parses evaluation JSON outputs and calculates average scores 
             for IF, TA, VC, LP, SE, and any other metrics present.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

def load_json(file_path: Path) -> List[Dict[str, Any]]:
    """Loads and validates the JSON file."""
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            # Try to handle case where user might pass a single object wrapped in dict
            if isinstance(data, dict) and "tasks" in data:
                 return data["tasks"]
            print("❌ Error: JSON content is expected to be a list of objects.")
            sys.exit(1)
            
        return data
    except json.JSONDecodeError:
        print(f"❌ Error: Failed to decode JSON from {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

def calculate_averages(data: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """
    Iterates through data and calculates sums and counts for all metrics found.
    Dynamic: Doesn't hardcode metric names, adapts to whatever is in the JSON.
    """
    # Structure: {'IF': {'sum': 10, 'count': 2}, 'SE': {'sum': 5, 'count': 1}, ...}
    stats = {} 

    for item in data:
        # Support both flat structure or nested 'evaluation_results'
        results = item.get('evaluation_results', item)
        
        if not results or not isinstance(results, dict):
            continue

        for metric_key, metric_data in results.items():
            # We expect structure like: "IF": {"IF": 5, "rationale": "..."}
            # Or simple structure: "IF": 5
            score = 0
            
            # 1. Handle nested dictionary format (Standard from previous tool)
            if isinstance(metric_data, dict):
                # Try to find the score inside using the key name (e.g. data['IF']['IF'])
                if metric_key in metric_data:
                    score = metric_data[metric_key]
                # Fallback: look for generic 'score' key
                elif 'score' in metric_data:
                    score = metric_data['score']
                else:
                    continue # Skip if no score found
            
            # 2. Handle flat numeric format
            elif isinstance(metric_data, (int, float)):
                score = metric_data
            else:
                continue

            # Initialize metric stats if new
            if metric_key not in stats:
                stats[metric_key] = {'sum': 0.0, 'count': 0}
            
            # Accumulate
            try:
                stats[metric_key]['sum'] += float(score)
                stats[metric_key]['count'] += 1
            except ValueError:
                pass # Ignore non-numeric scores

    return stats

def print_report(stats: Dict[str, Dict], filename: str):
    """Prints a formatted table of the results."""
    print(f"\n📊 Evaluation Report for: {filename}")
    print("=" * 50)
    print(f"{'Metric':<10} | {'Average':<10} | {'Valid Samples':<15}")
    print("-" * 50)
    
    # Sort keys for consistent output (e.g., IF, LP, SE, TA, VC)
    sorted_keys = sorted(stats.keys())
    
    if not sorted_keys:
        print("No valid metrics found.")
        return

    for key in sorted_keys:
        total_sum = stats[key]['sum']
        count = stats[key]['count']
        average = total_sum / count if count > 0 else 0.0
        
        print(f"{key:<10} | {average:<10.2f} | {count:<15}")
    print("=" * 50)
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="Calculate average scores from evaluation JSON.")
    parser.add_argument("file_path", type=Path, help="Path to the .json results file")
    
    args = parser.parse_args()
    
    data = load_json(args.file_path)
    stats = calculate_averages(data)
    print_report(stats, args.file_path.name)

if __name__ == "__main__":
    main()
