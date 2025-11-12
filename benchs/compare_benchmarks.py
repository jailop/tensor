#!/usr/bin/env python3
"""
Benchmark Comparison Tool

Compares two benchmark runs and generates a detailed comparison report.
Shows performance improvements/regressions between runs.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def load_benchmark_json(filepath: str) -> Dict:
    """Load benchmark JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def parse_benchmark_name(name: str) -> Tuple[str, str]:
    """Parse benchmark name to extract test name and variant."""
    # Example: BM_NormalizeL1_1D_Small_mean
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in ['mean', 'median', 'stddev', 'cv']:
        return parts[0], parts[1]
    return name, 'single'


def format_time(ns: float) -> str:
    """Format time in nanoseconds to human-readable format."""
    if ns < 1000:
        return f"{ns:.2f} ns"
    elif ns < 1000000:
        return f"{ns/1000:.2f} us"
    elif ns < 1000000000:
        return f"{ns/1000000:.2f} ms"
    else:
        return f"{ns/1000000000:.2f} s"


def calculate_change(baseline: float, current: float) -> Tuple[float, str]:
    """Calculate percentage change and direction."""
    if baseline == 0:
        return 0.0, "N/A"
    
    change = ((current - baseline) / baseline) * 100
    if abs(change) < 0.5:
        direction = "→"  # No significant change
    elif change > 0:
        direction = "↓"  # Slower (regression)
    else:
        direction = "↑"  # Faster (improvement)
    
    return change, direction


def compare_benchmarks(baseline_file: str, current_file: str, output_file: str = None):
    """Compare two benchmark runs and generate report."""
    
    # Load benchmark data
    try:
        baseline_data = load_benchmark_json(baseline_file)
        current_data = load_benchmark_json(current_file)
    except Exception as e:
        print(f"Error loading benchmark files: {e}")
        return False
    
    # Extract benchmark results
    baseline_benchmarks = {b['name']: b for b in baseline_data.get('benchmarks', [])}
    current_benchmarks = {b['name']: b for b in current_data.get('benchmarks', [])}
    
    # Find common benchmarks
    common_names = set(baseline_benchmarks.keys()) & set(current_benchmarks.keys())
    
    if not common_names:
        print("No common benchmarks found between the two runs.")
        return False
    
    # Prepare comparison data
    comparisons = []
    total_improvement = 0
    total_regression = 0
    faster_count = 0
    slower_count = 0
    
    for name in sorted(common_names):
        baseline = baseline_benchmarks[name]
        current = current_benchmarks[name]
        
        baseline_time = baseline.get('real_time', baseline.get('cpu_time', 0))
        current_time = current.get('real_time', current.get('cpu_time', 0))
        
        change, direction = calculate_change(baseline_time, current_time)
        
        comparisons.append({
            'name': name,
            'baseline_time': baseline_time,
            'current_time': current_time,
            'change': change,
            'direction': direction
        })
        
        if direction == "↑":
            faster_count += 1
            total_improvement += abs(change)
        elif direction == "↓":
            slower_count += 1
            total_regression += abs(change)
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("BENCHMARK COMPARISON REPORT")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"Baseline: {Path(baseline_file).name}")
    report_lines.append(f"Current:  {Path(current_file).name}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("-" * 100)
    report_lines.append(f"{'Benchmark Name':<60} {'Baseline':>15} {'Current':>15} {'Change':>8}")
    report_lines.append("-" * 100)
    
    for comp in comparisons:
        baseline_str = format_time(comp['baseline_time'])
        current_str = format_time(comp['current_time'])
        
        if comp['direction'] == "→":
            change_str = f"{comp['direction']} {abs(comp['change']):>5.1f}%"
        else:
            change_str = f"{comp['direction']} {abs(comp['change']):>5.1f}%"
        
        # Clean up benchmark name for display
        display_name = comp['name'].replace('BM_', '').replace('_mean', '')
        report_lines.append(f"{display_name:<60} {baseline_str:>15} {current_str:>15} {change_str:>8}")
    
    report_lines.append("-" * 100)
    report_lines.append("")
    report_lines.append("SUMMARY")
    report_lines.append("-" * 100)
    report_lines.append(f"Total benchmarks compared: {len(comparisons)}")
    report_lines.append(f"Faster (↑):  {faster_count:3d} benchmarks (avg improvement: {total_improvement/faster_count if faster_count > 0 else 0:.1f}%)")
    report_lines.append(f"Slower (↓):  {slower_count:3d} benchmarks (avg regression:  {total_regression/slower_count if slower_count > 0 else 0:.1f}%)")
    report_lines.append(f"No change (→): {len(comparisons) - faster_count - slower_count:3d} benchmarks")
    report_lines.append("")
    
    if faster_count > slower_count:
        report_lines.append("✓ Overall: Performance IMPROVED")
    elif slower_count > faster_count:
        report_lines.append("✗ Overall: Performance REGRESSED")
    else:
        report_lines.append("→ Overall: Performance UNCHANGED")
    
    report_lines.append("=" * 100)
    
    # Output report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nComparison report saved to: {output_file}")
    
    return True


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: compare_benchmarks.py <baseline.json> <current.json> [output.txt]")
        print()
        print("Compare two benchmark JSON files and generate a comparison report.")
        print()
        print("Arguments:")
        print("  baseline.json  - Path to baseline benchmark results")
        print("  current.json   - Path to current benchmark results")
        print("  output.txt     - (Optional) Path to save comparison report")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    current_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(baseline_file):
        print(f"Error: Baseline file not found: {baseline_file}")
        sys.exit(1)
    
    if not os.path.exists(current_file):
        print(f"Error: Current file not found: {current_file}")
        sys.exit(1)
    
    success = compare_benchmarks(baseline_file, current_file, output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
