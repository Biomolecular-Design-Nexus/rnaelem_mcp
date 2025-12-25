#!/usr/bin/env python3
"""
RNAelem Use Case 3: Motif Scanning
===================================

This script demonstrates how to use trained RNAelem models to scan new
RNA sequences for motifs. This is useful for:

- Applying trained models to new datasets
- Batch processing multiple sequence files
- Analyzing motif presence in target sequences

Usage:
    python examples/use_case_3_motif_scanning.py [options]

Example:
    python examples/use_case_3_motif_scanning.py --sequences examples/data/positive.fa --model elem_simple_out/model-1/train.model
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import tempfile

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print("Output:", result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return None

def parse_scan_results(raw_file):
    """Parse RNAelem scan results and extract key information."""
    results = []
    try:
        with open(raw_file, 'r') as f:
            current_result = {}
            for line in f:
                line = line.strip()
                if line.startswith('id:'):
                    if current_result:
                        results.append(current_result)
                    current_result = {'id': line.split(':', 1)[1].strip()}
                elif line.startswith('motif region:'):
                    current_result['motif_region'] = line.split(':', 1)[1].strip()
                elif line.startswith('exist prob:'):
                    current_result['exist_prob'] = float(line.split(':', 1)[1].strip())
                elif line.startswith('seq:'):
                    current_result['sequence'] = line.split(':', 1)[1].strip()
                elif line.startswith('mot:'):
                    current_result['motif_alignment'] = line.split(':', 1)[1].strip()

            # Don't forget the last result
            if current_result:
                results.append(current_result)

    except Exception as e:
        print(f"Error parsing results: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="RNAelem motif scanning")

    parser.add_argument("--sequences", "-s",
                       default="examples/data/positive.fa",
                       help="Input FASTA file with sequences to scan")
    parser.add_argument("--model", "-m",
                       help="Path to trained RNAelem model file (.model)")
    parser.add_argument("--model-dir", "-M",
                       help="Path to directory with multiple models (alternative to --model)")
    parser.add_argument("--output", "-o",
                       default="scan_out",
                       help="Output directory name")
    parser.add_argument("--force", "-F", action="store_true",
                       help="Force overwrite output directory")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Minimum motif existence probability threshold")

    args = parser.parse_args()

    # Check if input files exist
    if not Path(args.sequences).exists():
        print(f"Error: Sequences file {args.sequences} not found!")
        return 1

    # Check model specification
    if not args.model and not args.model_dir:
        print("Error: Must specify either --model or --model-dir!")
        return 1

    if args.model and not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found!")
        return 1

    if args.model_dir and not Path(args.model_dir).exists():
        print(f"Error: Model directory {args.model_dir} not found!")
        return 1

    # Check if output directory exists and force flag
    if Path(args.output).exists() and not args.force:
        print(f"Error: Output directory {args.output} exists! Use --force to overwrite.")
        return 1

    print("=" * 60)
    print("RNAelem Motif Scanning")
    print("=" * 60)

    # Run RNAelem scan
    cmd = ["elem", "scan", "-s", args.sequences, "-o", args.output]

    if args.model:
        cmd.extend(["-m", args.model])
        scan_mode = "single model"
    else:
        cmd.extend(["-M", args.model_dir])
        scan_mode = "multiple models"

    if args.force:
        cmd.append("-F")

    print(f"Scanning mode: {scan_mode}")
    result = run_command(cmd, "Scanning sequences for motifs")

    if result is None:
        print("Scanning failed!")
        return 1

    print("\n" + "=" * 60)
    print("Scanning completed successfully!")
    print("=" * 60)

    # Parse and display results
    output_dir = Path(args.output)
    if args.model:
        # Single model results
        raw_file = output_dir / "scan.raw"
        if raw_file.exists():
            results = parse_scan_results(raw_file)
            print(f"\nResults for {len(results)} sequences:")
            print("-" * 40)
            significant_count = 0
            for result in results:
                prob = result.get('exist_prob', 0.0)
                if prob >= args.threshold:
                    significant_count += 1
                    print(f"ID: {result.get('id', 'Unknown')}")
                    print(f"  Motif region: {result.get('motif_region', 'N/A')}")
                    print(f"  Exist prob: {prob:.3f}")
                    print(f"  Motif: {result.get('motif_alignment', 'N/A')}")
                    print()

            print(f"Sequences with motif (prob >= {args.threshold}): {significant_count}/{len(results)}")
    else:
        # Multiple model results
        print("\nResults from multiple models:")
        for model_dir in output_dir.glob("model-*"):
            if model_dir.is_dir():
                raw_file = model_dir / "scan.raw"
                if raw_file.exists():
                    results = parse_scan_results(raw_file)
                    significant_count = sum(1 for r in results
                                          if r.get('exist_prob', 0.0) >= args.threshold)
                    print(f"  {model_dir.name}: {significant_count}/{len(results)} sequences above threshold")

    return 0

if __name__ == "__main__":
    sys.exit(main())