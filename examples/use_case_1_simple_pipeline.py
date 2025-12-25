#!/usr/bin/env python3
"""
RNAelem Use Case 1: Simple Motif Discovery Pipeline
====================================================

This script demonstrates the basic RNAelem pipeline for RNA sequence-structure
motif discovery using the toy example from the documentation.

Based on the tRNA example, this finds a motif from RNA sequences with a
simple hairpin loop pattern.

Usage:
    python examples/use_case_1_simple_pipeline.py [options]

Example:
    python examples/use_case_1_simple_pipeline.py --input examples/data/positive.fa --pattern examples/data/simple_pattern.txt
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser(
        description="RNAelem simple motif discovery pipeline")

    parser.add_argument("--input", "-i",
                       default="examples/data/positive.fa",
                       help="Input FASTA file with positive sequences")
    parser.add_argument("--pattern", "-p",
                       default="examples/data/simple_pattern.txt",
                       help="Pattern file (one pattern per line)")
    parser.add_argument("--output", "-o",
                       default="elem_simple_out",
                       help="Output directory name")
    parser.add_argument("--force", "-F", action="store_true",
                       help="Force overwrite output directory")

    args = parser.parse_args()

    # Check if input files exist
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found!")
        return 1

    if not Path(args.pattern).exists():
        print(f"Error: Pattern file {args.pattern} not found!")
        return 1

    # Check if output directory exists and force flag
    if Path(args.output).exists() and not args.force:
        print(f"Error: Output directory {args.output} exists! Use --force to overwrite.")
        return 1

    print("=" * 60)
    print("RNAelem Simple Motif Discovery Pipeline")
    print("=" * 60)

    # Run RNAelem pipeline
    cmd = [
        "elem", "pipeline",
        "-p", args.input,
        "-m", args.pattern,
        "-o", args.output
    ]

    if args.force:
        cmd.append("-F")

    result = run_command(cmd, "Running RNAelem pipeline")

    if result is None:
        print("Pipeline failed!")
        return 1

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

    # Check output files
    output_dir = Path(args.output)
    if output_dir.exists():
        print(f"\nOutput files in {args.output}:")
        for item in output_dir.rglob("*"):
            if item.is_file():
                print(f"  {item}")

    print(f"\nTo scan new sequences with the trained model:")
    print(f"elem scan -s new_sequences.fa -m {args.output}/model-1/train.model")

    return 0

if __name__ == "__main__":
    sys.exit(main())