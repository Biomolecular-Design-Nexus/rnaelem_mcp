#!/usr/bin/env python3
"""
RNAelem Use Case 2: Manual Parallel Training
==============================================

This script demonstrates manual step-by-step motif discovery for larger
datasets where you want more control over the process.

This is useful for:
- Understanding the training process
- Running on clusters without Grid Engine
- Custom parameter tuning

Usage:
    python examples/use_case_2_manual_training.py [options]

Example:
    python examples/use_case_2_manual_training.py --input examples/data/positive.fa --pattern-list examples/data/pattern_list --max-patterns 5
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

def count_patterns(pattern_file):
    """Count number of patterns in pattern file."""
    try:
        with open(pattern_file, 'r') as f:
            patterns = [line.strip() for line in f if line.strip()]
        return len(patterns)
    except Exception as e:
        print(f"Error reading pattern file: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description="RNAelem manual step-by-step motif discovery")

    parser.add_argument("--input", "-i",
                       default="examples/data/positive.fa",
                       help="Input FASTA file with positive sequences")
    parser.add_argument("--pattern-list", "-p",
                       default="examples/data/pattern_list",
                       help="Pattern list file")
    parser.add_argument("--output", "-o",
                       default="elem_manual_out",
                       help="Output directory name")
    parser.add_argument("--force", "-F", action="store_true",
                       help="Force overwrite output directory")
    parser.add_argument("--max-patterns", type=int, default=5,
                       help="Maximum number of patterns to train (for demo)")
    parser.add_argument("--num-motifs", type=int, default=3,
                       help="Number of best motifs to select and refine")

    args = parser.parse_args()

    # Check if input files exist
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found!")
        return 1

    if not Path(args.pattern_list).exists():
        print(f"Error: Pattern list file {args.pattern_list} not found!")
        return 1

    # Check if output directory exists and force flag
    if Path(args.output).exists() and not args.force:
        print(f"Error: Output directory {args.output} exists! Use --force to overwrite.")
        return 1

    # Count total patterns
    total_patterns = count_patterns(args.pattern_list)
    if total_patterns == 0:
        print("Error: No patterns found in pattern file!")
        return 1

    patterns_to_train = min(args.max_patterns, total_patterns)

    print("=" * 60)
    print("RNAelem Manual Parallel Training")
    print("=" * 60)
    print(f"Total patterns in file: {total_patterns}")
    print(f"Patterns to train (demo): {patterns_to_train}")

    # Step 1: Initialize training
    print("\nStep 1: Initialize training")
    cmd = [
        "elem", "init",
        "-p", args.input,
        "-m", args.pattern_list,
        "-o", args.output
    ]
    if args.force:
        cmd.append("-F")

    result = run_command(cmd, "Initializing training")
    if result is None:
        print("Initialization failed!")
        return 1

    # Step 2: Train specific patterns
    print(f"\nStep 2: Train patterns 1-{patterns_to_train}")
    for pattern_idx in range(1, patterns_to_train + 1):
        print(f"\nTraining pattern {pattern_idx}/{patterns_to_train}")
        cmd = [
            "elem", "train",
            "--elem-out", args.output,
            "--pattern-index", str(pattern_idx)
        ]

        result = run_command(cmd, f"Training pattern {pattern_idx}")
        if result is None:
            print(f"Training pattern {pattern_idx} failed!")
            return 1

    # Step 3: Select best motifs
    print(f"\nStep 3: Select top {args.num_motifs} motifs")
    cmd = [
        "elem", "select",
        "--elem-out", args.output,
        "--num-motifs", str(args.num_motifs)
    ]

    result = run_command(cmd, f"Selecting top {args.num_motifs} motifs")
    if result is None:
        print("Selection failed!")
        return 1

    # Step 4: Refine selected motifs
    print(f"\nStep 4: Refine selected motifs")
    for motif_idx in range(1, args.num_motifs + 1):
        print(f"\nRefining motif {motif_idx}/{args.num_motifs}")
        cmd = [
            "elem", "refine",
            "--elem-out", args.output,
            "--pattern-index", str(motif_idx)
        ]

        result = run_command(cmd, f"Refining motif {motif_idx}")
        if result is None:
            print(f"Refining motif {motif_idx} failed!")
            return 1

    print("\n" + "=" * 60)
    print("Manual training completed successfully!")
    print("=" * 60)

    # Check output files
    output_dir = Path(args.output)
    if output_dir.exists():
        print(f"\nTrained models:")
        for i in range(1, args.num_motifs + 1):
            model_dir = output_dir / f"model-{i}"
            if model_dir.exists():
                print(f"  Model {i}: {model_dir}")
                model_file = model_dir / "train.model"
                if model_file.exists():
                    print(f"    Model file: {model_file}")

    print(f"\nTo scan new sequences with the best model:")
    print(f"elem scan -s new_sequences.fa -m {args.output}/model-1/train.model")

    return 0

if __name__ == "__main__":
    sys.exit(main())