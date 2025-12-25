#!/usr/bin/env python3
"""
RNAelem Use Case 4: Motif Visualization
========================================

This script demonstrates how to generate visualizations of discovered motifs,
including sequence logos and RNA secondary structure diagrams.

Features:
- Generate sequence profile logos (PNG/SVG)
- Create RNA secondary structure diagrams (PNG/EPS)
- Batch visualization of multiple motifs

Usage:
    python examples/use_case_4_visualization.py [options]

Example:
    python examples/use_case_4_visualization.py --model-dir elem_simple_out --output viz_out
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

def find_model_directories(base_dir):
    """Find all model directories in the base directory."""
    model_dirs = []
    base_path = Path(base_dir)
    if base_path.exists():
        for item in base_path.glob("model-*"):
            if item.is_dir():
                model_file = item / "train.model"
                if model_file.exists():
                    model_dirs.append(item)
    return model_dirs

def visualize_motif(model_dir, output_dir, base_threshold=1.5):
    """Visualize a single motif model."""
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_dir.name
    print(f"\nVisualizing {model_name}...")

    # Output paths
    rss_eps = output_dir / f"{model_name}_rss.eps"
    rss_png = output_dir / f"{model_name}_rss.png"
    prf_svg = output_dir / f"{model_name}_prf.svg"
    prf_png = output_dir / f"{model_name}_prf.png"

    # Generate visualizations using draw_motif.py
    cmd = [
        "draw_motif.py",
        str(model_dir),
        str(rss_eps),
        str(prf_svg),
        str(base_threshold)
    ]

    result = run_command(cmd, f"Generating motif diagrams for {model_name}")
    if result is None:
        print(f"Failed to generate diagrams for {model_name}")
        return False

    # Convert SVG to PNG for sequence logo
    if prf_svg.exists():
        cmd = [
            "rsvg-convert",
            "-f", "png",
            "--background-color=white",
            "-o", str(prf_png),
            str(prf_svg)
        ]
        result = run_command(cmd, f"Converting SVG to PNG for {model_name}")

    # Convert EPS to PNG for secondary structure
    if rss_eps.exists():
        cmd = [
            "convert",
            "-density", "320",
            str(rss_eps),
            str(rss_png)
        ]
        result = run_command(cmd, f"Converting EPS to PNG for {model_name}")

    return True

def main():
    parser = argparse.ArgumentParser(
        description="RNAelem motif visualization")

    parser.add_argument("--model-dir", "-d",
                       help="Directory containing RNAelem model directories")
    parser.add_argument("--single-model", "-m",
                       help="Path to single model directory to visualize")
    parser.add_argument("--output", "-o",
                       default="viz_out",
                       help="Output directory for visualizations")
    parser.add_argument("--base-threshold", "-t", type=float, default=1.5,
                       help="Base threshold for visualization (lower = more bases shown)")
    parser.add_argument("--force", "-F", action="store_true",
                       help="Force overwrite output directory")

    args = parser.parse_args()

    if not args.model_dir and not args.single_model:
        print("Error: Must specify either --model-dir or --single-model!")
        return 1

    # Check if output directory exists
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"Error: Output directory {args.output} exists! Use --force to overwrite.")
        return 1

    print("=" * 60)
    print("RNAelem Motif Visualization")
    print("=" * 60)

    # Check if required tools are available
    tools_available = True
    for tool in ["draw_motif.py", "rsvg-convert", "convert"]:
        result = subprocess.run(["which", tool], capture_output=True)
        if result.returncode != 0:
            print(f"Warning: {tool} not found in PATH. Some visualizations may fail.")
            if tool == "draw_motif.py":
                tools_available = False

    if not tools_available:
        print("Error: draw_motif.py is required but not found!")
        return 1

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total_count = 0

    if args.single_model:
        # Visualize single model
        model_path = Path(args.single_model)
        if not model_path.exists():
            print(f"Error: Model directory {args.single_model} not found!")
            return 1

        total_count = 1
        if visualize_motif(model_path, output_path, args.base_threshold):
            success_count = 1

    else:
        # Visualize all models in directory
        model_dirs = find_model_directories(args.model_dir)
        if not model_dirs:
            print(f"Error: No model directories found in {args.model_dir}")
            return 1

        total_count = len(model_dirs)
        print(f"Found {total_count} model directories")

        for model_dir in model_dirs:
            if visualize_motif(model_dir, output_path, args.base_threshold):
                success_count += 1

    print("\n" + "=" * 60)
    print("Visualization completed!")
    print("=" * 60)
    print(f"Successfully visualized: {success_count}/{total_count} models")

    # List generated files
    if output_path.exists():
        print(f"\nGenerated files in {args.output}:")
        for file in sorted(output_path.glob("*")):
            if file.is_file():
                print(f"  {file.name}")

    print("\nVisualization types:")
    print("  *_rss.eps/png: RNA secondary structure diagrams")
    print("  *_prf.svg/png: Sequence profile logos")

    return 0

if __name__ == "__main__":
    sys.exit(main())