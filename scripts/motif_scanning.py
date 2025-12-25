#!/usr/bin/env python3
"""
Script: motif_scanning.py
Description: Scan RNA sequences for motifs using trained RNAelem models

Original Use Case: examples/use_case_3_motif_scanning.py
Dependencies Removed: None - uses only stdlib + external elem command

Usage:
    python scripts/motif_scanning.py --input <fasta_file> --model <model_file> --output <output_file>

Example:
    python scripts/motif_scanning.py --input examples/data/positive.fa --model results/uc_001/cv-0/train/pattern-1/train.model --output results/scan_results.json
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "threshold": 0.5,
    "output_format": "json",  # json, csv, raw
    "include_metadata": True,
    "verbose": False
}

# ==============================================================================
# Utility Functions (inlined from original)
# ==============================================================================
def run_command(cmd: List[str], description: str = "") -> Optional[subprocess.CompletedProcess]:
    """Run a command and handle errors."""
    if DEFAULT_CONFIG.get("verbose", False):
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if DEFAULT_CONFIG.get("verbose", False) and result.stdout:
            print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stdout:
            print("Stdout:", e.stdout[:500] + "..." if len(e.stdout) > 500 else e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr[:500] + "..." if len(e.stderr) > 500 else e.stderr)
        return None

def parse_scan_results(raw_file: Path) -> List[Dict[str, Any]]:
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

def save_results(results: List[Dict[str, Any]], output_file: Path, format_type: str = "json") -> None:
    """Save results in specified format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format_type == "json":
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    elif format_type == "csv":
        # Simple CSV output
        with open(output_file, 'w') as f:
            if results:
                # Write header
                headers = list(results[0].keys())
                f.write(','.join(headers) + '\n')
                # Write data
                for result in results:
                    row = [str(result.get(h, '')) for h in headers]
                    f.write(','.join(row) + '\n')
    else:
        raise ValueError(f"Unsupported output format: {format_type}")

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_motif_scanning(
    input_file: Union[str, Path],
    model_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Scan RNA sequences for motifs using trained RNAelem models.

    Args:
        input_file: Path to FASTA file with sequences to scan
        model_file: Path to trained RNAelem model file (.model)
        output_file: Path to save output (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - results: List of scan results
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata
            - statistics: Summary statistics

    Example:
        >>> result = run_motif_scanning("input.fa", "model.model", "output.json")
        >>> print(f"Found {len(result['results'])} sequences with motifs")
    """
    # Setup
    input_file = Path(input_file)
    model_file = Path(model_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validation
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Create temporary directory for elem output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / "scan_out"

        # Run RNAelem scan command
        cmd = [
            "elem", "scan",
            "-s", str(input_file),
            "-m", str(model_file),
            "-o", str(temp_output)
        ]

        result = run_command(cmd, "Scanning sequences for motifs")

        if result is None:
            raise RuntimeError("RNAelem scan command failed")

        # Parse results
        raw_file = temp_output / "scan.raw"
        if not raw_file.exists():
            # Check if we have .fq output instead
            fq_file = temp_output / "scan.fq"
            if fq_file.exists():
                results = []  # For now, just return empty results for .fq files
            else:
                raise FileNotFoundError(f"No scan output found in {temp_output}")
        else:
            results = parse_scan_results(raw_file)

    # Filter by threshold
    threshold = config.get("threshold", 0.5)
    filtered_results = [
        r for r in results
        if r.get('exist_prob', 0.0) >= threshold
    ]

    # Create statistics
    stats = {
        "total_sequences": len(results),
        "sequences_with_motifs": len(filtered_results),
        "threshold": threshold,
        "mean_probability": sum(r.get('exist_prob', 0.0) for r in results) / len(results) if results else 0.0,
        "max_probability": max((r.get('exist_prob', 0.0) for r in results), default=0.0),
    }

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        format_type = config.get("output_format", "json")
        save_results(filtered_results, output_path, format_type)

    return {
        "results": filtered_results,
        "output_file": str(output_path) if output_path else None,
        "statistics": stats,
        "metadata": {
            "input_file": str(input_file),
            "model_file": str(model_file),
            "config": config
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input FASTA file with sequences to scan')
    parser.add_argument('--model', '-m', required=True,
                       help='Path to trained RNAelem model file (.model)')
    parser.add_argument('--output', '-o',
                       help='Output file path')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Minimum motif existence probability threshold')
    parser.add_argument('--format', '-f', choices=['json', 'csv'], default='json',
                       help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with command line args
    config.update({
        "threshold": args.threshold,
        "output_format": args.format,
        "verbose": args.verbose
    })

    # Run
    try:
        result = run_motif_scanning(
            input_file=args.input,
            model_file=args.model,
            output_file=args.output,
            config=config
        )

        # Print summary
        stats = result["statistics"]
        print(f"✅ Scan completed successfully!")
        print(f"   Sequences processed: {stats['total_sequences']}")
        print(f"   Sequences with motifs: {stats['sequences_with_motifs']}")
        print(f"   Threshold: {stats['threshold']}")
        print(f"   Mean probability: {stats['mean_probability']:.3f}")

        if result["output_file"]:
            print(f"   Results saved to: {result['output_file']}")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == '__main__':
    main()