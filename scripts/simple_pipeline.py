#!/usr/bin/env python3
"""
Script: simple_pipeline.py
Description: Run RNAelem motif discovery pipeline with a single command

Original Use Case: examples/use_case_1_simple_pipeline.py
Dependencies Removed: None - uses only stdlib + external elem command

Usage:
    python scripts/simple_pipeline.py --input <fasta_file> --pattern <pattern_file> --output <output_dir>

Example:
    python scripts/simple_pipeline.py --input examples/data/positive.fa --pattern examples/data/simple_pattern.txt --output results/pipeline_output
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import json
import subprocess
import shutil
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "force_overwrite": False,
    "verbose": False,
    "include_intermediate_files": True
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

def collect_output_files(output_dir: Path) -> Dict[str, List[str]]:
    """Collect and categorize output files from the pipeline."""
    files = {
        "models": [],
        "logs": [],
        "data": [],
        "other": []
    }

    if not output_dir.exists():
        return files

    for item in output_dir.rglob("*"):
        if item.is_file():
            relative_path = str(item.relative_to(output_dir))

            if item.suffix == ".model":
                files["models"].append(relative_path)
            elif "log" in item.name.lower() or item.suffix == ".log":
                files["logs"].append(relative_path)
            elif item.suffix in [".fa", ".fasta", ".fq", ".fastq"]:
                files["data"].append(relative_path)
            else:
                files["other"].append(relative_path)

    return files

def validate_input_files(input_file: Path, pattern_file: Path) -> None:
    """Validate input files exist and have reasonable content."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not pattern_file.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_file}")

    # Basic content validation
    with open(input_file, 'r') as f:
        content = f.read()
        if not content.startswith('>'):
            raise ValueError(f"Input file {input_file} does not appear to be a FASTA file")

    with open(pattern_file, 'r') as f:
        pattern_content = f.read().strip()
        if not pattern_content:
            raise ValueError(f"Pattern file {pattern_file} is empty")

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_simple_pipeline(
    input_file: Union[str, Path],
    pattern_file: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run RNAelem motif discovery pipeline with a single command.

    Args:
        input_file: Path to FASTA file with positive sequences
        pattern_file: Path to pattern file (one pattern per line in dot-bracket notation)
        output_dir: Path to output directory
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - output_dir: Path to output directory
            - output_files: Dict categorizing output files
            - metadata: Execution metadata
            - success: Boolean indicating success

    Example:
        >>> result = run_simple_pipeline("input.fa", "pattern.txt", "output/")
        >>> print(f"Pipeline created {len(result['output_files']['models'])} models")
    """
    # Setup
    input_file = Path(input_file)
    pattern_file = Path(pattern_file)
    output_dir = Path(output_dir)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validation
    validate_input_files(input_file, pattern_file)

    # Check output directory
    if output_dir.exists():
        if not config.get("force_overwrite", False):
            raise FileExistsError(f"Output directory {output_dir} exists! Use force_overwrite=True to overwrite.")
        else:
            # Remove existing directory
            shutil.rmtree(output_dir)

    # Run RNAelem pipeline command
    cmd = [
        "elem", "pipeline",
        "-p", str(input_file),
        "-m", str(pattern_file),
        "-o", str(output_dir)
    ]

    if config.get("force_overwrite", False):
        cmd.append("-F")

    result = run_command(cmd, "Running RNAelem motif discovery pipeline")

    if result is None:
        raise RuntimeError("RNAelem pipeline command failed")

    # Collect output files
    output_files = collect_output_files(output_dir)

    return {
        "output_dir": str(output_dir),
        "output_files": output_files,
        "success": True,
        "metadata": {
            "input_file": str(input_file),
            "pattern_file": str(pattern_file),
            "config": config,
            "total_models": len(output_files["models"]),
            "total_files": sum(len(files) for files in output_files.values())
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
                       help='Input FASTA file with positive sequences')
    parser.add_argument('--pattern', '-p', required=True,
                       help='Pattern file (one pattern per line in dot-bracket notation)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory name')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON)')
    parser.add_argument('--force', '-F', action='store_true',
                       help='Force overwrite output directory')
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
        "force_overwrite": args.force,
        "verbose": args.verbose
    })

    # Run
    try:
        result = run_simple_pipeline(
            input_file=args.input,
            pattern_file=args.pattern,
            output_dir=args.output,
            config=config
        )

        # Print summary
        print(f"✅ Pipeline completed successfully!")
        print(f"   Output directory: {result['output_dir']}")
        print(f"   Models created: {result['metadata']['total_models']}")
        print(f"   Total files: {result['metadata']['total_files']}")

        # Show file structure
        files = result["output_files"]
        if files["models"]:
            print(f"   Model files: {', '.join(files['models'][:3])}{'...' if len(files['models']) > 3 else ''}")

        print(f"\nTo scan sequences with trained models:")
        if files["models"]:
            model_path = Path(result['output_dir']) / files["models"][0]
            print(f"   python scripts/motif_scanning.py --input new_sequences.fa --model {model_path}")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == '__main__':
    main()