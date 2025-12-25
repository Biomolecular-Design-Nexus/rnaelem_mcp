#!/usr/bin/env python3
"""
MCP Server for RNAelem

Provides both synchronous and asynchronous (submit) APIs for RNA motif discovery and analysis.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("RNAelem")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def scan_motifs(
    input_file: str,
    model_file: str,
    threshold: float = 0.5,
    output_format: str = "json",
    output_file: Optional[str] = None
) -> dict:
    """
    Scan RNA sequences for motifs using trained RNAelem models.

    Fast operation suitable for small to medium sequence files (~2-5 minutes).
    For large-scale scanning, use submit_motif_scanning.

    Args:
        input_file: Path to FASTA file with RNA sequences
        model_file: Path to trained RNAelem model (.model file)
        threshold: Motif existence probability threshold (0.0-1.0, default: 0.5)
        output_format: Output format - "json" or "csv" (default: "json")
        output_file: Optional path to save results

    Returns:
        Dictionary with scan results, statistics, and output_file path

    Example:
        scan_motifs("examples/data/positive.fa", "results/uc_001/cv-0/train/pattern-1/train.model", threshold=0.7)
    """
    from motif_scanning import run_motif_scanning

    try:
        result = run_motif_scanning(
            input_file=input_file,
            model_file=model_file,
            output_file=output_file,
            threshold=threshold,
            output_format=output_format,
            include_metadata=True
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Motif scanning failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def analyze_sequences_ml(
    positive_file: str,
    negative_file: Optional[str] = None,
    model_type: str = "random_forest",
    test_size: float = 0.3,
    n_estimators: int = 100,
    output_dir: Optional[str] = None
) -> dict:
    """
    Machine learning analysis of RNA sequences using RNAelem features.

    Fast operation suitable for small to medium datasets (~3-8 minutes).
    For large-scale ML training, use submit_ml_analysis.

    Args:
        positive_file: Path to FASTA file with positive RNA sequences
        negative_file: Path to negative sequences (optional, will generate if not provided)
        model_type: ML algorithm - "random_forest" or "gradient_boost" (default: "random_forest")
        test_size: Train/test split ratio (0.1-0.5, default: 0.3)
        n_estimators: Number of trees in ensemble (default: 100)
        output_dir: Optional directory to save model and results

    Returns:
        Dictionary with model metrics, feature importance, and output files

    Example:
        analyze_sequences_ml("examples/data/positive.fa", model_type="gradient_boost", n_estimators=200)
    """
    from ml_analysis import run_ml_analysis

    try:
        result = run_ml_analysis(
            positive_file=positive_file,
            negative_file=negative_file,
            output_dir=output_dir,
            model_type=model_type,
            test_size=test_size,
            n_estimators=n_estimators
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"ML analysis failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_motif_discovery(
    input_file: str,
    pattern_file: str,
    output_dir: Optional[str] = None,
    force_overwrite: bool = False,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit RNAelem motif discovery pipeline for background processing.

    This operation typically takes 30-60+ minutes. Use get_job_status() to monitor
    progress and get_job_result() to retrieve results when completed.

    Args:
        input_file: Path to FASTA file with positive RNA sequences
        pattern_file: Path to pattern file (dot-bracket notation)
        output_dir: Directory to save trained models and results
        force_overwrite: Overwrite existing output directory (default: False)
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs

    Example:
        submit_motif_discovery("examples/data/positive.fa", "examples/data/simple_pattern.txt", "results/pipeline")
    """
    script_path = str(SCRIPTS_DIR / "simple_pipeline.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "pattern": pattern_file,
            "output": output_dir,
            "force_overwrite": force_overwrite,
            "verbose": True
        },
        job_name=job_name or "motif_discovery"
    )

@mcp.tool()
def submit_motif_scanning(
    input_file: str,
    model_file: str,
    threshold: float = 0.5,
    output_format: str = "json",
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit large-scale motif scanning for background processing.

    Use this for batch scanning of many sequences or when processing very large files.
    For quick scans of small files, use scan_motifs instead.

    Args:
        input_file: Path to FASTA file with RNA sequences
        model_file: Path to trained RNAelem model
        threshold: Motif existence probability threshold (default: 0.5)
        output_format: Output format - "json" or "csv" (default: "json")
        output_dir: Directory to save outputs
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the scanning job

    Example:
        submit_motif_scanning("large_sequences.fa", "model.model", threshold=0.8, output_format="csv")
    """
    script_path = str(SCRIPTS_DIR / "motif_scanning.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "model": model_file,
            "threshold": threshold,
            "output_format": output_format,
            "output": output_dir,
            "verbose": True
        },
        job_name=job_name or "motif_scanning"
    )

@mcp.tool()
def submit_ml_analysis(
    positive_file: str,
    negative_file: Optional[str] = None,
    model_type: str = "random_forest",
    test_size: float = 0.3,
    n_estimators: int = 100,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit large-scale machine learning analysis for background processing.

    Use this for datasets with many features or when training complex models.
    For quick analysis of small datasets, use analyze_sequences_ml instead.

    Args:
        positive_file: Path to FASTA file with positive sequences
        negative_file: Path to negative sequences (optional)
        model_type: ML algorithm - "random_forest" or "gradient_boost"
        test_size: Train/test split ratio (default: 0.3)
        n_estimators: Number of trees in ensemble (default: 100)
        output_dir: Directory to save outputs
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the ML analysis job

    Example:
        submit_ml_analysis("large_positive.fa", model_type="gradient_boost", n_estimators=500)
    """
    script_path = str(SCRIPTS_DIR / "ml_analysis.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "positive": positive_file,
            "negative": negative_file,
            "model_type": model_type,
            "test_size": test_size,
            "n_estimators": n_estimators,
            "output": output_dir,
            "verbose": True
        },
        job_name=job_name or "ml_analysis"
    )

# ==============================================================================
# Batch Processing Tools
# ==============================================================================

@mcp.tool()
def submit_batch_motif_scanning(
    input_files: List[str],
    model_file: str,
    threshold: float = 0.5,
    output_format: str = "json",
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch motif scanning for multiple input files.

    Processes multiple FASTA files with the same model in a single job.
    Suitable for:
    - Processing many sequence files at once
    - Large-scale motif analysis across datasets
    - Parallel processing of independent sequence files

    Args:
        input_files: List of FASTA file paths to scan
        model_file: Path to trained RNAelem model
        threshold: Motif existence probability threshold (default: 0.5)
        output_format: Output format - "json" or "csv" (default: "json")
        output_dir: Directory to save all outputs
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch job

    Example:
        submit_batch_motif_scanning(["seq1.fa", "seq2.fa", "seq3.fa"], "model.model", threshold=0.7)
    """
    # For batch processing, we'll run the motif scanning script multiple times
    # This could be enhanced to a dedicated batch script in the future
    script_path = str(SCRIPTS_DIR / "motif_scanning.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input_files": input_files,  # The job manager will handle the list appropriately
            "model": model_file,
            "threshold": threshold,
            "output_format": output_format,
            "output": output_dir,
            "verbose": True
        },
        job_name=job_name or f"batch_scanning_{len(input_files)}_files"
    )

@mcp.tool()
def submit_batch_ml_analysis(
    positive_files: List[str],
    negative_files: Optional[List[str]] = None,
    model_type: str = "random_forest",
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch ML analysis for multiple datasets.

    Trains separate models for each positive dataset. Suitable for:
    - Comparing ML performance across multiple datasets
    - Large-scale comparative analysis
    - Parallel processing of independent datasets

    Args:
        positive_files: List of positive sequence FASTA files
        negative_files: Optional list of negative sequence files (same order as positive)
        model_type: ML algorithm for all analyses (default: "random_forest")
        output_dir: Directory to save all outputs
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch analysis

    Example:
        submit_batch_ml_analysis(["dataset1_pos.fa", "dataset2_pos.fa"], ["dataset1_neg.fa", "dataset2_neg.fa"])
    """
    script_path = str(SCRIPTS_DIR / "ml_analysis.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "positive_files": positive_files,
            "negative_files": negative_files,
            "model_type": model_type,
            "output": output_dir,
            "verbose": True
        },
        job_name=job_name or f"batch_ml_{len(positive_files)}_datasets"
    )

# ==============================================================================
# Validation and Utility Tools
# ==============================================================================

@mcp.tool()
def validate_input_files(
    input_file: str,
    file_type: str = "fasta"
) -> dict:
    """
    Validate input file format and content for RNAelem tools.

    Args:
        input_file: Path to file to validate
        file_type: Expected file type - "fasta", "pattern", or "model"

    Returns:
        Dictionary with validation results and file statistics

    Example:
        validate_input_files("sequences.fa", "fasta")
    """
    from pathlib import Path

    try:
        file_path = Path(input_file)

        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {input_file}"}

        result = {
            "status": "success",
            "file_path": str(file_path.resolve()),
            "file_size": file_path.stat().st_size,
            "file_type": file_type
        }

        if file_type == "fasta":
            # Basic FASTA validation
            with open(file_path) as f:
                content = f.read()
                seq_count = content.count('>')
                result.update({
                    "sequence_count": seq_count,
                    "total_chars": len(content),
                    "has_sequences": seq_count > 0
                })

        elif file_type == "pattern":
            # Pattern file validation
            with open(file_path) as f:
                content = f.read().strip()
                result.update({
                    "pattern_content": content,
                    "is_valid_pattern": set(content).issubset(set("().&")),
                    "pattern_length": len(content)
                })

        elif file_type == "model":
            # Model file validation (basic existence check)
            result.update({
                "is_model_file": file_path.suffix == ".model",
                "model_exists": True
            })

        return result

    except Exception as e:
        return {"status": "error", "error": f"Validation failed: {e}"}

@mcp.tool()
def get_example_data() -> dict:
    """
    Get information about available example datasets for testing.

    Returns:
        Dictionary with example files and their descriptions
    """
    examples_dir = MCP_ROOT / "examples" / "data"

    if not examples_dir.exists():
        return {"status": "error", "error": "Examples directory not found"}

    example_files = {}
    for file_path in examples_dir.rglob("*"):
        if file_path.is_file():
            rel_path = str(file_path.relative_to(MCP_ROOT))
            example_files[rel_path] = {
                "size": file_path.stat().st_size,
                "description": _get_file_description(file_path)
            }

    return {
        "status": "success",
        "examples_directory": str(examples_dir),
        "available_files": example_files
    }

def _get_file_description(file_path: Path) -> str:
    """Get description of example file based on name and type."""
    name = file_path.name.lower()
    suffix = file_path.suffix.lower()

    if "positive" in name:
        return "Positive RNA sequences for training"
    elif "negative" in name:
        return "Negative RNA sequences for training"
    elif "pattern" in name:
        return "RNA motif pattern in dot-bracket notation"
    elif suffix == ".fa" or suffix == ".fasta":
        return "RNA sequences in FASTA format"
    elif suffix == ".model":
        return "Trained RNAelem model file"
    else:
        return "Example data file"

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()