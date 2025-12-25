"""Shared utility functions for RNAelem MCP scripts."""
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any


def run_command(
    cmd: List[str],
    description: str = "",
    verbose: bool = False,
    capture_output: bool = True
) -> Optional[subprocess.CompletedProcess]:
    """Run a command and handle errors."""
    if verbose:
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=capture_output, text=True)
        if verbose and result.stdout:
            output = result.stdout
            print("Output:", output[:200] + "..." if len(output) > 200 else output)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stdout:
            print("Stdout:", e.stdout[:500] + "..." if len(e.stdout) > 500 else e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr[:500] + "..." if len(e.stderr) > 500 else e.stderr)
        return None


def validate_required_files(*file_paths: Path) -> None:
    """Validate that all required files exist."""
    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")


def merge_config(default_config: Dict[str, Any], user_config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Merge configuration dictionaries with precedence: kwargs > user_config > default_config."""
    config = default_config.copy()
    if user_config:
        config.update(user_config)
    config.update(kwargs)
    return config


def collect_files_by_pattern(directory: Path, patterns: List[str]) -> Dict[str, List[str]]:
    """Collect files in directory matching patterns."""
    result = {pattern: [] for pattern in patterns}

    if not directory.exists():
        return result

    for item in directory.rglob("*"):
        if item.is_file():
            relative_path = str(item.relative_to(directory))
            for pattern in patterns:
                if pattern in item.suffix or pattern in item.name.lower():
                    result[pattern].append(relative_path)

    return result


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def get_directory_info(directory: Path) -> Dict[str, Any]:
    """Get basic information about a directory."""
    if not directory.exists():
        return {"exists": False}

    files = list(directory.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    dir_count = sum(1 for f in files if f.is_dir())
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    return {
        "exists": True,
        "file_count": file_count,
        "dir_count": dir_count,
        "total_size": total_size,
        "total_size_formatted": format_file_size(total_size)
    }