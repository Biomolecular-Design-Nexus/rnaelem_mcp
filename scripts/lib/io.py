"""Shared I/O functions for RNAelem MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
import json
import subprocess
from pathlib import Path
from typing import Union, Any, Dict, Optional, List


def load_json(file_path: Union[str, Path]) -> dict:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)


def save_json(data: dict, file_path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_csv_simple(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """Save data to CSV file (simple implementation)."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not data:
        with open(file_path, 'w') as f:
            f.write("")
        return

    # Write header
    headers = list(data[0].keys())
    with open(file_path, 'w') as f:
        f.write(','.join(headers) + '\n')
        # Write data
        for row in data:
            values = [str(row.get(h, '')) for h in headers]
            f.write(','.join(values) + '\n')


def validate_fasta_file(file_path: Union[str, Path]) -> bool:
    """Validate that file is in FASTA format."""
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            return first_line.startswith('>')
    except Exception:
        return False


def count_fasta_sequences(file_path: Union[str, Path]) -> int:
    """Count number of sequences in FASTA file."""
    count = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
    except Exception:
        pass
    return count