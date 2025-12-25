# RNAelem MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported
2. **Self-Contained**: Functions inlined where possible
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts

| Script | Description | Repo Dependent | Config |
|--------|-------------|----------------|--------|
| `simple_pipeline.py` | Run RNAelem motif discovery pipeline | No | `configs/simple_pipeline_config.json` |
| `motif_scanning.py` | Scan sequences for motifs with trained models | No | `configs/motif_scanning_config.json` |
| `ml_analysis.py` | Machine learning analysis of RNA sequences | No | `configs/ml_analysis_config.json` |

## Dependencies Summary

All scripts use only:
- **Python Standard Library**: os, sys, subprocess, argparse, pathlib, tempfile, json
- **External Tools**: RNAelem suite (elem, dishuffle.py, kmer-psp.py)
- **ML Script Only**: numpy, pandas, sklearn (all common packages)

**No repo dependencies**: All scripts are independent of the original RNAelem repository code.

## Usage

```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env  # or: conda activate ./env

# Run motif discovery pipeline
python scripts/simple_pipeline.py --input examples/data/positive.fa --pattern examples/data/simple_pattern.txt --output results/pipeline_output

# Scan sequences for motifs (requires trained model)
python scripts/motif_scanning.py --input examples/data/positive.fa --model results/pipeline_output/cv-0/train/pattern-1/train.model --output results/scan_results.json

# Machine learning analysis
python scripts/ml_analysis.py --positive examples/data/positive.fa --output results/ml_analysis

# With custom config
python scripts/simple_pipeline.py --input FILE --output DIR --config configs/simple_pipeline_config.json
```

## Configuration Files

Each script has a corresponding JSON config file in `configs/`:

- `configs/simple_pipeline_config.json` - Pipeline execution settings
- `configs/motif_scanning_config.json` - Scanning thresholds and output formats
- `configs/ml_analysis_config.json` - ML model parameters and feature settings

## Shared Library

Common functions are in `scripts/lib/`:
- `io.py`: File loading/saving, format validation
- `utils.py`: Command execution, configuration merging, file utilities

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped:

```python
from scripts.simple_pipeline import run_simple_pipeline
from scripts.motif_scanning import run_motif_scanning
from scripts.ml_analysis import run_ml_analysis

# In MCP tool:
@mcp.tool()
def rnaelem_pipeline(input_file: str, pattern_file: str, output_dir: str):
    """Run RNAelem motif discovery pipeline."""
    return run_simple_pipeline(input_file, pattern_file, output_dir)

@mcp.tool()
def rnaelem_scan(input_file: str, model_file: str, output_file: str = None):
    """Scan RNA sequences for motifs using trained model."""
    return run_motif_scanning(input_file, model_file, output_file)

@mcp.tool()
def rnaelem_ml_analysis(positive_file: str, output_dir: str):
    """Perform ML analysis of RNA sequences."""
    return run_ml_analysis(positive_file, output_dir)
```

## Testing

All scripts include `--help` and basic validation:

```bash
# Test script help
python scripts/simple_pipeline.py --help
python scripts/motif_scanning.py --help
python scripts/ml_analysis.py --help

# Test with example data (pipeline will take ~60+ minutes)
python scripts/simple_pipeline.py --input examples/data/positive.fa --pattern examples/data/simple_pattern.txt --output /tmp/test_output
```

## Status from Step 4

Based on Step 4 execution results:
- ✅ **simple_pipeline.py**: Working (equivalent to UC-001)
- ✅ **motif_scanning.py**: Working (equivalent to UC-003)
- ⚠️ **ml_analysis.py**: Simplified from UC-005 (original had design issues)

All scripts are extracted from working use cases and simplified to remove dependencies while maintaining core functionality.