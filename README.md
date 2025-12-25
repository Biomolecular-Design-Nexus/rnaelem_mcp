# RNAelem MCP

> Model Completions Protocol (MCP) server for RNAelem - RNA motif discovery, analysis, and machine learning toolkit

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

RNAelem MCP provides both command-line scripts and MCP server integration for RNA motif discovery and analysis. Built on the robust RNAelem C++ toolkit, it offers complete workflows from basic motif discovery to advanced machine learning analysis of RNA sequences.

### Features
- **Motif Discovery**: Complete RNAelem pipeline for discovering RNA structural motifs
- **Motif Scanning**: Apply trained models to scan new sequences for known motifs
- **Machine Learning**: Enhanced motif analysis using ensemble methods (Random Forest, Gradient Boosting)
- **Batch Processing**: Handle multiple files and large datasets efficiently
- **Async Operations**: Long-running tasks run in background with progress tracking
- **Flexible APIs**: Both synchronous (fast) and asynchronous (submit) operation modes

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment
├── src/
│   └── server.py           # MCP server
│   └── jobs/               # Job management system
├── scripts/
│   ├── simple_pipeline.py  # RNAelem motif discovery pipeline
│   ├── motif_scanning.py   # Motif scanning with trained models
│   ├── ml_analysis.py      # Machine learning analysis
│   └── lib/                # Shared utilities
├── examples/
│   └── data/               # Demo data (tRNA sequences, patterns)
├── configs/                # Configuration files
├── repo/                   # Original RNAelem repository
└── reports/                # Setup and analysis reports
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- C++ compiler with C++14 support
- System packages: pkg-config, freetype2-dev

### Create Environment

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rnaelem_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install Python dependencies
pip install numpy scipy pandas scikit-learn loguru click tqdm

# Install MCP dependencies
pip install fastmcp loguru --ignore-installed

# Build RNAelem C++ binaries (already done during setup)
# cd repo/RNAelem && ./waf configure --prefix=$HOME/local && ./waf build
```

### Verify Installation

```bash
# Activate environment
mamba activate ./env

# Test Python imports
python -c "import numpy; import scipy; import pandas; import sklearn; import fastmcp; print('✅ All imports successful')"

# Test RNAelem binaries
elem --help
RNAelem --help
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/simple_pipeline.py` | Complete RNAelem motif discovery pipeline | See below |
| `scripts/motif_scanning.py` | Scan sequences using trained models | See below |
| `scripts/ml_analysis.py` | Machine learning enhanced analysis | See below |

### Script Examples

#### Simple Pipeline (Motif Discovery)

```bash
# Activate environment
mamba activate ./env

# Run complete RNAelem pipeline
python scripts/simple_pipeline.py \
  --input examples/data/positive.fa \
  --pattern examples/data/simple_pattern.txt \
  --output results/pipeline_demo \
  --verbose
```

**Parameters:**
- `--input, -i`: FASTA file with positive RNA sequences (required)
- `--pattern, -p`: Pattern file in dot-bracket notation (required)
- `--output, -o`: Output directory for models and results (default: auto-generated)
- `--force`: Overwrite existing output directory
- `--verbose`: Enable detailed progress output

#### Motif Scanning

```bash
# Scan sequences with a trained model
python scripts/motif_scanning.py \
  --input examples/data/positive.fa \
  --model results/pipeline_demo/model-1/train.model \
  --threshold 0.7 \
  --format json \
  --output results/scanning_demo.json
```

**Parameters:**
- `--input, -i`: FASTA file with RNA sequences to scan (required)
- `--model, -m`: Trained RNAelem model file (.model) (required)
- `--threshold, -t`: Motif existence probability threshold (default: 0.5)
- `--format, -f`: Output format - "json" or "csv" (default: json)
- `--output, -o`: Output file path (default: auto-generated)

#### Machine Learning Analysis

```bash
# ML analysis with automatic negative sequence generation
python scripts/ml_analysis.py \
  --positive examples/data/positive.fa \
  --model-type gradient_boost \
  --n-estimators 200 \
  --output results/ml_demo
```

**Parameters:**
- `--positive`: FASTA file with positive RNA sequences (required)
- `--negative`: FASTA file with negative sequences (optional, generated if not provided)
- `--model-type`: ML algorithm - "random_forest" or "gradient_boost" (default: random_forest)
- `--test-size`: Train/test split ratio (default: 0.3)
- `--n-estimators`: Number of trees in ensemble (default: 100)
- `--output`: Output directory for model and results

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name RNAelem
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add RNAelem -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "RNAelem": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rnaelem_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rnaelem_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from RNAelem?
```

#### Basic Motif Discovery
```
Use submit_motif_discovery with input file @examples/data/positive.fa and pattern file @examples/data/simple_pattern.txt
```

#### Quick Motif Scanning (requires trained model)
```
Use scan_motifs to scan @examples/data/positive.fa with a trained model, threshold 0.8
```

#### Machine Learning Analysis
```
Run analyze_sequences_ml on @examples/data/positive.fa using gradient_boost with 200 estimators
```

#### Long-Running Tasks (Submit API)
```
Submit motif discovery for @examples/data/positive.fa with pattern @examples/data/simple_pattern.txt
Then check the job status
```

#### Batch Processing
```
Process these files in batch using submit_batch_motif_scanning:
- @examples/data/positive.fa
- @results/new_sequences.fa
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/positive.fa` | Reference the demo RNA sequences |
| `@examples/data/simple_pattern.txt` | Reference the simple hairpin pattern |
| `@configs/ml_analysis_config.json` | Reference ML configuration |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "RNAelem": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rnaelem_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rnaelem_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Use scan_motifs with file examples/data/positive.fa and threshold 0.7
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `scan_motifs` | Scan RNA sequences for motifs | `input_file`, `model_file`, `threshold`, `output_format`, `output_file` |
| `analyze_sequences_ml` | ML analysis of RNA sequences | `positive_file`, `negative_file`, `model_type`, `test_size`, `n_estimators`, `output_dir` |
| `validate_input_files` | Validate file format and content | `input_file`, `file_type` |
| `get_example_data` | List available example datasets | None |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_motif_discovery` | Complete RNAelem motif discovery pipeline | `input_file`, `pattern_file`, `output_dir`, `force_overwrite`, `job_name` |
| `submit_motif_scanning` | Large-scale motif scanning | `input_file`, `model_file`, `threshold`, `output_format`, `output_dir`, `job_name` |
| `submit_ml_analysis` | Large-scale ML training | `positive_file`, `negative_file`, `model_type`, `test_size`, `n_estimators`, `output_dir`, `job_name` |

### Batch Processing Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_batch_motif_scanning` | Process multiple files with same model | `input_files`, `model_file`, `threshold`, `output_format`, `output_dir`, `job_name` |
| `submit_batch_ml_analysis` | Train models for multiple datasets | `positive_files`, `negative_files`, `model_type`, `output_dir`, `job_name` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs |

---

## Examples

### Example 1: Basic Motif Discovery

**Goal:** Discover RNA motifs from tRNA sequences using a simple hairpin pattern

**Using Script:**
```bash
python scripts/simple_pipeline.py \
  --input examples/data/positive.fa \
  --pattern examples/data/simple_pattern.txt \
  --output results/basic_discovery/
```

**Using MCP (in Claude Code):**
```
Use submit_motif_discovery to process @examples/data/positive.fa with pattern @examples/data/simple_pattern.txt and save results to results/basic_discovery/
```

**Expected Output:**
- Trained models in `results/basic_discovery/model-*/`
- Cross-validation results
- Model selection statistics

### Example 2: Motif Scanning

**Goal:** Scan sequences for known motifs using a trained model

**Using Script:**
```bash
python scripts/motif_scanning.py \
  --input examples/data/positive.fa \
  --model results/basic_discovery/model-1/train.model \
  --threshold 0.8 \
  --format csv \
  --output results/scanning.csv
```

**Using MCP (in Claude Code):**
```
Run scan_motifs on @examples/data/positive.fa using model @results/basic_discovery/model-1/train.model with threshold 0.8 and csv format
```

**Expected Output:**
- Motif existence probabilities for each sequence
- Motif alignments and positions
- Statistics summary

### Example 3: Machine Learning Analysis

**Goal:** Enhanced motif analysis using ensemble machine learning

**Using Script:**
```bash
python scripts/ml_analysis.py \
  --positive examples/data/positive.fa \
  --model-type gradient_boost \
  --n-estimators 500 \
  --output results/ml_enhanced/
```

**Using MCP (in Claude Code):**
```
Submit analyze_sequences_ml for @examples/data/positive.fa using gradient_boost with 500 estimators, save to results/ml_enhanced/
```

**Expected Output:**
- Trained ML models with feature importance
- Performance metrics (accuracy, precision, recall)
- Cross-validation results

### Example 4: Batch Processing

**Goal:** Process multiple sequence files with the same trained model

**Using MCP (in Claude Code):**
```
Submit batch motif scanning for multiple files:
- @examples/data/positive.fa
- @results/new_sequences_1.fa
- @results/new_sequences_2.fa
Use model @results/basic_discovery/model-1/train.model with threshold 0.7
```

**Expected Output:**
- Scanning results for each input file
- Combined statistics across all files
- Progress tracking for the batch job

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `positive.fa` | 76 tRNA sequences with CAU anticodon from Rfam RF00005 | All tools |
| `simple_pattern.txt` | Simple hairpin pattern `(.....)`  | `submit_motif_discovery` |
| `pattern_list` | Complete search patterns in dot-bracket notation | Advanced motif discovery |

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `simple_pipeline_config.json` | Pipeline configuration | `force_overwrite`, `verbose`, `cross_validation` |
| `motif_scanning_config.json` | Scanning parameters | `threshold`, `output_format`, `include_alignments` |
| `ml_analysis_config.json` | ML model settings | `model_type`, `test_size`, `n_estimators`, `cross_validation` |

### Config Example

```json
{
  "model_type": "gradient_boost",
  "n_estimators": 200,
  "test_size": 0.3,
  "cross_validation": 5
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
pip install numpy scipy pandas scikit-learn loguru click tqdm fastmcp
```

**Problem:** Import errors
```bash
# Verify installation
python -c "from src.server import mcp; print('✅ Server imports work')"
```

**Problem:** RNAelem binaries not found
```bash
# Check binaries are available
elem --help
RNAelem --help

# If missing, rebuild
cd repo/RNAelem
./waf configure --prefix=$HOME/local
./waf build
./waf install
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove RNAelem
claude mcp add RNAelem -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Tools not working
```bash
# Test server directly
python -c "
from src.server import mcp
print(list(mcp.list_tools().keys()))
"
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job log
cat jobs/<job_id>/job.log
```

**Problem:** Job failed
```
Use get_job_log with job_id "<job_id>" and tail 100 to see error details
```

### File Format Issues

**Problem:** FASTA format errors
```bash
# Validate file format
python -c "
from Bio import SeqIO
try:
    records = list(SeqIO.parse('examples/data/positive.fa', 'fasta'))
    print(f'✅ Found {len(records)} sequences')
except Exception as e:
    print(f'❌ FASTA error: {e}')
"
```

**Problem:** Pattern file format errors
```bash
# Check pattern file
cat examples/data/simple_pattern.txt
# Should show: (.....
```

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Run basic validation
python -c "
from src.server import mcp
from src.jobs.manager import job_manager
print('✅ All components imported successfully')
"
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
fastmcp dev src/server.py
```

---

## Performance Notes

- **Sync Tools**: Best for small files (< 100 sequences, < 10 min runtime)
- **Submit Tools**: Required for large files or long-running operations (> 10 min)
- **Batch Tools**: Optimal for multiple files with same parameters
- **Memory Usage**: ~500MB for complete environment, varies by dataset size

---

## License

This MCP server implementation is provided under the same license terms as the original RNAelem software.

## Credits

Based on [RNAelem](https://github.com/fukunagatsu/RNAelem) by Fukunaga Lab.
MCP integration developed for NucleicMCP toolkit.