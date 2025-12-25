# RNAelem Use Cases and Examples

This directory contains example scripts demonstrating the main use cases of RNAelem for RNA sequence-structure motif discovery.

## Available Use Cases

### UC-001: Simple Motif Discovery Pipeline
**Script:** `use_case_1_simple_pipeline.py`
**Complexity:** Simple
**Priority:** High

A straightforward pipeline that demonstrates the basic RNAelem workflow using the toy example from the documentation.

**Inputs:**
- Positive sequences (FASTA format)
- Pattern file (dot-bracket notation)

**Outputs:**
- Trained motif models
- Sequence logos and structure diagrams

**Example Usage:**
```bash
python examples/use_case_1_simple_pipeline.py --input examples/data/positive.fa --pattern examples/data/simple_pattern.txt
```

### UC-002: Manual Parallel Training
**Script:** `use_case_2_manual_training.py`
**Complexity:** Medium
**Priority:** High

Step-by-step motif discovery with manual control over each phase. Useful for understanding the training process and custom parameter tuning.

**Inputs:**
- Positive sequences (FASTA format)
- Pattern list file (multiple patterns)

**Outputs:**
- Multiple trained models
- Cross-validation results
- Model selection results

**Example Usage:**
```bash
python examples/use_case_2_manual_training.py --input examples/data/positive.fa --pattern-list examples/data/pattern_list --max-patterns 5
```

### UC-003: Motif Scanning
**Script:** `use_case_3_motif_scanning.py`
**Complexity:** Simple
**Priority:** High

Apply trained RNAelem models to scan new RNA sequences for motifs.

**Inputs:**
- Sequences to scan (FASTA format)
- Trained model file(s)

**Outputs:**
- Motif alignment results
- Existence probabilities
- Motif regions

**Example Usage:**
```bash
# Single model scanning
python examples/use_case_3_motif_scanning.py --sequences examples/data/positive.fa --model elem_simple_out/model-1/train.model

# Multiple model scanning
python examples/use_case_3_motif_scanning.py --sequences examples/data/positive.fa --model-dir elem_simple_out
```

### UC-004: Motif Visualization
**Script:** `use_case_4_visualization.py`
**Complexity:** Medium
**Priority:** Medium

Generate visualizations of discovered motifs including sequence logos and RNA secondary structure diagrams.

**Inputs:**
- Trained model directories
- Visualization parameters

**Outputs:**
- Sequence profile logos (SVG/PNG)
- RNA secondary structure diagrams (EPS/PNG)

**Example Usage:**
```bash
python examples/use_case_4_visualization.py --model-dir elem_simple_out --output viz_out
```

### UC-005: Machine Learning Analysis
**Script:** `use_case_5_ml_analysis.py`
**Complexity:** Complex
**Priority:** Medium

Enhanced motif analysis using machine learning with CatBoost, LightGBM, or Random Forest.

**Inputs:**
- Positive sequences (FASTA format)
- Optional negative sequences

**Outputs:**
- ML model performance metrics
- Feature importance analysis
- Classification results

**Example Usage:**
```bash
python examples/use_case_5_ml_analysis.py --positive examples/data/positive.fa --model-type catboost --output ml_analysis_out
```

## Demo Data

### `examples/data/positive.fa`
Sample tRNA sequences from Rfam family RF00005. Contains 76 seed sequences with 'CAU' anticodon exhibiting tRNA clover structure.

### `examples/data/pattern_list`
Complete list of search patterns in dot-bracket notation. Includes simple loops, stems, and complex secondary structures with insertion regions (*).

### `examples/data/simple_pattern.txt`
Single hairpin loop pattern `(.....)` for the toy example.

## Requirements

All scripts require the RNAelem environment to be activated:

```bash
# Activate environment
conda activate ./env

# Run any use case
python examples/use_case_X_name.py [arguments]
```

## Workflow Examples

### Complete Pipeline
1. **Simple Discovery:** Start with use case 1 to get familiar with basic workflow
2. **Manual Training:** Use case 2 for more control and understanding
3. **Scanning:** Use case 3 to apply models to new data
4. **Visualization:** Use case 4 to generate publication-ready figures
5. **ML Analysis:** Use case 5 for advanced feature analysis

### Quick Start
For a quick demonstration of RNAelem capabilities:
```bash
# Activate environment
conda activate ./env

# Run simple pipeline
python examples/use_case_1_simple_pipeline.py

# Scan with trained model
python examples/use_case_3_motif_scanning.py --model elem_simple_out/model-1/train.model

# Generate visualizations
python examples/use_case_4_visualization.py --model-dir elem_simple_out
```

## Troubleshooting

### Missing Tools
Some visualization features require additional tools:
- `rsvg-convert`: For SVG to PNG conversion
- `convert` (ImageMagick): For EPS to PNG conversion

Install on Ubuntu/Debian:
```bash
sudo apt-get install librsvg2-bin imagemagick
```

### ML Dependencies
For machine learning features, install optional dependencies:
```bash
pip install catboost lightgbm
```

## Output Structure

Each use case creates its own output directory:
```
./
├── elem_simple_out/          # Use case 1 outputs
├── elem_manual_out/          # Use case 2 outputs
├── scan_out/                 # Use case 3 outputs
├── viz_out/                  # Use case 4 outputs
└── ml_analysis_out/          # Use case 5 outputs
```