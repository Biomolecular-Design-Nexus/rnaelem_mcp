# Step 3: Use Cases Extraction Report - RNAelem MCP

## Executive Summary

Successfully identified and extracted 5 core RNAelem use cases into standalone Python scripts. Created comprehensive workflow examples covering basic motif discovery through advanced machine learning analysis. All scripts include proper error handling, CLI interfaces, and documentation.

## Use Case Identification Process

### Source Analysis
**Primary Sources:**
- `repo/RNAelem/README.md` - Official documentation and toy example
- `repo/RNAelem/script/elem` - 558-line main pipeline script
- `repo/RNAelem/material/` - Demo data and examples
- `repo/RNAelem/script/` - Utility scripts (draw_motif.py, kmer-psp.py, dishuffle.py)

### Methodology
1. Analyzed elem script command structure (init, train, select, refine, scan, pipeline)
2. Identified common workflow patterns from documentation
3. Extracted reusable components and patterns
4. Created standalone scripts with proper CLI interfaces

## Extracted Use Cases

### UC-001: Simple Motif Discovery Pipeline
**File:** `examples/use_case_1_simple_pipeline.py`
**Complexity:** Simple
**Priority:** High

**Purpose:** Demonstrates basic RNAelem workflow using single command pipeline

**Key Features:**
- Single-command execution via `elem pipeline`
- Uses toy example from documentation
- Input validation and error handling
- Progress tracking and status reporting

**Input Requirements:**
- Positive sequences (FASTA format)
- Pattern file (dot-bracket notation)

**Sample Command:**
```bash
python examples/use_case_1_simple_pipeline.py --input examples/data/positive.fa --pattern examples/data/simple_pattern.txt
```

### UC-002: Manual Parallel Training
**File:** `examples/use_case_2_manual_training.py`
**Complexity:** Medium
**Priority:** High

**Purpose:** Step-by-step manual control over motif discovery process

**Key Features:**
- Multi-stage workflow (init → train → select → refine)
- Pattern iteration and parallel training
- Cross-validation support
- Model selection and comparison
- Detailed progress tracking

**Workflow Steps:**
1. Initialize workspace (`elem init`)
2. Train multiple models in parallel (`elem train`)
3. Select best models (`elem select`)
4. Refine selected models (`elem refine`)

**Sample Command:**
```bash
python examples/use_case_2_manual_training.py --input examples/data/positive.fa --pattern-list examples/data/pattern_list --max-patterns 5
```

### UC-003: Motif Scanning
**File:** `examples/use_case_3_motif_scanning.py`
**Complexity:** Simple
**Priority:** High

**Purpose:** Apply trained models to scan new sequences for motifs

**Key Features:**
- Single model or batch model scanning
- Raw output parsing and interpretation
- Existence probability extraction
- Motif alignment results
- CSV output formatting

**Output Parsing:**
- Extracts sequence IDs, existence probabilities
- Parses motif alignments and positions
- Handles multiple model results

**Sample Commands:**
```bash
# Single model
python examples/use_case_3_motif_scanning.py --sequences examples/data/positive.fa --model elem_simple_out/model-1/train.model

# Multiple models
python examples/use_case_3_motif_scanning.py --sequences examples/data/positive.fa --model-dir elem_simple_out
```

### UC-004: Motif Visualization
**File:** `examples/use_case_4_visualization.py`
**Complexity:** Medium
**Priority:** Medium

**Purpose:** Generate publication-ready visualizations of discovered motifs

**Key Features:**
- Sequence logo generation (`draw_motif.py`)
- SVG to PNG conversion (`rsvg-convert`)
- EPS to PNG conversion (`ImageMagick convert`)
- Batch processing of multiple models
- Graceful degradation when tools missing

**Visualization Types:**
- Sequence profile logos (SVG/PNG)
- RNA secondary structure diagrams (EPS/PNG)
- Combined visualization outputs

**Sample Command:**
```bash
python examples/use_case_4_visualization.py --model-dir elem_simple_out --output viz_out
```

### UC-005: Machine Learning Analysis
**File:** `examples/use_case_5_ml_analysis.py`
**Complexity:** Complex
**Priority:** Medium

**Purpose:** Enhanced motif analysis using machine learning

**Key Features:**
- Negative sequence generation (`dishuffle.py`)
- K-mer feature extraction (`kmer-psp.py`)
- Multiple ML models (CatBoost, LightGBM, Random Forest)
- Feature importance analysis
- Cross-validation and performance metrics
- Graceful fallback to sklearn when advanced libraries unavailable

**ML Pipeline:**
1. Generate negative sequences if not provided
2. Extract k-mer features from positive/negative sets
3. Train classification model
4. Evaluate performance and feature importance
5. Save results and visualizations

**Sample Command:**
```bash
python examples/use_case_5_ml_analysis.py --positive examples/data/positive.fa --model-type catboost --output ml_analysis_out
```

## Demo Data Integration

### Data Files Copied
- `examples/data/positive.fa` - 76 tRNA sequences from Rfam RF00005
- `examples/data/pattern_list` - Complete search patterns in dot-bracket notation
- `examples/data/simple_pattern.txt` - Single hairpin pattern for toy example

### Data Characteristics
- **Sequence Type:** tRNA with 'CAU' anticodon
- **Structure:** Cloverleaf secondary structure
- **Format:** FASTA with proper headers
- **Pattern Format:** Dot-bracket with `*` for insertion regions

## Technical Implementation Details

### Common Features Across All Scripts
- **CLI Interface:** argparse with comprehensive help
- **Error Handling:** Subprocess error capture and reporting
- **Input Validation:** File existence and format checking
- **Progress Tracking:** Status messages and progress indicators
- **Output Management:** Configurable output directories
- **Force Overwrite:** --force flag for development workflows

### Error Handling Patterns
- Subprocess call wrapping with error capture
- File existence validation before execution
- Graceful degradation when optional tools missing
- Clear error messages with troubleshooting hints

### Integration Points
- All scripts use RNAelem binaries from environment PATH
- Consistent output directory structures
- Compatible with MCP server integration
- Modular design for easy extension

## Workflow Relationships

### Typical Usage Sequence
1. **Start:** Use Case 1 (Simple Pipeline) for initial exploration
2. **Advanced:** Use Case 2 (Manual Training) for parameter tuning
3. **Application:** Use Case 3 (Motif Scanning) for new data analysis
4. **Visualization:** Use Case 4 for publication figures
5. **Analysis:** Use Case 5 for detailed ML analysis

### Dependencies
- UC-003 requires models from UC-001 or UC-002
- UC-004 requires models from UC-001 or UC-002
- UC-005 is independent but can use results from others

## Documentation Integration

### README Files Created
- `examples/README.md` - Comprehensive use case documentation
- Individual script help via `--help` flag
- Code comments explaining RNAelem integration

### Usage Examples
- Complete command examples with actual file paths
- Parameter explanations and options
- Expected output descriptions
- Troubleshooting guidance

## Success Metrics

- ✅ 5 use cases successfully extracted
- ✅ All scripts have CLI interfaces
- ✅ Proper error handling implemented
- ✅ Demo data successfully integrated
- ✅ Documentation created
- ✅ Workflow relationships established
- ✅ MCP server compatibility maintained

## Recommendations

### Immediate Use
1. Start with Use Case 1 for basic functionality verification
2. Progress to Use Case 2 for understanding advanced features
3. Use Case 3 for practical motif scanning applications

### Advanced Development
1. Use Case 4 for publication-quality visualization
2. Use Case 5 for machine learning enhanced analysis
3. Extend scripts for custom workflows

### Production Deployment
- All scripts ready for MCP server integration
- Modular design supports custom extensions
- Error handling suitable for automated workflows

## Conclusion

Successfully extracted and implemented 5 comprehensive use cases covering the full spectrum of RNAelem functionality. Scripts provide robust foundation for both interactive use and MCP server integration, with proper error handling and documentation for production deployment.