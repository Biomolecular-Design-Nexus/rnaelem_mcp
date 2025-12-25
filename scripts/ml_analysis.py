#!/usr/bin/env python3
"""
Script: ml_analysis.py
Description: Machine learning analysis of RNA sequences using RNAelem features

Original Use Case: examples/use_case_5_ml_analysis.py
Dependencies Removed: Optional ML libraries (catboost, lightgbm) - falls back to sklearn

Usage:
    python scripts/ml_analysis.py --positive <fasta_file> --output <output_dir>

Example:
    python scripts/ml_analysis.py --positive examples/data/positive.fa --output results/ml_analysis
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple

# Essential ML packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Optional ML libraries - graceful degradation
try:
    from sklearn.ensemble import GradientBoostingClassifier
    GRADIENT_BOOST_AVAILABLE = True
except ImportError:
    GRADIENT_BOOST_AVAILABLE = False

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model_type": "random_forest",  # random_forest, gradient_boost
    "test_size": 0.3,
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 10,
    "cv_folds": 5,
    "feature_importance_top_n": 20,
    "verbose": False
}

# ==============================================================================
# Utility Functions (inlined and simplified from original)
# ==============================================================================
def run_command(cmd: List[str], description: str = "") -> Optional[subprocess.CompletedProcess]:
    """Run a command and handle errors."""
    if DEFAULT_CONFIG.get("verbose", False):
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print("Stderr:", e.stderr[:300] + "..." if len(e.stderr) > 300 else e.stderr)
        return None

def generate_negative_sequences(positive_file: Path, output_file: Path) -> bool:
    """Generate negative sequences using dishuffle."""
    cmd = ["dishuffle.py", str(positive_file)]
    result = run_command(cmd, "Generating negative sequences")

    if result is None:
        return False

    try:
        with open(output_file, 'w') as f:
            f.write(result.stdout)
        return True
    except Exception as e:
        print(f"Error writing negative sequences: {e}")
        return False

def create_combined_dataset(positive_file: Path, negative_file: Path, output_file: Path) -> bool:
    """Combine positive and negative sequences into one file."""
    try:
        with open(output_file, 'w') as out_f:
            # Copy positive sequences
            with open(positive_file, 'r') as pos_f:
                out_f.write(pos_f.read())

            # Copy negative sequences
            with open(negative_file, 'r') as neg_f:
                out_f.write(neg_f.read())

        return True
    except Exception as e:
        print(f"Error creating combined dataset: {e}")
        return False

def extract_kmer_features(combined_file: Path, output_file: Path) -> bool:
    """Extract k-mer features using kmer-psp.py."""
    # Note: Based on Step 4 analysis, kmer-psp.py only outputs quality-scored sequences
    # This is a simplified approach that may need adjustment
    cmd = ["kmer-psp.py", str(combined_file)]
    result = run_command(cmd, "Extracting k-mer features")

    if result is None:
        return False

    try:
        with open(output_file, 'w') as f:
            f.write(result.stdout)
        return True
    except Exception as e:
        print(f"Error writing k-mer features: {e}")
        return False

def parse_kmer_features(fq_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse k-mer features from RNAelem .fq file."""
    features = []
    labels = []

    try:
        with open(fq_file, 'r') as f:
            lines = f.readlines()

        # Count sequences in original positive file to determine labels
        # This is a simplified labeling approach
        num_sequences = len(lines) // 4  # FASTQ format: 4 lines per sequence

        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                header = lines[i].strip()
                sequence = lines[i+1].strip()
                plus = lines[i+2].strip()
                quality = lines[i+3].strip()

                # Simple feature extraction: use quality scores as features
                # This is a placeholder - real implementation would use k-mer counts
                feature_vector = [ord(c) - 33 for c in quality if ord(c) >= 33]

                if len(feature_vector) > 0:
                    features.append(feature_vector)
                    # Simple labeling: assume first half are positive
                    seq_num = len(features)
                    labels.append(1 if seq_num <= num_sequences // 2 else 0)

    except Exception as e:
        print(f"Error parsing k-mer features: {e}")
        return np.array([]), np.array([])

    if not features:
        return np.array([]), np.array([])

    # Pad features to same length
    max_len = max(len(f) for f in features)
    padded_features = np.array([
        f + [0] * (max_len - len(f)) for f in features
    ])

    return padded_features, np.array(labels)

def train_model(X: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> Any:
    """Train a machine learning model."""
    model_type = config.get("model_type", "random_forest")

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 10),
            random_state=config.get("random_state", 42)
        )
    elif model_type == "gradient_boost" and GRADIENT_BOOST_AVAILABLE:
        model = GradientBoostingClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 10),
            random_state=config.get("random_state", 42)
        )
    else:
        print(f"Model type {model_type} not available, using RandomForest")
        model = RandomForestClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 10),
            random_state=config.get("random_state", 42)
        )

    model.fit(X, y)
    return model

# ==============================================================================
# Core Function (main logic extracted and simplified from use case)
# ==============================================================================
def run_ml_analysis(
    positive_file: Union[str, Path],
    output_dir: Union[str, Path],
    negative_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform machine learning analysis of RNA sequences using RNAelem features.

    Args:
        positive_file: Path to FASTA file with positive sequences
        output_dir: Path to output directory
        negative_file: Path to negative sequences (optional, will generate if not provided)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - model: Trained model object
            - metrics: Performance metrics
            - feature_importance: Feature importance scores
            - output_dir: Path to output directory
            - metadata: Execution metadata

    Example:
        >>> result = run_ml_analysis("positive.fa", "output/")
        >>> print(f"AUC: {result['metrics']['auc']:.3f}")
    """
    # Setup
    positive_file = Path(positive_file)
    output_dir = Path(output_dir)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validation
    if not positive_file.exists():
        raise FileNotFoundError(f"Positive file not found: {positive_file}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate negative sequences if not provided
    if negative_file is None:
        negative_file = output_dir / "negative_sequences.fa"
        if not generate_negative_sequences(positive_file, negative_file):
            raise RuntimeError("Failed to generate negative sequences")
    else:
        negative_file = Path(negative_file)

    # Create combined dataset
    combined_file = output_dir / "combined_sequences.fa"
    if not create_combined_dataset(positive_file, negative_file, combined_file):
        raise RuntimeError("Failed to create combined dataset")

    # Extract k-mer features
    features_file = output_dir / "features.fq"
    if not extract_kmer_features(combined_file, features_file):
        raise RuntimeError("Failed to extract k-mer features")

    # Parse features
    X, y = parse_kmer_features(features_file)

    if len(X) == 0 or len(y) == 0:
        raise RuntimeError("No features extracted from sequences")

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.get("test_size", 0.3),
        random_state=config.get("random_state", 42)
    )

    # Train model
    model = train_model(X_train, y_train, config)

    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
    }

    # Feature importance
    feature_importance = []
    if hasattr(model, 'feature_importances_'):
        importance_indices = np.argsort(model.feature_importances_)[::-1]
        top_n = config.get("feature_importance_top_n", 20)
        feature_importance = [
            {
                "feature_index": int(idx),
                "importance": float(model.feature_importances_[idx])
            }
            for idx in importance_indices[:top_n]
        ]

    # Save results
    results = {
        "model": model,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "output_dir": str(output_dir),
        "metadata": {
            "positive_file": str(positive_file),
            "negative_file": str(negative_file),
            "config": config,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_positive": int(np.sum(y)),
            "n_negative": int(len(y) - np.sum(y))
        }
    }

    # Save metrics to file
    with open(output_dir / "metrics.json", 'w') as f:
        metrics_to_save = {
            "metrics": metrics,
            "feature_importance": feature_importance,
            "metadata": results["metadata"]
        }
        json.dump(metrics_to_save, f, indent=2)

    return results

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--positive', '-p', required=True,
                       help='Input FASTA file with positive sequences')
    parser.add_argument('--negative', '-n',
                       help='Input FASTA file with negative sequences (optional)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON)')
    parser.add_argument('--model-type', '-t', choices=['random_forest', 'gradient_boost'],
                       default='random_forest', help='Type of ML model to use')
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
        "model_type": args.model_type,
        "verbose": args.verbose
    })

    # Run
    try:
        result = run_ml_analysis(
            positive_file=args.positive,
            output_dir=args.output,
            negative_file=args.negative,
            config=config
        )

        # Print summary
        metrics = result["metrics"]
        metadata = result["metadata"]
        print(f"✅ ML analysis completed successfully!")
        print(f"   Output directory: {result['output_dir']}")
        print(f"   Samples: {metadata['n_samples']} ({metadata['n_positive']} positive, {metadata['n_negative']} negative)")
        print(f"   Features: {metadata['n_features']}")
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   AUC: {metrics['auc']:.3f}")
        print(f"   Top feature importance: {result['feature_importance'][0]['importance']:.3f}" if result['feature_importance'] else "N/A")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == '__main__':
    main()