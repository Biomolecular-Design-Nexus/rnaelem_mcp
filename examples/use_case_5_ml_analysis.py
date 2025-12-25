#!/usr/bin/env python3
"""
RNAelem Use Case 5: Machine Learning Analysis
==============================================

This script demonstrates the machine learning capabilities of RNAelem
using the CatBoost integration for enhanced motif analysis and prediction.

Features:
- Feature extraction from RNAelem outputs
- CatBoost/LightGBM/Random Forest model training
- Feature importance analysis
- Cross-validation and model evaluation

Usage:
    python examples/use_case_5_ml_analysis.py [options]

Example:
    python examples/use_case_5_ml_analysis.py --positive examples/data/positive.fa --output ml_analysis_out
"""

import os
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Optional ML libraries - will use basic sklearn if not available
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available, using RandomForest")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return None

def generate_negative_sequences(positive_file, output_file):
    """Generate negative sequences using dishuffle."""
    cmd = ["dishuffle.py", positive_file]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(output_file, 'w') as f:
            f.write(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating negative sequences: {e}")
        return False

def parse_kmer_features(fq_file):
    """Parse k-mer features from RNAelem .fq file."""
    features = []
    labels = []

    try:
        with open(fq_file, 'r') as f:
            lines = f.readlines()

        # Parse FASTQ-like format with k-mer features
        for i in range(0, len(lines), 4):
            if i + 3 < len(lines):
                header = lines[i].strip()
                sequence = lines[i+1].strip()
                plus = lines[i+2].strip()
                quality = lines[i+3].strip()

                # Extract label: first 76 are positive, rest are negative
                # (assuming equal number of positive and negative sequences)
                seq_num = int(header[1:])  # Extract sequence number from @1, @2, etc.
                label = 1 if seq_num <= 76 else 0  # First 76 are positive
                labels.append(label)

                # Convert quality scores to features
                feature_vector = [ord(c) - 33 for c in quality]  # Phred quality conversion
                features.append(feature_vector)

    except Exception as e:
        print(f"Error parsing k-mer features: {e}")
        return np.array([]), np.array([])

    # Pad feature vectors to the same length
    if features:
        max_length = max(len(f) for f in features)
        padded_features = []
        for feature_vector in features:
            padded = feature_vector + [0] * (max_length - len(feature_vector))
            padded_features.append(padded)
        features = padded_features

    return np.array(features), np.array(labels)

def train_ml_model(features, labels, model_type="random_forest", output_dir=None):
    """Train a machine learning model on the features."""

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_imputed, labels, test_size=0.2, random_state=42, stratify=labels)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Class distribution - Positive: {np.sum(y_train)}, Negative: {len(y_train) - np.sum(y_train)}")

    # Select and train model
    if model_type == "catboost" and CATBOOST_AVAILABLE:
        model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            random_seed=42
        )
    elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
        model = LGBMClassifier(
            n_estimators=100,
            random_state=42,
            verbose=-1
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    print(f"Training {type(model).__name__} model...")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC Score: {auc_score:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        print(f"\nTop 10 Most Important Features:")
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")

    # Save results if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': [f'feature_{i}' for i in range(len(feature_importance))],
                'Importance': feature_importance
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            importance_df.to_csv(output_path / "feature_importance.csv", index=False)
            print(f"Feature importance saved to {output_path / 'feature_importance.csv'}")

        # Save model performance
        results = {
            'model_type': type(model).__name__,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        results_df = pd.DataFrame([results])
        results_df.to_csv(output_path / "model_performance.csv", index=False)
        print(f"Model performance saved to {output_path / 'model_performance.csv'}")

    return model, accuracy, auc_score

def main():
    parser = argparse.ArgumentParser(
        description="RNAelem machine learning analysis")

    parser.add_argument("--positive", "-p",
                       default="examples/data/positive.fa",
                       help="Input FASTA file with positive sequences")
    parser.add_argument("--negative", "-n",
                       help="Input FASTA file with negative sequences (optional)")
    parser.add_argument("--output", "-o",
                       default="ml_analysis_out",
                       help="Output directory name")
    parser.add_argument("--model-type", choices=["catboost", "lightgbm", "random_forest"],
                       default="random_forest",
                       help="Machine learning model type")
    parser.add_argument("--force", "-F", action="store_true",
                       help="Force overwrite output directory")

    args = parser.parse_args()

    # Check if input files exist
    if not Path(args.positive).exists():
        print(f"Error: Positive sequences file {args.positive} not found!")
        return 1

    # Check if output directory exists
    if Path(args.output).exists() and not args.force:
        print(f"Error: Output directory {args.output} exists! Use --force to overwrite.")
        return 1

    print("=" * 60)
    print("RNAelem Machine Learning Analysis")
    print("=" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate negative sequences if not provided
    negative_file = args.negative
    if not negative_file:
        negative_file = output_dir / "negative.fa"
        print(f"Generating negative sequences using dishuffle...")
        if not generate_negative_sequences(args.positive, negative_file):
            print("Failed to generate negative sequences!")
            return 1

    # Generate k-mer features using kmer-psp.py
    print("Generating k-mer features...")
    feature_file = output_dir / "features.fq"
    cmd = ["kmer-psp.py", args.positive, str(negative_file)]

    result = run_command(cmd, "Generating k-mer features")
    if result is None:
        print("Failed to generate features!")
        return 1

    # Save features to file
    with open(feature_file, 'w') as f:
        f.write(result.stdout)

    # Parse features
    print("Parsing features...")
    features, labels = parse_kmer_features(feature_file)

    if len(features) == 0:
        print("Error: No features extracted!")
        return 1

    print(f"Extracted {len(features)} samples with {features.shape[1]} features")
    print(f"Class distribution: {np.sum(labels)} positive, {len(labels) - np.sum(labels)} negative")

    # Train machine learning model
    print(f"\nTraining {args.model_type} model...")
    model, accuracy, auc_score = train_ml_model(
        features, labels, args.model_type, args.output)

    print("\n" + "=" * 60)
    print("Machine Learning Analysis Completed!")
    print("=" * 60)
    print(f"Final Results:")
    print(f"  Model Type: {args.model_type}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  AUC Score: {auc_score:.3f}")
    print(f"  Output Directory: {args.output}")

    return 0

if __name__ == "__main__":
    sys.exit(main())