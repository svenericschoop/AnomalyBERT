#!/usr/bin/env python3
"""
Custom data preprocessing script for unlabeled training data.
This script processes your custom training_data_anomalyBERT.txt file 
and prepares it for AnomalyBERT training.
"""

import os
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler


def preprocess_custom_data(input_file, output_dir='processed', dataset_name='CUSTOM'):
    """
    Preprocess custom unlabeled training data for AnomalyBERT.
    
    Args:
        input_file (str): Path to the input training data file
        output_dir (str): Directory to save processed data
        dataset_name (str): Name for the dataset
    """
    
    print(f"Processing custom data from: {input_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print("Loading data...")
    data = np.genfromtxt(input_file, dtype=np.float32, delimiter=',')
    print(f"Data shape: {data.shape}")
    
    # Data is already scaled between 0 and 1, so we skip normalization
    print("Data is already scaled between 0 and 1, skipping normalization...")
    normalized_data = data.copy()
    
    # Verify data range
    data_min, data_max = normalized_data.min(), normalized_data.max()
    print(f"Data range: [{data_min:.6f}, {data_max:.6f}]")
    
    # Only clip if values are slightly outside [0, 1] range due to floating point precision
    if data_min < -0.001 or data_max > 1.001:
        print("Warning: Data values outside [0, 1] range detected. Clipping values...")
        normalized_data = np.clip(normalized_data, 0, 1)
    else:
        print("Data is properly scaled within [0, 1] range.")
    
    # Save the processed data
    train_file = os.path.join(output_dir, f"{dataset_name}_train.npy")
    np.save(train_file, normalized_data)
    
    print(f"Processed data saved to: {train_file}")
    print(f"Final data shape: {normalized_data.shape}")
    print(f"Data range: [{normalized_data.min():.6f}, {normalized_data.max():.6f}]")
    
    # Create a dummy test file for compatibility (using a subset of training data)
    # This is needed because the training script expects test data
    test_size = min(1000, len(normalized_data) // 4)  # Use 25% or max 1000 samples
    test_data = normalized_data[:test_size]
    test_file = os.path.join(output_dir, f"{dataset_name}_test.npy")
    np.save(test_file, test_data)
    
    # Create dummy test labels (all zeros - no anomalies in test set)
    test_labels = np.zeros(len(test_data), dtype=np.int32)
    label_file = os.path.join(output_dir, f"{dataset_name}_test_label.npy")
    np.save(label_file, test_labels)
    
    print(f"Dummy test data saved to: {test_file}")
    print(f"Dummy test labels saved to: {label_file}")
    
    return train_file, test_file, label_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess custom unlabeled training data for AnomalyBERT")
    parser.add_argument("--input_file", required=True, type=str, 
                       help="Path to the input training data file")
    parser.add_argument("--output_dir", default="processed", type=str,
                       help="Directory to save processed data")
    parser.add_argument("--dataset_name", default="CUSTOM", type=str,
                       help="Name for the dataset")
    
    args = parser.parse_args()
    
    preprocess_custom_data(args.input_file, args.output_dir, args.dataset_name)
