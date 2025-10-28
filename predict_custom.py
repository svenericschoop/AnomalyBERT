#!/usr/bin/env python3
"""
Custom dataset prediction script for AnomalyBERT.
This script uses a trained AnomalyBERT model to predict anomalies in your custom dataset.
"""

import os
import json
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pandas as pd

import utils.config as config
from models.anomaly_transformer import get_anomaly_transformer
from estimate import estimate


def load_model(model_path, state_dict_path, device, hyperparams):
    """
    Load the trained AnomalyBERT model.
    
    Args:
        model_path (str): Path to the model file
        state_dict_path (str): Path to the state dict file
        device: PyTorch device
        hyperparams (dict): Model hyperparameters
    
    Returns:
        Loaded model
    """
    print(f"Loading model from: {model_path}")
    
    # Create model with same architecture as training
    model = get_anomaly_transformer(
        input_d_data=hyperparams['n_features'],
        output_d_data=1,  # BCE loss uses 1 output
        patch_size=hyperparams['patch_size'],
        d_embed=hyperparams['d_embed'],
        hidden_dim_rate=4.,
        max_seq_len=hyperparams['n_features'],
        positional_encoding=None,
        relative_position_embedding=True,
        transformer_n_layer=hyperparams['n_layer'],
        transformer_n_head=8,
        dropout=hyperparams['dropout']
    ).to(device)
    
    # Load state dict if provided
    if state_dict_path and os.path.exists(state_dict_path):
        print(f"Loading state dict from: {state_dict_path}")
        model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
    else:
        print(f"Loading full model from: {model_path}")
        model = torch.load(model_path, map_location=device)
    
    model.eval()
    return model


def predict_anomalies(data, model, hyperparams, device, batch_size=16, window_sliding=16):
    """
    Predict anomaly scores for the given data.
    
    Args:
        data (np.array): Input data
        model: Trained AnomalyBERT model
        hyperparams (dict): Model hyperparameters
        device: PyTorch device
        batch_size (int): Batch size for prediction
        window_sliding (int): Window sliding step
    
    Returns:
        Anomaly scores as numpy array
    """
    print("Computing anomaly scores...")
    
    # Set up data divisions (use total for custom data)
    divisions = [[0, len(data)]]
    
    # Set up post-activation (sigmoid for BCE loss)
    post_activation = torch.nn.Sigmoid().to(device)
    
    # Estimate anomaly scores
    anomaly_scores = estimate(
        data, model, post_activation, 1, batch_size, 
        window_sliding, divisions, None, device
    )
    
    return anomaly_scores.cpu().numpy()


def find_optimal_threshold(scores, labels=None, method='quantile', quantile=0.95):
    """
    Find optimal threshold for anomaly detection.
    
    Args:
        scores (np.array): Anomaly scores
        labels (np.array): Ground truth labels (optional)
        method (str): Method to use ('quantile', 'otsu', 'precision_recall')
        quantile (float): Quantile threshold (for quantile method)
    
    Returns:
        Optimal threshold value
    """
    if method == 'quantile':
        threshold = np.quantile(scores, quantile)
        print(f"Using quantile threshold: {threshold:.6f} (quantile={quantile})")
        return threshold
    
    elif method == 'otsu' and labels is not None:
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(scores)
        print(f"Using Otsu threshold: {threshold:.6f}")
        return threshold
    
    elif method == 'precision_recall' and labels is not None:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        threshold = thresholds[optimal_idx]
        print(f"Using precision-recall optimal threshold: {threshold:.6f} (F1={f1_scores[optimal_idx]:.4f})")
        return threshold
    
    else:
        # Default to 95th percentile
        threshold = np.quantile(scores, 0.95)
        print(f"Using default quantile threshold: {threshold:.6f}")
        return threshold


def detect_anomalies(scores, threshold):
    """
    Detect anomalies based on threshold.
    
    Args:
        scores (np.array): Anomaly scores
        threshold (float): Anomaly threshold
    
    Returns:
        Binary anomaly labels
    """
    return (scores > threshold).astype(int)


def visualize_results(data, scores, anomalies, threshold, save_path=None):
    """
    Visualize the prediction results.
    
    Args:
        data (np.array): Original data
        scores (np.array): Anomaly scores
        anomalies (np.array): Detected anomalies
        threshold (float): Anomaly threshold
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Original data
    axes[0].plot(data, alpha=0.7, linewidth=0.8)
    axes[0].set_title('Original Data')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Anomaly scores
    axes[1].plot(scores, color='blue', alpha=0.7, linewidth=0.8)
    axes[1].axhline(y=threshold, color='red', linestyle='--', alpha=0.8, label=f'Threshold: {threshold:.4f}')
    axes[1].set_title('Anomaly Scores')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Detected anomalies
    anomaly_indices = np.where(anomalies == 1)[0]
    axes[2].plot(data, alpha=0.7, linewidth=0.8, color='blue', label='Normal')
    if len(anomaly_indices) > 0:
        axes[2].scatter(anomaly_indices, data[anomaly_indices], 
                       color='red', s=20, alpha=0.8, label=f'Anomalies ({len(anomaly_indices)})')
    axes[2].set_title('Detected Anomalies')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def save_results(scores, anomalies, threshold, output_path):
    """
    Save prediction results to files.
    
    Args:
        scores (np.array): Anomaly scores
        anomalies (np.array): Detected anomalies
        threshold (float): Anomaly threshold
        output_path (str): Base path for output files
    """
    # Save anomaly scores
    scores_path = f"{output_path}_scores.npy"
    np.save(scores_path, scores)
    print(f"Anomaly scores saved to: {scores_path}")
    
    # Save binary predictions
    predictions_path = f"{output_path}_predictions.npy"
    np.save(predictions_path, anomalies)
    print(f"Binary predictions saved to: {predictions_path}")
    
    # Save summary statistics
    summary_path = f"{output_path}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Anomaly Detection Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Total data points: {len(scores)}\n")
        f.write(f"Anomaly threshold: {threshold:.6f}\n")
        f.write(f"Detected anomalies: {np.sum(anomalies)}\n")
        f.write(f"Anomaly rate: {np.sum(anomalies) / len(anomalies):.4f}\n")
        f.write(f"Score statistics:\n")
        f.write(f"  Min: {np.min(scores):.6f}\n")
        f.write(f"  Max: {np.max(scores):.6f}\n")
        f.write(f"  Mean: {np.mean(scores):.6f}\n")
        f.write(f"  Std: {np.std(scores):.6f}\n")
        f.write(f"  Median: {np.median(scores):.6f}\n")
        f.write(f"  95th percentile: {np.quantile(scores, 0.95):.6f}\n")
        f.write(f"  99th percentile: {np.quantile(scores, 0.99):.6f}\n")
    
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict anomalies in custom dataset using trained AnomalyBERT")
    parser.add_argument("--data_file", required=True, type=str, 
                       help="Path to the data file (.npy or .txt)")
    parser.add_argument("--model_dir", required=True, type=str,
                       help="Path to the model directory (e.g., logs/250923121043_CUSTOM)")
    parser.add_argument("--state_dict", default="state_dict.pt", type=str,
                       help="State dict filename (default: state_dict.pt)")
    parser.add_argument("--output_dir", default="predictions", type=str,
                       help="Output directory for results")
    parser.add_argument("--threshold_method", default="quantile", type=str,
                       choices=["quantile", "otsu", "precision_recall"],
                       help="Method for threshold selection")
    parser.add_argument("--threshold_quantile", default=0.95, type=float,
                       help="Quantile for threshold selection (0.0-1.0)")
    parser.add_argument("--custom_threshold", default=None, type=float,
                       help="Custom threshold value (overrides other methods)")
    parser.add_argument("--batch_size", default=16, type=int,
                       help="Batch size for prediction")
    parser.add_argument("--window_sliding", default=16, type=int,
                       help="Window sliding step")
    parser.add_argument("--gpu_id", default=0, type=int,
                       help="GPU ID to use")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization plots")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save visualization plots")
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load hyperparameters
    hyperparams_path = os.path.join(args.model_dir, "hyperparameters.txt")
    if not os.path.exists(hyperparams_path):
        raise FileNotFoundError(f"Hyperparameters file not found: {hyperparams_path}")
    
    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)
    
    print("Model hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Load data
    print(f"\nLoading data from: {args.data_file}")
    if args.data_file.endswith('.npy'):
        data = np.load(args.data_file)
    elif args.data_file.endswith('.txt'):
        data = np.genfromtxt(args.data_file, delimiter=',', dtype=np.float32)
    else:
        raise ValueError("Data file must be .npy or .txt format")
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{np.min(data):.6f}, {np.max(data):.6f}]")
    
    # Load model
    model_path = os.path.join(args.model_dir, "model.pt")
    state_dict_path = os.path.join(args.model_dir, args.state_dict)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path, state_dict_path, device, hyperparams)
    
    # Predict anomalies
    scores = predict_anomalies(data, model, hyperparams, device, 
                              args.batch_size, args.window_sliding)
    
    # Determine threshold
    if args.custom_threshold is not None:
        threshold = args.custom_threshold
        print(f"Using custom threshold: {threshold}")
    else:
        threshold = find_optimal_threshold(scores, method=args.threshold_method, 
                                         quantile=args.threshold_quantile)
    
    # Detect anomalies
    anomalies = detect_anomalies(scores, threshold)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    data_name = os.path.splitext(os.path.basename(args.data_file))[0]
    model_name = os.path.basename(args.model_dir)
    output_base = os.path.join(args.output_dir, f"{data_name}_{model_name}")
    
    # Save results
    save_results(scores, anomalies, threshold, output_base)
    
    # Visualization
    if args.visualize or args.save_plots:
        plot_path = f"{output_base}_plot.png" if args.save_plots else None
        visualize_results(data, scores, anomalies, threshold, plot_path)
    
    print(f"\nPrediction completed!")
    print(f"Detected {np.sum(anomalies)} anomalies out of {len(anomalies)} data points")
    print(f"Anomaly rate: {np.sum(anomalies) / len(anomalies):.4f}")


if __name__ == "__main__":
    main()
