#!/usr/bin/env python3

import os
import pandas as pd

def load_evaluation_results(RESULTS_DIR, VISUALIZATIONS_DIR):
    """Load evaluation results from results and visualizations directories
    
    Args:
        RESULTS_DIR (str): Path to the results directory
        VISUALIZATIONS_DIR (str): Path to the visualizations directory
        
    Returns:
        dict: Dictionary of evaluation results
    """
    results = {}

    # Load classification reports
    if os.path.exists(f"{RESULTS_DIR}/classification_reports.csv"):
        results['reports'] = pd.read_csv(f"{RESULTS_DIR}/classification_reports.csv")

    # Load images from visualizations directory first, then try results directory as fallback
    image_files = {
        'roc': ["roc_curves.png", "dnn_roc_curve.png", "svm_roc_curve.png", "logistic_roc_curve.png"],
        'cm': ["confusion_matrices.png", "confusion_matrix_deep_neural_network.png", "confusion_matrix_svm.png", 
               "confusion_matrix_logistic_regression.png", "confusion_matrix_perceptron.png"],
        'metrics': ["metrics_comparison.png", "model_metrics_comparison.png", "model_comparison.png", "model_radar_comparison.png"]
    }

    # Try to find each type of visualization
    for key, filenames in image_files.items():
        for filename in filenames:
            # First check visualizations directory
            viz_path = os.path.join(VISUALIZATIONS_DIR, filename)
            if os.path.exists(viz_path):
                results[key] = viz_path
                break
                
            # Then check results directory as fallback
            results_path = os.path.join(RESULTS_DIR, filename)
            if os.path.exists(results_path):
                results[key] = results_path
                break

    return results
