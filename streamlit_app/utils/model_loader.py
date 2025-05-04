#!/usr/bin/env python3

import os
import torch
from .model_utils import safe_load_pickle, DeepNeuralNetwork

def load_models(MODEL_DIR):
    """Load all available models from the model directory
    
    Args:
        MODEL_DIR (str): Path to the directory containing model files
        
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    load_errors = []

    # Check if model directory exists
    try:
        if not os.path.exists(MODEL_DIR):
            load_errors.append(f"Model directory not found: {MODEL_DIR}")
            return models
    except Exception as e:
        load_errors.append(f"Error accessing model directory: {e}")
        return models

    # Try to load Logistic Regression model
    if os.path.exists(f"{MODEL_DIR}/logistic_deepfake.pkl"):
        lr_model = safe_load_pickle(f"{MODEL_DIR}/logistic_deepfake.pkl", "Logistic Regression", load_errors)
        if lr_model is not None:
            models['lr'] = lr_model

            # Load the corresponding scaler
            if os.path.exists(f"{MODEL_DIR}/logistic_deepfake_scaler.pkl"):
                lr_scaler = safe_load_pickle(f"{MODEL_DIR}/logistic_deepfake_scaler.pkl", "Logistic Regression scaler", load_errors)
                if lr_scaler is not None:
                    models['lr_scaler'] = lr_scaler

    # Try to load SVM model
    if os.path.exists(f"{MODEL_DIR}/svm_deepfake.pkl"):
        svm_model = safe_load_pickle(f"{MODEL_DIR}/svm_deepfake.pkl", "SVM", load_errors)
        if svm_model is not None:
            models['svm'] = svm_model

            # Load the corresponding scaler
            if os.path.exists(f"{MODEL_DIR}/svm_deepfake_scaler.pkl"):
                svm_scaler = safe_load_pickle(f"{MODEL_DIR}/svm_deepfake_scaler.pkl", "SVM scaler", load_errors)
                if svm_scaler is not None:
                    models['svm_scaler'] = svm_scaler

    # Try to load Perceptron model
    if os.path.exists(f"{MODEL_DIR}/perceptron_deepfake.pkl"):
        perceptron_model = safe_load_pickle(f"{MODEL_DIR}/perceptron_deepfake.pkl", "Perceptron", load_errors)
        if perceptron_model is not None:
            models['perceptron'] = perceptron_model

            # Load the corresponding scaler
            if os.path.exists(f"{MODEL_DIR}/perceptron_deepfake_scaler.pkl"):
                perceptron_scaler = safe_load_pickle(f"{MODEL_DIR}/perceptron_deepfake_scaler.pkl", "Perceptron scaler", load_errors)
                if perceptron_scaler is not None:
                    models['perceptron_scaler'] = perceptron_scaler

    # Try to load DNN model
    if os.path.exists(f"{MODEL_DIR}/dnn_deepfake.pt"):
        try:
            # Try to load with map_location to ensure it loads on CPU if needed
            models['dnn'] = torch.load(f"{MODEL_DIR}/dnn_deepfake.pt", map_location=torch.device('cpu'))
        except Exception as e:
            try:
                # Alternative approach: Create a new model instance and load state_dict
                checkpoint = torch.load(f"{MODEL_DIR}/dnn_deepfake.pt", map_location=torch.device('cpu'))

                # Check if we have model info available
                if os.path.exists(f"{MODEL_DIR}/dnn_deepfake_info.pkl"):
                    model_info = safe_load_pickle(f"{MODEL_DIR}/dnn_deepfake_info.pkl", "DNN info", load_errors)
                    if model_info is not None:
                        input_size = model_info.get('input_size', 20)  # Default to 20 if not found

                        # Initialize model with the correct input size
                        dnn_model = DeepNeuralNetwork(input_size)
                        dnn_model.load_state_dict(checkpoint)
                        models['dnn'] = dnn_model
                else:
                    # Fallback to default parameters
                    input_size = 20  # This should match the feature size used during training
                    dnn_model = DeepNeuralNetwork(input_size)
                    dnn_model.load_state_dict(checkpoint)
                    models['dnn'] = dnn_model
            except Exception as e2:
                # Log error but don't display it in the UI
                load_errors.append(f"Failed to load DNN model: {e}, Alternative method error: {e2}")

        # Load the corresponding scaler
        if os.path.exists(f"{MODEL_DIR}/dnn_deepfake_scaler.pkl"):
            dnn_scaler = safe_load_pickle(f"{MODEL_DIR}/dnn_deepfake_scaler.pkl", "DNN scaler", load_errors)
            if dnn_scaler is not None:
                models['dnn_scaler'] = dnn_scaler

    # Use a common scaler for all models if available, or use model-specific scalers
    if 'lr_scaler' in models:
        models['scaler'] = models['lr_scaler']
    elif 'svm_scaler' in models:
        models['scaler'] = models['svm_scaler']
    elif 'perceptron_scaler' in models:
        models['scaler'] = models['perceptron_scaler']
    elif 'dnn_scaler' in models:
        models['scaler'] = models['dnn_scaler']

    # If no models are found, provide dummy models for demo purposes
    if not models or len(models) <= 1:
        models['lr'] = "DEMO"
        models['svm'] = "DEMO"
        models['dnn'] = "DEMO"
        models['scaler'] = None

    return models
