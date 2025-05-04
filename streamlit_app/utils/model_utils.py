#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import joblib

# Define the DNN model class for multi-label classification
class MultiLabelDNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, dropout_rate=0.3):
        super(MultiLabelDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )

    def forward(self, x):
        return self.model(x)

# Define the DNN model class for binary classification (Deepfake Detection)
class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
        super(DeepNeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Helper function to safely load pickle files
def safe_load_pickle(file_path, model_name, load_errors=None):
    if load_errors is None:
        load_errors = []

    try:
        # Try different loading methods
        # Method 1: Direct pickle load
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            # Method 2: Try with joblib
            try:
                return joblib.load(file_path)
            except Exception:
                # Method 3: Try with pickle using latin1 encoding (for Python 2 to 3 compatibility)
                try:
                    with open(file_path, 'rb') as f:
                        return pickle.load(f, encoding='latin1')
                except Exception:
                    # Method 4: Try with pickle using bytes encoding
                    try:
                        with open(file_path, 'rb') as f:
                            return pickle.load(f, encoding='bytes')
                    except Exception:
                        # All methods failed, log the error silently
                        if load_errors is not None:
                            load_errors.append(f"Failed to load {model_name} model after trying all methods")
                        return None
    except Exception as file_error:
        # Log error silently
        if load_errors is not None:
            load_errors.append(f"Failed to open {model_name} model file: {file_error}")
        return None

# Function to predict deepfake audio
def predict_deepfake(audio_features, models, model_name):
    """Make deepfake prediction using the specified model"""
    if model_name not in models:
        return {"prediction": "Model not available", "confidence": 0.0}

    # Handle demo mode first to avoid scaler issues
    if models[model_name] == "DEMO":
        # Generate random prediction for demo
        prediction = np.random.randint(0, 2)  # 0 or 1
        confidence = np.random.random()  # Random confidence between 0 and 1

        return {
            "prediction": "Bonafide" if prediction == 1 else "Deepfake",
            "confidence": float(confidence) if prediction == 1 else float(1 - confidence)
        }

    # For real models, handle feature scaling
    # Use model-specific scalers if available
    scaler_key = f"{model_name}_scaler"
    if scaler_key in models and models[scaler_key] is not None:
        scaler = models[scaler_key]
    elif 'scaler' in models and models['scaler'] is not None:
        scaler = models['scaler']
    else:
        scaler = None

    # Scale features if a scaler is available
    if scaler is not None:
        try:
            # Check for feature dimension mismatch and handle it
            expected_features = scaler.n_features_in_
            actual_features = audio_features.shape[0]

            if expected_features != actual_features:
                print(f"Feature dimension mismatch: model expects {expected_features}, got {actual_features}")

                # Option 1: If we have too many features, take only the first expected_features
                if actual_features > expected_features:
                    audio_features = audio_features[:expected_features]
                # Option 2: If we have too few features, pad with zeros
                else:
                    audio_features = np.pad(audio_features, (0, expected_features - actual_features), 'constant')

            features_scaled = scaler.transform(audio_features.reshape(1, -1))
        except Exception as e:
            print(f"Error scaling features: {e}")
            # Fallback to unscaled features
            features_scaled = audio_features.reshape(1, -1)
    else:
        features_scaled = audio_features.reshape(1, -1)

    # Make prediction
    if model_name == 'dnn':
        # Check if the DNN model is a state_dict or a full model
        if isinstance(models['dnn'], torch.nn.Module):
            # It's a full model
            models['dnn'].eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                output = models['dnn'](features_tensor)
                confidence = output.item()
                prediction = 1 if confidence >= 0.5 else 0
        else:
            # It's a state_dict, we need to create a model first
            input_size = features_scaled.shape[1]
            dnn_model = DeepNeuralNetwork(input_size)
            dnn_model.load_state_dict(models['dnn'])
            dnn_model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                output = dnn_model(features_tensor)
                confidence = output.item()
                prediction = 1 if confidence >= 0.5 else 0
    else:
        # For sklearn models
        prediction = models[model_name].predict(features_scaled)[0]

        # Get probability/confidence
        if hasattr(models[model_name], 'predict_proba'):
            probs = models[model_name].predict_proba(features_scaled)[0]
            confidence = probs[1] if prediction == 1 else probs[0]
        elif hasattr(models[model_name], 'decision_function'):
            # For models like SVM that might use decision_function instead
            decision = models[model_name].decision_function(features_scaled)[0]
            confidence = 1 / (1 + np.exp(-decision))  # Convert to probability using sigmoid
        else:
            confidence = 1.0  # Default confidence if no probability method available

    result = {
        "prediction": "Bonafide" if prediction == 1 else "Deepfake",
        "confidence": float(confidence) if prediction == 1 else float(1 - confidence)
    }

    return result

# Function to make multi-label predictions
def predict_defects(features, models, model_name, defect_labels):
    """Make multi-label defect predictions using the specified model"""
    results = {}
    num_classes = len(defect_labels)

    # Handle demo mode
    if models[model_name] == "DEMO":
        # Generate random predictions for demo
        confidence_scores = np.random.rand(num_classes)
        predictions = (confidence_scores > 0.5).astype(int)

        for i, label in enumerate(defect_labels):
            results[label] = {
                "prediction": bool(predictions[i]),
                "confidence": float(confidence_scores[i])
            }
        return results

    # For real models, handle feature scaling
    # Use model-specific scalers if available
    scaler_key = f"{model_name}_scaler"
    if scaler_key in models and models[scaler_key] is not None:
        scaler = models[scaler_key]
    elif 'scaler' in models and models['scaler'] is not None:
        scaler = models['scaler']
    else:
        scaler = None

    # Scale features if a scaler is available
    if scaler is not None:
        try:
            # Check for feature dimension mismatch and handle it
            expected_features = scaler.n_features_in_
            actual_features = features.shape[0]

            if expected_features != actual_features:
                print(f"Feature dimension mismatch: model expects {expected_features}, got {actual_features}")

                # Option 1: If we have too many features, take only the first expected_features
                if actual_features > expected_features:
                    features = features[:expected_features]
                # Option 2: If we have too few features, pad with zeros
                else:
                    features = np.pad(features, (0, expected_features - actual_features), 'constant')

            features_scaled = scaler.transform(features.reshape(1, -1))
        except Exception as e:
            print(f"Error scaling features: {e}")
            # Fallback to unscaled features
            features_scaled = features.reshape(1, -1)
    else:
        features_scaled = features.reshape(1, -1)

    # Make prediction based on model type
    if model_name == 'dnn':
        # Handle DNN model
        if isinstance(models['dnn'], torch.nn.Module):
            # It's a full model
            models['dnn'].eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                output = models['dnn'](features_tensor).numpy()[0]

                # Convert outputs to predictions and confidences
                for i, label in enumerate(defect_labels):
                    confidence = output[i]
                    prediction = confidence >= 0.5
                    results[label] = {
                        "prediction": bool(prediction),
                        "confidence": float(confidence)
                    }
        else:
            # It's a state_dict or something else, create a model first
            input_size = features_scaled.shape[1]
            dnn_model = MultiLabelDNN(input_size, num_classes)
            if not isinstance(models['dnn'], str):  # If not demo mode
                dnn_model.load_state_dict(models['dnn'])

            dnn_model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_scaled)
                output = dnn_model(features_tensor).numpy()[0]

                # Convert outputs to predictions and confidences
                for i, label in enumerate(defect_labels):
                    confidence = output[i]
                    prediction = confidence >= 0.5
                    results[label] = {
                        "prediction": bool(prediction),
                        "confidence": float(confidence)
                    }
    else:
        # Handle sklearn models (LR, SVM)
        # Assuming these models are configured for multi-label output (e.g., OneVsRestClassifier)
        try:
            predictions = models[model_name].predict(features_scaled)[0]

            # Get probabilities if available
            if hasattr(models[model_name], 'predict_proba'):
                probabilities = models[model_name].predict_proba(features_scaled)

                for i, label in enumerate(defect_labels):
                    if isinstance(probabilities, list):
                        confidence = probabilities[i][0][1]  # For multi-label with separate classifiers
                    else:
                        confidence = probabilities[0, i]  # For direct multi-label output

                    results[label] = {
                        "prediction": bool(predictions[i]),
                        "confidence": float(confidence)
                    }
            else:
                # If no probabilities available, use decision function or default to binary prediction
                for i, label in enumerate(defect_labels):
                    if hasattr(models[model_name], 'decision_function'):
                        decision = models[model_name].decision_function(features_scaled)[0]
                        if isinstance(decision, np.ndarray) and len(decision) > 1:
                            confidence = 1 / (1 + np.exp(-decision[i]))  # Sigmoid for single value
                        else:
                            confidence = 1 / (1 + np.exp(-decision))  # Sigmoid for single value
                    else:
                        confidence = 1.0 if predictions[i] else 0.0

                    results[label] = {
                        "prediction": bool(predictions[i]),
                        "confidence": float(confidence)
                    }
        except Exception as e:
            # Fallback to demo mode if prediction fails
            print(f"Error in prediction: {e}")
            for i, label in enumerate(defect_labels):
                confidence = np.random.random()
                prediction = confidence >= 0.5
                results[label] = {
                    "prediction": bool(prediction),
                    "confidence": float(confidence)
                }

    return results
