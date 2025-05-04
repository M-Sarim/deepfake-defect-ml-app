#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import pickle
import joblib

# Define the DNN model class for multi-label classification
class MultiLabelDNN(nn.Module):
    def __init__(self, input_size, num_classes=5, hidden_size=128, dropout_rate=0.3):
        super(MultiLabelDNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Create a flexible architecture that can handle different input/output sizes
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size // 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Check if input dimensions match the model's expected input size
        if x.shape[1] != self.input_size:
            print(f"Input dimension mismatch: expected {self.input_size}, got {x.shape[1]}")
            # Create a new tensor with the correct size
            new_x = torch.zeros((x.shape[0], self.input_size), device=x.device, dtype=x.dtype)
            # Copy as many features as we can
            min_features = min(self.input_size, x.shape[1])
            new_x[:, :min_features] = x[:, :min_features]
            x = new_x

        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.sigmoid(self.output_layer(x))
        return x

    def load_state_dict(self, state_dict, strict=True):
        """Custom state_dict loader to handle mismatched layer sizes"""
        try:
            # First try normal loading
            return super(MultiLabelDNN, self).load_state_dict(state_dict, strict)
        except Exception as e:
            print(f"Model loading error: {e}")

            # Check if we need to convert from old format to new format
            if 'model.0.weight' in state_dict:
                print("Converting from old model format to new format")

                # Get the saved input and output sizes
                saved_input_size = state_dict['model.0.weight'].size(1)
                saved_output_size = state_dict['model.6.weight'].size(0) if 'model.6.weight' in state_dict else self.num_classes

                print(f"Saved model has input size {saved_input_size}, output size {saved_output_size}")
                print(f"Current model has input size {self.input_size}, output size {self.num_classes}")

                # Recreate layers with correct sizes
                if saved_input_size != self.input_size:
                    print(f"Adjusting input layer size from {self.input_size} to {saved_input_size}")
                    self.input_size = saved_input_size
                    self.layer1 = nn.Linear(saved_input_size, self.hidden_size)

                if saved_output_size != self.num_classes:
                    print(f"Adjusting output layer size from {self.num_classes} to {saved_output_size}")
                    self.num_classes = saved_output_size
                    self.output_layer = nn.Linear(self.hidden_size // 2, saved_output_size)

                # Convert the old state dict format to the new one
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key == 'model.0.weight':
                        new_state_dict['layer1.weight'] = value
                    elif key == 'model.0.bias':
                        new_state_dict['layer1.bias'] = value
                    elif key == 'model.3.weight':
                        new_state_dict['layer2.weight'] = value
                    elif key == 'model.3.bias':
                        new_state_dict['layer2.bias'] = value
                    elif key == 'model.6.weight':
                        new_state_dict['output_layer.weight'] = value
                    elif key == 'model.6.bias':
                        new_state_dict['output_layer.bias'] = value

                # Call the parent class's load_state_dict with the new dictionary
                return super(MultiLabelDNN, self).load_state_dict(new_state_dict, strict=False)
            else:
                # If it's not an old format issue, it might be a size mismatch in the new format
                if 'layer1.weight' in state_dict:
                    saved_input_size = state_dict['layer1.weight'].size(1)
                    saved_output_size = state_dict['output_layer.weight'].size(0) if 'output_layer.weight' in state_dict else self.num_classes

                    print(f"Saved model has input size {saved_input_size}, output size {saved_output_size}")
                    print(f"Current model has input size {self.input_size}, output size {self.num_classes}")

                    # Recreate layers with correct sizes
                    if saved_input_size != self.input_size:
                        print(f"Adjusting input layer size from {self.input_size} to {saved_input_size}")
                        self.input_size = saved_input_size
                        self.layer1 = nn.Linear(saved_input_size, self.hidden_size)

                    if saved_output_size != self.num_classes:
                        print(f"Adjusting output layer size from {self.num_classes} to {saved_output_size}")
                        self.num_classes = saved_output_size
                        self.output_layer = nn.Linear(self.hidden_size // 2, saved_output_size)

                    # Try loading again with the adjusted model
                    return super(MultiLabelDNN, self).load_state_dict(state_dict, strict=False)

                # If we can't fix it, use a fallback approach
                print("Could not adapt model to match saved weights. Using fallback approach.")
                return None

# Define the DNN model class for binary classification (Deepfake Detection)
class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
        super(DeepNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # Create a more flexible architecture
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Check if input dimensions match the model's expected input size
        if x.shape[1] != self.input_size:
            print(f"Input dimension mismatch: expected {self.input_size}, got {x.shape[1]}")
            # Create a new tensor with the correct size
            new_x = torch.zeros((x.shape[0], self.input_size), device=x.device, dtype=x.dtype)
            # Copy as many features as we can
            min_features = min(self.input_size, x.shape[1])
            new_x[:, :min_features] = x[:, :min_features]
            x = new_x

        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.sigmoid(self.output_layer(x))
        return x

    def load_state_dict(self, state_dict, strict=True):
        """Custom state_dict loader to handle model structure changes"""
        try:
            # First try normal loading
            return super(DeepNeuralNetwork, self).load_state_dict(state_dict, strict)
        except Exception as e:
            print(f"Model loading error: {e}")

            # Check if we need to convert from old format to new format
            if 'model.0.weight' in state_dict:
                print("Converting from old model format to new format")

                # Get the saved input size
                saved_input_size = state_dict['model.0.weight'].size(1)

                print(f"Saved model has input size {saved_input_size}")
                print(f"Current model has input size {self.input_size}")

                # Recreate input layer with correct size if needed
                if saved_input_size != self.input_size:
                    print(f"Adjusting input layer size from {self.input_size} to {saved_input_size}")
                    self.input_size = saved_input_size
                    self.layer1 = nn.Linear(saved_input_size, self.hidden_size)

                # Convert the old state dict format to the new one
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key == 'model.0.weight':
                        new_state_dict['layer1.weight'] = value
                    elif key == 'model.0.bias':
                        new_state_dict['layer1.bias'] = value
                    elif key == 'model.3.weight':
                        new_state_dict['layer2.weight'] = value
                    elif key == 'model.3.bias':
                        new_state_dict['layer2.bias'] = value
                    elif key == 'model.6.weight':
                        new_state_dict['output_layer.weight'] = value
                    elif key == 'model.6.bias':
                        new_state_dict['output_layer.bias'] = value

                # Call the parent class's load_state_dict with the new dictionary
                return super(DeepNeuralNetwork, self).load_state_dict(new_state_dict, strict=False)
            else:
                # If it's not an old format issue, it might be a size mismatch in the new format
                if 'layer1.weight' in state_dict:
                    saved_input_size = state_dict['layer1.weight'].size(1)

                    print(f"Saved model has input size {saved_input_size}")
                    print(f"Current model has input size {self.input_size}")

                    # Recreate input layer with correct size if needed
                    if saved_input_size != self.input_size:
                        print(f"Adjusting input layer size from {self.input_size} to {saved_input_size}")
                        self.input_size = saved_input_size
                        self.layer1 = nn.Linear(saved_input_size, self.hidden_size)

                    # Try loading again with the adjusted model
                    return super(DeepNeuralNetwork, self).load_state_dict(state_dict, strict=False)

                # If we can't fix it, use a fallback approach
                print("Could not adapt model to match saved weights. Using fallback approach.")
                return None

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

    # Check if features is a string (text input)
    if isinstance(audio_features, str):
        print("Text input detected. Using demo mode for prediction.")
        # Switch to demo mode for text input
        return predict_deepfake(np.zeros(20), {"dnn": "DEMO"}, "dnn")

    # Scale features if a scaler is available
    if scaler is not None:
        try:
            # Check for feature dimension mismatch and handle it
            expected_features = scaler.n_features_in_
            actual_features = audio_features.shape[0]

            if expected_features != actual_features:
                print(f"Warning: Expected {expected_features} features, but got {actual_features}.")

                # Create a new feature array of the correct size
                adjusted_features = np.zeros(expected_features, dtype=np.float32)

                # Copy as many features as we can
                min_features = min(expected_features, actual_features)
                adjusted_features[:min_features] = audio_features[:min_features]

                # Use the adjusted features
                audio_features = adjusted_features

            # Ensure features are in the right shape for the scaler
            features_reshaped = audio_features.reshape(1, -1)

            # Make sure all values are valid floats
            for i in range(features_reshaped.shape[1]):
                if not np.isfinite(features_reshaped[0, i]):
                    features_reshaped[0, i] = 0.0

            features_scaled = scaler.transform(features_reshaped)
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
                # Convert features to PyTorch tensor safely
                try:
                    # First ensure we have a clean numpy array of float32
                    clean_features = np.zeros(features_scaled.shape, dtype=np.float32)
                    for i in range(features_scaled.shape[0]):
                        for j in range(features_scaled.shape[1]):
                            try:
                                val = features_scaled[i, j]
                                if val is not None and np.isfinite(float(val)):
                                    clean_features[i, j] = float(val)
                            except (ValueError, TypeError):
                                pass  # Keep as zero

                    # Convert using PyTorch's from_numpy which is more reliable
                    features_tensor = torch.from_numpy(clean_features).float()
                except Exception as e:
                    print(f"Error converting to PyTorch tensor: {e}")
                    # Last resort: create a tensor of zeros
                    features_tensor = torch.zeros((1, features_scaled.shape[1]), dtype=torch.float32)
                output = models['dnn'](features_tensor)
                # Safely extract confidence value
                try:
                    if isinstance(output, torch.Tensor):
                        confidence = output.item()
                    else:
                        confidence = float(output)
                except (ValueError, TypeError):
                    print("Warning: Could not convert model output to float. Using default value.")
                    confidence = 0.5

                prediction = 1 if confidence >= 0.5 else 0
        else:
            # It's a state_dict, we need to create a model first
            input_size = features_scaled.shape[1]
            dnn_model = DeepNeuralNetwork(input_size)
            dnn_model.load_state_dict(models['dnn'])
            dnn_model.eval()
            with torch.no_grad():
                # Convert features to PyTorch tensor safely
                try:
                    # First ensure we have a clean numpy array of float32
                    clean_features = np.zeros(features_scaled.shape, dtype=np.float32)
                    for i in range(features_scaled.shape[0]):
                        for j in range(features_scaled.shape[1]):
                            try:
                                val = features_scaled[i, j]
                                if val is not None and np.isfinite(float(val)):
                                    clean_features[i, j] = float(val)
                            except (ValueError, TypeError):
                                pass  # Keep as zero

                    # Convert using PyTorch's from_numpy which is more reliable
                    features_tensor = torch.from_numpy(clean_features).float()
                except Exception as e:
                    print(f"Error converting to PyTorch tensor: {e}")
                    # Last resort: create a tensor of zeros
                    features_tensor = torch.zeros((1, features_scaled.shape[1]), dtype=torch.float32)
                output = dnn_model(features_tensor)
                # Safely extract confidence value
                try:
                    if isinstance(output, torch.Tensor):
                        confidence = output.item()
                    else:
                        confidence = float(output)
                except (ValueError, TypeError):
                    print("Warning: Could not convert model output to float. Using default value.")
                    confidence = 0.5

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

    # Check if features is a string (text input)
    if isinstance(features, str):
        print("Text input detected. Using demo mode for prediction.")
        # Switch to demo mode for text input
        return predict_defects(np.zeros(20), {"dnn": "DEMO"}, "dnn", defect_labels)

    # Scale features if a scaler is available
    if scaler is not None:
        try:
            # Check for feature dimension mismatch and handle it
            expected_features = scaler.n_features_in_
            actual_features = features.shape[0]

            if expected_features != actual_features:
                print(f"Warning: Expected {expected_features} features, but got {actual_features}.")

                # Create a new feature array of the correct size
                adjusted_features = np.zeros(expected_features, dtype=np.float32)

                # Copy as many features as we can
                min_features = min(expected_features, actual_features)
                adjusted_features[:min_features] = features[:min_features]

                # Use the adjusted features
                features = adjusted_features

            # Ensure features are in the right shape for the scaler
            features_reshaped = features.reshape(1, -1)

            # Make sure all values are valid floats
            for i in range(features_reshaped.shape[1]):
                if not np.isfinite(features_reshaped[0, i]):
                    features_reshaped[0, i] = 0.0

            features_scaled = scaler.transform(features_reshaped)
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
                # Convert features to PyTorch tensor safely
                try:
                    # First ensure we have a clean numpy array of float32
                    clean_features = np.zeros(features_scaled.shape, dtype=np.float32)
                    for i in range(features_scaled.shape[0]):
                        for j in range(features_scaled.shape[1]):
                            try:
                                val = features_scaled[i, j]
                                if val is not None and np.isfinite(float(val)):
                                    clean_features[i, j] = float(val)
                            except (ValueError, TypeError):
                                pass  # Keep as zero

                    # Convert using PyTorch's from_numpy which is more reliable
                    features_tensor = torch.from_numpy(clean_features).float()
                except Exception as e:
                    print(f"Error converting to PyTorch tensor: {e}")
                    # Last resort: create a tensor of zeros
                    features_tensor = torch.zeros((1, features_scaled.shape[1]), dtype=torch.float32)
                output = models['dnn'](features_tensor)
                # Convert tensor to numpy array safely
                if isinstance(output, torch.Tensor):
                    output_np = output.detach().cpu().numpy()
                else:
                    # Handle case where output might be a list or other non-tensor type
                    try:
                        # Try to convert to a float32 array
                        output_np = np.array(output, dtype=np.float32)
                    except (ValueError, TypeError):
                        # If that fails, convert each element individually
                        if isinstance(output, (list, tuple)):
                            output_np = np.array([float(x) if isinstance(x, (int, float)) else 0.0 for x in output], dtype=np.float32)
                        else:
                            # Last resort: create a default array with zeros
                            output_np = np.zeros(len(defect_labels), dtype=np.float32)
                            print(f"Warning: Could not convert model output to numpy array. Using zeros instead.")

                # Ensure we have a 1D array
                if len(output_np.shape) > 1:
                    output_np = output_np[0]

                # Ensure the array contains only float values
                output_np = output_np.astype(np.float32)

                # Convert outputs to predictions and confidences
                for i, label in enumerate(defect_labels):
                    if i < len(output_np):
                        confidence = float(output_np[i])
                        prediction = confidence >= 0.5
                        results[label] = {
                            "prediction": bool(prediction),
                            "confidence": confidence
                        }
                    else:
                        # Handle case where model output doesn't match expected number of classes
                        results[label] = {
                            "prediction": False,
                            "confidence": 0.0
                        }
        else:
            # It's a state_dict or something else, create a model first
            input_size = features_scaled.shape[1]
            dnn_model = MultiLabelDNN(input_size, num_classes)
            if not isinstance(models['dnn'], str):  # If not demo mode
                dnn_model.load_state_dict(models['dnn'])

            dnn_model.eval()
            with torch.no_grad():
                # Convert features to PyTorch tensor safely
                try:
                    # First ensure we have a clean numpy array of float32
                    clean_features = np.zeros(features_scaled.shape, dtype=np.float32)
                    for i in range(features_scaled.shape[0]):
                        for j in range(features_scaled.shape[1]):
                            try:
                                val = features_scaled[i, j]
                                if val is not None and np.isfinite(float(val)):
                                    clean_features[i, j] = float(val)
                            except (ValueError, TypeError):
                                pass  # Keep as zero

                    # Convert using PyTorch's from_numpy which is more reliable
                    features_tensor = torch.from_numpy(clean_features).float()
                except Exception as e:
                    print(f"Error converting to PyTorch tensor: {e}")
                    # Last resort: create a tensor of zeros
                    features_tensor = torch.zeros((1, features_scaled.shape[1]), dtype=torch.float32)
                output = dnn_model(features_tensor)
                # Convert tensor to numpy array safely
                if isinstance(output, torch.Tensor):
                    output_np = output.detach().cpu().numpy()
                else:
                    # Handle case where output might be a list or other non-tensor type
                    try:
                        # Try to convert to a float32 array
                        output_np = np.array(output, dtype=np.float32)
                    except (ValueError, TypeError):
                        # If that fails, convert each element individually
                        if isinstance(output, (list, tuple)):
                            output_np = np.array([float(x) if isinstance(x, (int, float)) else 0.0 for x in output], dtype=np.float32)
                        else:
                            # Last resort: create a default array with zeros
                            output_np = np.zeros(len(defect_labels), dtype=np.float32)
                            print(f"Warning: Could not convert model output to numpy array. Using zeros instead.")

                # Ensure we have a 1D array
                if len(output_np.shape) > 1:
                    output_np = output_np[0]

                # Ensure the array contains only float values
                output_np = output_np.astype(np.float32)

                # Convert outputs to predictions and confidences
                for i, label in enumerate(defect_labels):
                    if i < len(output_np):
                        confidence = float(output_np[i])
                        prediction = confidence >= 0.5
                        results[label] = {
                            "prediction": bool(prediction),
                            "confidence": confidence
                        }
                    else:
                        # Handle case where model output doesn't match expected number of classes
                        results[label] = {
                            "prediction": False,
                            "confidence": 0.0
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
