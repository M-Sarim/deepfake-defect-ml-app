#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(feature_names, feature_values):
    """Create a horizontal bar chart of feature values

    Args:
        feature_names (list): List of feature names
        feature_values (numpy.ndarray or list): Array of feature values

    Returns:
        matplotlib.figure.Figure: Figure with feature importance visualization
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert feature_values to numeric if they're strings
    numeric_values = []
    for val in feature_values:
        try:
            numeric_values.append(float(val))
        except (ValueError, TypeError):
            # If conversion fails, use 0 as a fallback
            numeric_values.append(0.0)

    feature_values = np.array(numeric_values)

    # Sort features by absolute value
    indices = np.argsort(np.abs(feature_values))

    # Plot horizontal bars
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, feature_values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])

    # Add labels and title
    ax.set_xlabel('Feature Value')
    ax.set_title('Feature Values Visualization')

    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

def plot_radar_chart(predictions):
    """Create a radar chart showing confidence for each defect type

    Args:
        predictions (dict): Dictionary of predictions with confidence scores

    Returns:
        matplotlib.figure.Figure: Figure with radar chart visualization
    """
    # Extract confidence scores
    labels = list(predictions.keys())
    values = [predictions[label]["confidence"] for label in labels]

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # Set the angles for each label
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]  # Close the loop
    angles += angles[:1]  # Close the loop
    labels += labels[:1]  # Close the labels loop

    # Plot values
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    # Set labels
    ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])

    # Set y-axis limits
    ax.set_ylim(0, 1)

    # Add grid and labels
    ax.grid(True)
    plt.title('Defect Prediction Confidence', size=15)

    return fig

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Create a confusion matrix visualization

    Args:
        cm (numpy.ndarray): Confusion matrix array
        class_names (list): List of class names
        title (str): Title for the plot

    Returns:
        matplotlib.figure.Figure: Figure with confusion matrix visualization
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)

    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    plt.tight_layout()
    return fig
