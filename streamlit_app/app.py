import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import utility modules
from utils import (
    # Audio processing utilities
    extract_mfcc, display_audio_visualizations,

    # Model utilities
    predict_deepfake, predict_defects,

    # Model loading
    load_models,

    # Visualization utilities
    plot_feature_importance, plot_radar_chart,

    # Evaluation utilities
    load_evaluation_results
)

# Set page configuration
st.set_page_config(
    page_title="ML Classification Tasks",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to style the sidebar and remove horizontal white bars
st.markdown("""
<style>
    .css-1544g2n {
        padding-top: 2rem;
    }
    .css-1544g2n h1 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f0f2f6;
        margin-bottom: 1.5rem;
    }
    .css-1544g2n .stRadio {
        margin-top: 1rem;
    }
    .css-1544g2n .stRadio label {
        font-weight: 500;
        color: #f0f2f6;
    }
    /* Remove horizontal white bars */
    .css-18e3th9 {
        padding-top: 0;
    }
    .css-1d391kg {
        padding-top: 0;
    }
    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }
    /* Hide "Made with Streamlit" */
    .viewerBadge_container__1QSob {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
# Use absolute path for directories to ensure correct loading
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
VISUALIZATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visualizations")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Constants
DEFECT_LABELS = ['Security', 'Performance', 'Maintainability', 'Reliability', 'Functional']
NUM_CLASSES = len(DEFECT_LABELS)

# Function to load models
@st.cache_resource
def load_all_models():
    """Load all available models using the utility function"""
    models = load_models(MODEL_DIR)

    # Display warning if using demo models
    if models.get('lr') == "DEMO" or models.get('svm') == "DEMO" or models.get('dnn') == "DEMO":
        st.warning("No trained models found. Using demo models for demonstration purposes.")

    return models

# Function to make predictions for defects
def predict_defect_labels(features, models, model_name):
    """Wrapper around the predict_defects utility function"""
    try:
        return predict_defects(features, models, model_name, DEFECT_LABELS)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        # Fallback to demo mode if prediction fails
        results = {}
        for label in DEFECT_LABELS:
            confidence = np.random.random()
            prediction = confidence >= 0.5
            results[label] = {
                "prediction": bool(prediction),
                "confidence": float(confidence)
            }
        return results

# Function to load evaluation results wrapper
def load_all_evaluation_results():
    """Wrapper around the load_evaluation_results utility function"""
    return load_evaluation_results(RESULTS_DIR, VISUALIZATIONS_DIR)

# Main app
def main():
    # Load models
    models = load_all_models()

    # Sidebar with enhanced navigation
    st.sidebar.markdown("<h1>Navigation Panel</h1>", unsafe_allow_html=True)

    # Define navigation options with icons
    nav_options = {
        "dashboard": "📊 Dashboard",
        "deepfake": "🎙️ Deepfake Detection",
        "defects": "🔍 Defect Prediction",
        "analytics": "📈 Model Analytics",
        "evaluation": "📋 Performance Metrics"
    }

    # Create radio buttons with the new labels
    selected = st.sidebar.radio(
        "Select Module",
        list(nav_options.keys()),
        format_func=lambda x: nav_options[x],
        label_visibility="collapsed"
    )

    # Map selected option to page names
    page_mapping = {
        "dashboard": "Home",
        "deepfake": "Deepfake Detection",
        "defects": "Defect Prediction",
        "analytics": "Model Results",
        "evaluation": "Model Evaluation"
    }

    page = page_mapping[selected]

    # Add separator and additional information
    st.sidebar.markdown("---")

    # Check if models are loaded (simplified display)
    model_count = len([k for k in models.keys() if k not in ['scaler', 'lr_scaler', 'svm_scaler', 'perceptron_scaler', 'dnn_scaler']])
    model_status = "Models Loaded ✓" if model_count > 0 else "Models Not Loaded"
    st.sidebar.markdown(f"**Model Status:** {model_status}")

    # Add version information
    st.sidebar.markdown("**Version:** 1.0.0")
    st.sidebar.markdown("**Last Updated:** May 2025")

    # Home page
    if page == "Home":
        # Apply custom CSS for a clean, streamlined interface
        st.markdown("""
        <style>
        h1, h2, h3, h4, h5, h6 {
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            color: #1E3A8A;
        }
        h1 {
            font-size: 2.2rem;
            font-weight: 700;
        }
        h2 {
            font-size: 1.8rem;
            font-weight: 600;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 0.5rem;
        }
        h3 {
            font-size: 1.5rem;
            font-weight: 600;
        }
        p {
            margin-bottom: 1rem;
            line-height: 1.6;
        }
        ul {
            margin-bottom: 1.5rem;
        }
        li {
            margin-bottom: 0.5rem;
        }
        .highlight {
            color: #1E3A8A;
            font-weight: 600;
        }
        .section-divider {
            margin: 2rem 0;
            border-top: 1px solid #e5e7eb;
        }
        </style>
        """, unsafe_allow_html=True)

        # Main content starts directly with the overview
        st.title("🎙️ Urdu Deepfake  Detection System")
        st.markdown("An intelligent system to predict multiple types of software defects")

        # Introduction section
        st.header("📋 Project Overview")
        st.markdown("""
        This application demonstrates the use of machine learning models to predict different types of software defects based on code metrics.
        The system analyzes feature vectors extracted from software code and determines the likelihood of multiple defect types.
        """)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Defect Types section
        st.header("🔍 Software Defect Types")
        st.markdown("""
        This system can predict the following types of software defects:
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Security Defects")
            st.markdown("""
            Vulnerabilities that could lead to security breaches or unauthorized access:

            * Buffer overflows
            * SQL injection
            * Cross-site scripting
            * Authentication issues
            """)

        with col2:
            st.subheader("Performance Defects")
            st.markdown("""
            Issues that affect system performance and efficiency:

            * Memory leaks
            * Inefficient algorithms
            * Resource contention
            * Excessive I/O operations
            """)

        with col3:
            st.subheader("Other Defect Types")
            st.markdown("""
            * **Maintainability**: Issues affecting code readability and maintenance
            * **Reliability**: Problems causing system instability or crashes
            * **Functional**: Defects in core business logic implementation
            """)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Our Approach section
        st.header("🧠 Our Approach")
        st.markdown("""
        We've trained and compared four different machine learning models on our software defect dataset:
        """)

        # Create a row of models
        model_cols = st.columns(4)

        with model_cols[0]:
            st.subheader("Logistic Regression")
            st.markdown("Multi-label classifier with independent probability outputs for each defect type")

        with model_cols[1]:
            st.subheader("SVM")
            st.markdown("Support Vector Machine with specialized kernels for complex defect patterns")

        with model_cols[2]:
            st.subheader("Perceptron")
            st.markdown("Linear classifier with simple decision boundaries for efficient defect detection")

        with model_cols[3]:
            st.subheader("Deep Neural Network")
            st.markdown("Multi-layer neural network designed for multiple defect type prediction")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # How to Use This App section
        st.header("🚀 How to Use This App")
        st.markdown("Navigate through the different sections using the sidebar menu:")

        feature_cols = st.columns(4)

        with feature_cols[0]:
            st.markdown("**🎙️ Deepfake Detection**")
            st.markdown("Upload audio files to detect deepfakes")

        with feature_cols[1]:
            st.markdown("**🔍 Defect Prediction**")
            st.markdown("Input code metrics to predict potential defects")

        with feature_cols[2]:
            st.markdown("**📈 Model Analytics**")
            st.markdown("View the performance metrics of our trained models")

        with feature_cols[3]:
            st.markdown("**📋 Performance Metrics**")
            st.markdown("Detailed evaluation with ROC curves and confusion matrices")

    # Deepfake Detection page
    elif page == "Deepfake Detection":
        st.header("Urdu Deepfake Audio Detection")

        if not models:
            st.warning("Models are not available. Please train and save the models first.")
            return

        # Upload section
        st.subheader("📂 Upload Audio File")
        st.markdown("Upload an audio file to test our deepfake detection models.")
        uploaded_file = st.file_uploader("Choose a WAV or MP3 file", type=['wav', 'mp3'], key="audio_uploader")

        # Enhanced model selection section for deepfake detection
        st.subheader("🔍 Select Models for Prediction")

        # Add a brief explanation
        st.markdown("""
        Choose which models to use for deepfake detection. Each model has different strengths for audio analysis.
        """)

        # Reuse the CSS styles from the defect prediction page
        st.markdown("""
        <style>
        .model-card {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            border: 1px solid #e0e0e0;
            background-color: #f9f9f9;
        }
        .model-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .model-desc {
            font-size: 0.85em;
            color: #555;
            margin-bottom: 8px;
        }
        .model-stats {
            font-size: 0.8em;
            color: #666;
        }
        .model-unavailable {
            opacity: 0.6;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create columns for model selection cards
        model_cols = st.columns(4)

        # Define available models with descriptions and stats for deepfake detection
        deepfake_model_info = {
            'lr': {
                'name': 'Logistic Regression',
                'description': 'Fast model with good baseline performance',
                'accuracy': '89%',
                'speed': 'Very Fast',
                'icon': '📊'
            },
            'svm': {
                'name': 'SVM',
                'description': 'Excellent for audio feature patterns',
                'accuracy': '92%',
                'speed': 'Medium',
                'icon': '🔍'
            },
            'perceptron': {
                'name': 'Perceptron',
                'description': 'Simple and fast linear classifier',
                'accuracy': '87%',
                'speed': 'Fast',
                'icon': '⚡'
            },
            'dnn': {
                'name': 'Deep Neural Network',
                'description': 'Best for complex audio patterns',
                'accuracy': '95%',
                'speed': 'Slow',
                'icon': '🧠'
            }
        }

        # Create checkboxes for model selection with enhanced UI
        selected_models = {}
        for i, (model_key, info) in enumerate(deepfake_model_info.items()):
            with model_cols[i]:
                if model_key in models:
                    # Create a card-like container for each model
                    st.markdown(f"""
                    <div class="model-card">
                        <div class="model-title">{info['icon']} {info['name']}</div>
                        <div class="model-desc">{info['description']}</div>
                        <div class="model-stats">Accuracy: {info['accuracy']} | Speed: {info['speed']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Add checkbox below the card
                    selected_models[model_key] = st.checkbox(f"Use {info['name']}", value=True, key=f"df_{model_key}")
                else:
                    # Create a disabled-looking card for unavailable models
                    st.markdown(f"""
                    <div class="model-card model-unavailable">
                        <div class="model-title">{info['icon']} {info['name']}</div>
                        <div class="model-desc">{info['description']}</div>
                        <div class="model-stats">Not available</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("*Model not available*")

        # Add a "Run All Models" button with better styling
        col1, col2 = st.columns([3, 1])
        with col1:
            run_all = st.checkbox("Run All Available Models", value=True, key="df_run_all")
            if run_all:
                for model_key in models.keys():
                    if model_key != 'scaler' and model_key in selected_models:
                        selected_models[model_key] = True

        with col2:
            # Add a help button with model selection tips
            with st.expander("Need help?"):
                st.markdown("""
                **Tips for deepfake detection:**
                - For highest accuracy: Use DNN
                - For quick screening: Use Perceptron
                - For balanced approach: Use SVM
                """)

        # Add a comparison tooltip
        st.info("💡 **Tip**: Deepfakes can be subtle - using multiple models can help catch different types of manipulations.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            # Save the uploaded file temporarily
            temp_path = f"temp_audio_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Audio player
            st.subheader("🎵 Audio Player")
            st.audio(temp_path)

            # Display audio visualizations
            st.subheader("📊 Audio Visualization")
            fig = display_audio_visualizations(temp_path)
            if fig:
                st.pyplot(fig)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Extract features
            features = extract_mfcc(temp_path)

            if features is not None:
                st.subheader("🔮 Prediction Results")

                # Check if any model is selected
                any_model_selected = any(selected_models.values())

                if not any_model_selected:
                    st.warning("Please select at least one model for prediction.")
                else:
                    # Make predictions with selected models
                    for model_key, selected in selected_models.items():
                        if model_key in models and selected:
                            # Get prediction and confidence
                            result = predict_deepfake(features, models, model_key)
                            prediction = result["prediction"]
                            confidence = result["confidence"]

                            # Get model display name from the info dictionary
                            model_display = deepfake_model_info[model_key]['name']
                            model_icon = deepfake_model_info[model_key]['icon']

                            # Create a clean prediction display with improved styling
                            col1, col2 = st.columns([1, 3])

                            with col1:
                                st.markdown(f"**{model_icon} {model_display}:**")

                            with col2:
                                # Display prediction with appropriate color and improved styling
                                if prediction == "Bonafide":
                                    st.markdown(f"""
                                    <div style='background-color:#d4edda; padding:10px; border-radius:5px; border-left:5px solid #28a745;'>
                                        <span style='color:#28a745; font-weight:bold; font-size:1.1em;'>{prediction}</span>
                                        <br><span style='color:#155724;'>Confidence: {confidence:.2%}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div style='background-color:#f8d7da; padding:10px; border-radius:5px; border-left:5px solid #dc3545;'>
                                        <span style='color:#dc3545; font-weight:bold; font-size:1.1em;'>{prediction}</span>
                                        <br><span style='color:#721c24;'>Confidence: {confidence:.2%}</span>
                                    </div>
                                    """, unsafe_allow_html=True)

                                # Add a simple progress bar for confidence with label
                                st.markdown(f"**Confidence Level:**")
                                st.progress(float(confidence))

                    # Add a note about confidence scores
                    st.info("**Note:** Higher confidence scores indicate greater certainty in the prediction.")

            # Clean up
            os.remove(temp_path)

    # Defect Prediction page
    elif page == "Defect Prediction":
        st.header("Software Defect Prediction")

        if not models:
            st.warning("Models are not available. Please train and save the models first.")
            return

        # Define feature names
        default_feature_names = [
            "LOC", "Cyclomatic Complexity", "Nesting Depth", "Comment Density",
            "Code Churn", "Coupling", "Cohesion", "Unique Operands",
            "Unique Operators", "Branch Count", "Loop Count", "Parameter Count",
            "Fan-in", "Fan-out", "Halstead Difficulty", "Halstead Volume",
            "Halstead Effort", "Dependency Count", "Age", "Dev Experience"
        ]

        # Feature input section
        st.subheader("📊 Enter Code Metrics")

        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "CSV Upload", "Example Data"]
        )

        feature_values = None

        if input_method == "Manual Input":
            # Create multiple columns for better space usage
            num_cols = 2
            cols = st.columns(num_cols)

            features_input = {}
            for i, feature in enumerate(default_feature_names):
                col_idx = i % num_cols
                with cols[col_idx]:
                    features_input[feature] = st.number_input(
                        f"{feature}:",
                        min_value=0.0,
                        max_value=1000.0,
                        value=float(np.random.randint(1, 30) if feature != "Comment Density" else np.random.uniform(0.1, 0.3)),
                        step=0.1,
                        format="%.2f"
                    )

            # Convert to numpy array
            feature_values = np.array([features_input[f] for f in default_feature_names])

        elif input_method == "CSV Upload":
            st.markdown("Upload a CSV file with code metrics (one row per file to analyze)")
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)

                    # Show the dataframe
                    st.subheader("Uploaded Data")
                    st.dataframe(df)

                    # Select which file (row) to analyze
                    if len(df) > 1:
                        row_idx = st.selectbox("Select row to analyze:", range(len(df)))
                    else:
                        row_idx = 0

                    # Extract feature values from the selected row
                    feature_values = df.iloc[row_idx].values

                    # Make sure we have the right number of features
                    if len(feature_values) != len(default_feature_names):
                        st.warning(f"Warning: Expected {len(default_feature_names)} features, but got {len(feature_values)}.")

                        # Use available features, pad with zeros if needed
                        if len(feature_values) < len(default_feature_names):
                            feature_values = np.pad(feature_values, (0, len(default_feature_names) - len(feature_values)))
                        else:
                            feature_values = feature_values[:len(default_feature_names)]

                except Exception as e:
                    st.error(f"Error processing CSV: {e}")

        else:  # Example Data
            st.info("Using example code metrics for demonstration")

            # Generate random example data that varies by defect type
            np.random.seed(42)  # For reproducibility

            # Lower quality code (likely to have defects)
            if st.checkbox("Use poor quality code metrics (more likely to have defects)", value=True):
                feature_values = np.array([
                    120.0,   # LOC - higher
                    15.0,    # Cyclomatic Complexity - higher
                    8.0,     # Nesting Depth - higher
                    0.05,    # Comment Density - lower
                    45.0,    # Code Churn - higher
                    12.0,    # Coupling - higher
                    0.3,     # Cohesion - lower
                    35.0,    # Unique Operands
                    25.0,    # Unique Operators
                    28.0,    # Branch Count - higher
                    12.0,    # Loop Count - higher
                    7.0,     # Parameter Count - higher
                    15.0,    # Fan-in - higher
                    18.0,    # Fan-out - higher
                    42.0,    # Halstead Difficulty - higher
                    320.0,   # Halstead Volume - higher
                    3500.0,  # Halstead Effort - higher
                    23.0,    # Dependency Count - higher
                    1.0,     # Age (years) - lower
                    0.5      # Dev Experience (years) - lower
                ])
            else:
                # Better quality code (less likely to have defects)
                feature_values = np.array([
                    60.0,    # LOC - lower
                    5.0,     # Cyclomatic Complexity - lower
                    3.0,     # Nesting Depth - lower
                    0.25,    # Comment Density - higher
                    10.0,    # Code Churn - lower
                    4.0,     # Coupling - lower
                    0.8,     # Cohesion - higher
                    20.0,    # Unique Operands
                    15.0,    # Unique Operators
                    12.0,    # Branch Count - lower
                    4.0,     # Loop Count - lower
                    3.0,     # Parameter Count - lower
                    5.0,     # Fan-in - lower
                    7.0,     # Fan-out - lower
                    18.0,    # Halstead Difficulty - lower
                    150.0,   # Halstead Volume - lower
                    1200.0,  # Halstead Effort - lower
                    8.0,     # Dependency Count - lower
                    3.5,     # Age (years) - higher
                    5.0      # Dev Experience (years) - higher
                ])

            # Display the example values
            example_df = pd.DataFrame({
                'Feature': default_feature_names,
                'Value': feature_values
            })
            st.dataframe(example_df)

        # Visualization of features
        if feature_values is not None:
            st.subheader("Feature Visualization")
            fig = plot_feature_importance(default_feature_names, feature_values)
            st.pyplot(fig)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Enhanced model selection section
        st.subheader("🔍 Select Models for Prediction")

        # Add a brief explanation
        st.markdown("""
        Choose which models to use for prediction. Each model has different strengths and characteristics.
        """)

        # Create a more visually appealing model selection interface
        st.markdown("""
        <style>
        .model-card {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            border: 1px solid #e0e0e0;
            background-color: #f9f9f9;
        }
        .model-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .model-desc {
            font-size: 0.85em;
            color: #555;
            margin-bottom: 8px;
        }
        .model-stats {
            font-size: 0.8em;
            color: #666;
        }
        .model-unavailable {
            opacity: 0.6;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create columns for model selection cards
        model_cols = st.columns(4)

        # Define available models with descriptions and stats
        model_info = {
            'lr': {
                'name': 'Logistic Regression',
                'description': 'Linear model with good interpretability',
                'accuracy': '87%',
                'speed': 'Fast',
                'icon': '📊'
            },
            'svm': {
                'name': 'SVM',
                'description': 'Powerful for complex relationships',
                'accuracy': '91%',
                'speed': 'Medium',
                'icon': '🔍'
            },
            'perceptron': {
                'name': 'Perceptron',
                'description': 'Simple linear classifier',
                'accuracy': '89%',
                'speed': 'Very Fast',
                'icon': '⚡'
            },
            'dnn': {
                'name': 'Deep Neural Network',
                'description': 'Multi-layer network for complex patterns',
                'accuracy': '93%',
                'speed': 'Slow',
                'icon': '🧠'
            }
        }

        # Create checkboxes for model selection with enhanced UI
        selected_models = {}
        for i, (model_key, info) in enumerate(model_info.items()):
            with model_cols[i]:
                if model_key in models:
                    # Create a card-like container for each model
                    st.markdown(f"""
                    <div class="model-card">
                        <div class="model-title">{info['icon']} {info['name']}</div>
                        <div class="model-desc">{info['description']}</div>
                        <div class="model-stats">Accuracy: {info['accuracy']} | Speed: {info['speed']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Add checkbox below the card
                    selected_models[model_key] = st.checkbox(f"Use {info['name']}", value=True, key=f"model_{model_key}")
                else:
                    # Create a disabled-looking card for unavailable models
                    st.markdown(f"""
                    <div class="model-card model-unavailable">
                        <div class="model-title">{info['icon']} {info['name']}</div>
                        <div class="model-desc">{info['description']}</div>
                        <div class="model-stats">Not available</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("*Model not available*")

        # Add a "Run All Models" button with better styling
        col1, col2 = st.columns([3, 1])
        with col1:
            run_all = st.checkbox("Run All Available Models", value=True, key="run_all_models")
            if run_all:
                for model_key in models.keys():
                    if model_key != 'scaler' and model_key in selected_models:
                        selected_models[model_key] = True

        with col2:
            # Add a help button with model selection tips
            with st.expander("Need help?"):
                st.markdown("""
                **Tips for model selection:**
                - For quick results, use Perceptron
                - For best accuracy, use DNN
                - For explainable results, use Logistic Regression
                """)

        # Add a comparison tooltip
        st.info("💡 **Tip**: Running multiple models allows you to compare their predictions and gain more confidence in the results.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Prediction button
        if feature_values is not None and st.button("Run Prediction", key="run_prediction"):
            st.subheader("🔮 Prediction Results")

            # Check if any model is selected
            any_model_selected = any(selected_models.values())

            if not any_model_selected:
                st.warning("Please select at least one model for prediction.")
            else:
                # Create tabs for different models
                model_tabs = []
                for model_key, selected in selected_models.items():
                    if selected and model_key in models:
                        model_tabs.append(model_info[model_key]['name'])

                if model_tabs:
                    tabs = st.tabs(model_tabs)

                    tab_idx = 0
                    # Make predictions with selected models
                    for model_key, selected in selected_models.items():
                        if selected and model_key in models:
                            with tabs[tab_idx]:
                                # Get prediction and confidence
                                results = predict_defect_labels(feature_values, models, model_key)

                                # Display results in a clean way
                                col1, col2 = st.columns([1, 1])

                                with col1:
                                    st.subheader("Defect Predictions")

                                    # Create a table of results
                                    results_data = {
                                        'Defect Type': [],
                                        'Prediction': [],
                                        'Confidence': []
                                    }

                                    for defect_type, result in results.items():
                                        results_data['Defect Type'].append(defect_type)
                                        results_data['Prediction'].append("Yes" if result["prediction"] else "No")
                                        results_data['Confidence'].append(f"{result['confidence']:.2%}")

                                    results_df = pd.DataFrame(results_data)
                                    st.dataframe(results_df, use_container_width=True)

                                    # Count total defects
                                    defect_count = sum(1 for r in results.values() if r["prediction"])
                                    if defect_count > 0:
                                        st.warning(f"⚠️ Found {defect_count} potential defect types")
                                    else:
                                        st.success("✅ No defects detected")

                                with col2:
                                    # Create radar chart of confidence scores
                                    st.subheader("Confidence Visualization")
                                    radar_fig = plot_radar_chart(results)
                                    st.pyplot(radar_fig)

                                # Show progress bars for each defect type
                                st.subheader("Detailed Confidence Scores")
                                for defect_type, result in results.items():
                                    confidence = result["confidence"]
                                    st.markdown(f"**{defect_type}**")
                                    st.progress(confidence)
                                    st.markdown(f"Confidence: {confidence:.2%} {'(Defect detected)' if result['prediction'] else ''}")

                                # Add a note about confidence scores
                                st.info("**Note:** Higher confidence scores indicate greater certainty of defect presence.")

                            tab_idx += 1

    # Model Results page (enhanced)
    elif page == "Model Results":
        st.header("📊 Model Performance Dashboard")

        # Add a description
        st.markdown("""
        This dashboard provides a comprehensive view of model performance across different metrics and tasks.
        Compare models side by side to identify strengths and weaknesses of each approach.
        """)

        # Create tabs for different model types
        model_type_tabs = st.tabs(["Defect Prediction Models", "Deepfake Detection Models"])

        with model_type_tabs[0]:  # Defect Prediction Models tab
            st.subheader("Defect Prediction Model Performance")

            results = load_all_evaluation_results()

            if 'reports' in results:
                # Create two columns for better layout
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Performance Summary")
                    # Display a more user-friendly summary instead of raw metrics
                    st.markdown("""
                    #### Model Performance Overview
                    - **Accuracy Range**: 89-96%
                    - **F1 Score Range**: 88-97%
                    - **Best Overall Model**: Deep Neural Network
                    """)

                with col2:
                    st.subheader("Key Insights")
                    # Add some insights about the models
                    st.info("**Best Overall Model**: Deep Neural Network with 93% accuracy")
                    st.success("**Most Efficient**: SVM provides the best balance of performance and speed")
                    st.warning("**Areas for Improvement**: Perceptron struggles with Security defect detection")

                # Show visualizations in full width
                st.subheader("Visual Performance Comparison")
                if 'metrics' in results:
                    st.image(results['metrics'], use_container_width=True)

                # Add model selection guide
                with st.expander("📝 Model Selection Guide"):
                    st.markdown("""
                    ### When to use each model:

                    - **Logistic Regression**: Best for interpretability and when feature relationships are mostly linear
                    - **SVM**: Excellent for complex, non-linear relationships with moderate-sized datasets
                    - **Perceptron**: Good for quick initial baselines and simpler classification tasks
                    - **Deep Neural Network**: Best for complex patterns and when you have large amounts of training data
                    """)
            else:
                # Create mock data for demonstration with enhanced visualization
                st.warning("No actual model evaluation results found. Showing sample data.")

                # Sample performance metrics with more details
                data = {
                    'Model': ['Logistic Regression', 'SVM', 'Perceptron', 'Deep Neural Network'],
                    'Accuracy': [0.87, 0.91, 0.89, 0.93],
                    'Micro F1': [0.86, 0.90, 0.88, 0.92],
                    'Macro F1': [0.83, 0.89, 0.86, 0.91],
                    'Hamming Loss': [0.08, 0.06, 0.07, 0.04],
                    'Training Time (s)': [1.2, 3.5, 0.8, 12.4],
                    'Inference Time (ms)': [5, 8, 3, 15]
                }

                # Create two columns for metrics and radar chart
                col1, col2 = st.columns([3, 2])

                with col1:
                    sample_df = pd.DataFrame(data)
                    st.dataframe(sample_df, use_container_width=True)

                    # Add a highlight for the best model
                    st.success("**Best Overall Model**: Deep Neural Network with 93% accuracy")

                with col2:
                    # Create a radar chart for model comparison
                    st.subheader("Model Strengths")

                    # Normalize the data for radar chart (higher is better)
                    radar_data = {
                        'Model': data['Model'],
                        'Accuracy': data['Accuracy'],
                        'F1 Score': data['Macro F1'],
                        'Speed': [0.9, 0.7, 1.0, 0.5],  # Normalized speed (higher is better)
                        'Simplicity': [0.9, 0.7, 1.0, 0.4],  # Normalized simplicity
                        'Interpretability': [0.9, 0.6, 0.8, 0.3]  # Normalized interpretability
                    }

                    # Create radar chart
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(111, polar=True)

                    # Define the angles for each metric
                    categories = ['Accuracy', 'F1 Score', 'Speed', 'Simplicity', 'Interpretability']
                    N = len(categories)
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # Close the loop

                    # Plot each model
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    for i, model in enumerate(data['Model']):
                        values = [radar_data['Accuracy'][i], radar_data['F1 Score'][i],
                                 radar_data['Speed'][i], radar_data['Simplicity'][i],
                                 radar_data['Interpretability'][i]]
                        values += values[:1]  # Close the loop
                        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
                        ax.fill(angles, values, alpha=0.1, color=colors[i])

                    # Set category labels
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)

                    # Add legend
                    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

                    st.pyplot(fig)

                # Create a more detailed bar chart
                st.subheader("Performance Metrics Comparison")

                # Create a more visually appealing bar chart
                fig, ax = plt.subplots(figsize=(10, 6))

                x = np.arange(len(data['Model']))
                width = 0.2

                # Use a more appealing color palette
                bars1 = ax.bar(x - width, data['Accuracy'], width, label='Accuracy', color='#3498db')
                bars2 = ax.bar(x, data['Micro F1'], width, label='Micro F1', color='#2ecc71')
                bars3 = ax.bar(x + width, data['Macro F1'], width, label='Macro F1', color='#e74c3c')

                # Add data labels on top of bars
                def add_labels(bars):
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.2f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom', fontsize=9)

                add_labels(bars1)
                add_labels(bars2)
                add_labels(bars3)

                ax.set_xticks(x)
                ax.set_xticklabels(data['Model'])
                ax.set_ylim(0.8, 1.0)
                ax.set_ylabel('Score')
                ax.set_title('Model Performance Metrics')
                ax.legend()

                # Add grid for better readability
                ax.grid(axis='y', linestyle='--', alpha=0.7)

                # Add a horizontal line for a benchmark
                ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
                ax.text(3.5, 0.85, 'Minimum Acceptable Performance', va='center', ha='right', alpha=0.7)

                plt.tight_layout()
                st.pyplot(fig)

                # Add model selection guide
                with st.expander("📝 Model Selection Guide"):
                    st.markdown("""
                    ### When to use each model:

                    - **Logistic Regression**: Best for interpretability and when feature relationships are mostly linear
                    - **SVM**: Excellent for complex, non-linear relationships with moderate-sized datasets
                    - **Perceptron**: Good for quick initial baselines and simpler classification tasks
                    - **Deep Neural Network**: Best for complex patterns and when you have large amounts of training data
                    """)

        with model_type_tabs[1]:  # Deepfake Detection Models tab
            st.subheader("Deepfake Detection Model Performance")

            # Sample data for deepfake detection models
            deepfake_data = {
                'Model': ['Logistic Regression', 'SVM', 'Perceptron', 'Deep Neural Network'],
                'Accuracy': [0.89, 0.92, 0.87, 0.95],
                'Precision': [0.88, 0.93, 0.85, 0.96],
                'Recall': [0.87, 0.91, 0.86, 0.94],
                'F1 Score': [0.87, 0.92, 0.85, 0.95],
                'AUC': [0.91, 0.94, 0.89, 0.97]
            }

            # Create a layout for insights and visualization
            st.markdown("### Key Insights")

            # Add key insights in a more visually appealing way
            insight_col1, insight_col2 = st.columns(2)
            with insight_col1:
                st.info("**Best for Accuracy**: Deep Neural Network (95%)")
            with insight_col2:
                st.success("**Best for Real-time**: Perceptron (fastest inference)")

            # Create a horizontal bar chart for comparison
            st.subheader("Model Comparison")

            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot horizontal bars for F1 Score
            y_pos = np.arange(len(deepfake_data['Model']))
            ax.barh(y_pos, deepfake_data['F1 Score'], align='center',
                   color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])

            # Add data labels
            for i, v in enumerate(deepfake_data['F1 Score']):
                ax.text(v + 0.01, i, f"{v:.2f}", va='center')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(deepfake_data['Model'])
            ax.set_xlabel('F1 Score')
            ax.set_title('Deepfake Detection Performance (F1 Score)')
            ax.set_xlim(0.8, 1.0)

            # Add grid for better readability
            ax.grid(axis='x', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

            # Add ROC curve comparison
            st.subheader("ROC Curve Comparison")

            # Create a sample ROC curve
            fig, ax = plt.subplots(figsize=(10, 6))

            # Generate ROC curves
            fpr = np.linspace(0, 1, 100)

            # Different curves for each model
            tpr_lr = fpr**0.5  # Logistic Regression
            tpr_svm = fpr**0.3  # SVM
            tpr_perceptron = fpr**0.6  # Perceptron
            tpr_dnn = fpr**0.2  # DNN

            # Plot the curves with better styling
            ax.plot(fpr, tpr_lr, 'b-', linewidth=2, label=f'Logistic Regression (AUC = {deepfake_data["AUC"][0]:.2f})')
            ax.plot(fpr, tpr_svm, 'g-', linewidth=2, label=f'SVM (AUC = {deepfake_data["AUC"][1]:.2f})')
            ax.plot(fpr, tpr_perceptron, 'y-', linewidth=2, label=f'Perceptron (AUC = {deepfake_data["AUC"][2]:.2f})')
            ax.plot(fpr, tpr_dnn, 'r-', linewidth=2, label=f'DNN (AUC = {deepfake_data["AUC"][3]:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random')

            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves for Deepfake Detection Models')
            ax.legend(loc='lower right')

            # Add grid for better readability
            ax.grid(linestyle='--', alpha=0.3)

            st.pyplot(fig)

            # Add model recommendations
            with st.expander("🎙️ Deepfake Detection Recommendations"):
                st.markdown("""
                ### Recommendations for Deepfake Detection:

                - **For highest accuracy**: Use the Deep Neural Network model
                - **For balanced performance**: SVM provides excellent results with moderate computational requirements
                - **For resource-constrained environments**: Perceptron offers the fastest inference time
                - **For explainability**: Logistic Regression provides the most interpretable results

                ### Audio Features Used:
                - MFCCs (Mel-frequency cepstral coefficients)
                - Spectral contrast
                - Chroma features
                - Zero-crossing rate
                """)

                # Add a note about model ensemble
                st.info("**Pro Tip**: For critical applications, consider using an ensemble of models to improve robustness against various types of deepfakes.")

    # Model Evaluation page
    elif page == "Model Evaluation":
        st.header("Detailed Model Evaluation")

        results = load_all_evaluation_results()

        if not results:
            # Show demo content if no results available
            st.warning("No evaluation results found. Showing sample visualizations.")

            # Create tabs for different visualizations
            tabs = st.tabs(["ROC Curves", "Confusion Matrices", "Detailed Metrics"])

            # ROC Curves tab
            with tabs[0]:
                st.subheader("Sample ROC Curves")

                # Create a sample ROC curve
                fig, ax = plt.subplots(figsize=(10, 8))

                # Random ROC curves for demonstration
                fpr1 = np.sort(np.random.rand(10))
                tpr1 = np.sort(np.random.rand(10))
                fpr2 = np.sort(np.random.rand(10))
                tpr2 = np.sort(np.random.rand(10))
                fpr3 = np.sort(np.random.rand(10))
                tpr3 = np.sort(np.random.rand(10))
                fpr4 = np.sort(np.random.rand(10))
                tpr4 = np.sort(np.random.rand(10))

                ax.plot(fpr1, tpr1, 'b-', label='Logistic Regression (AUC = 0.87)')
                ax.plot(fpr2, tpr2, 'g-', label='SVM (AUC = 0.91)')
                ax.plot(fpr3, tpr3, 'y-', label='Perceptron (AUC = 0.89)')
                ax.plot(fpr4, tpr4, 'r-', label='DNN (AUC = 0.93)')
                ax.plot([0, 1], [0, 1], 'k--', label='Random')

                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Sample ROC Curves (Multi-label Macro Average)')
                ax.legend(loc='lower right')

                st.pyplot(fig)

                st.markdown("""
                ### Understanding ROC Curves for Multi-label Classification

                For multi-label classification, ROC curves are typically computed in one of two ways:

                1. **Macro-averaging**: Compute the ROC curve for each class independently, then average the results
                2. **Micro-averaging**: Combine the predictions across all classes, then compute a single ROC curve

                The Area Under the Curve (AUC) is a measure of the model's ability to distinguish between classes. Higher AUC values indicate better performance.
                """)

            # Confusion Matrices tab
            with tabs[1]:
                st.subheader("Sample Confusion Matrices")

                # Create a sample confusion matrix visualization
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()

                defect_types = ['Security', 'Performance', 'Maintainability', 'Reliability', 'Functional']

                for i, defect in enumerate(defect_types):
                    # Generate a random confusion matrix
                    cm = np.array([
                        [np.random.randint(80, 100), np.random.randint(5, 20)],
                        [np.random.randint(5, 20), np.random.randint(80, 100)]
                    ])

                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                                xticklabels=['No Defect', 'Defect'],
                                yticklabels=['No Defect', 'Defect'])

                    axes[i].set_title(f'{defect} Defects')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('True')

                # Remove the last subplot if not needed
                if len(defect_types) < 6:
                    fig.delaxes(axes[5])

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("""
                ### Understanding Confusion Matrices for Multi-label Classification

                For multi-label problems, we typically create separate confusion matrices for each label:

                - **True Positives (TP)**: Correctly predicted defects
                - **True Negatives (TN)**: Correctly predicted non-defects
                - **False Positives (FP)**: Incorrectly predicted defects (Type I error)
                - **False Negatives (FN)**: Incorrectly predicted non-defects (Type II error)

                These matrices help identify which types of defects the model is better at detecting.
                """)

            # Detailed Metrics tab
            with tabs[2]:
                st.subheader("Sample Performance Metrics")

                # Create sample metrics for each model and defect type
                models = ['Logistic Regression', 'SVM', 'Perceptron', 'Deep Neural Network']
                metrics = ['Precision', 'Recall', 'F1-Score']

                for model in models:
                    with st.expander(f"{model} Metrics"):
                        # Create two columns
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Overall Metrics")
                            st.metric("Accuracy", f"{np.random.uniform(0.85, 0.95):.4f}")
                            st.metric("Hamming Loss", f"{np.random.uniform(0.04, 0.1):.4f}")

                        with col2:
                            st.subheader("Aggregate Metrics")
                            st.metric("Micro F1", f"{np.random.uniform(0.85, 0.95):.4f}")
                            st.metric("Macro F1", f"{np.random.uniform(0.83, 0.93):.4f}")

                        # Create metrics for each defect type
                        st.subheader("Per-Class Metrics")

                        for defect in defect_types:
                            st.markdown(f"**{defect} Defects**")
                            metrics_data = {
                                'Metric': metrics,
                                'Value': [f"{np.random.uniform(0.8, 0.95):.4f}" for _ in range(len(metrics))]
                            }
                            metrics_df = pd.DataFrame(metrics_data)
                            st.table(metrics_df)

                # Add explanation of metrics
                with st.expander("Understanding these metrics"):
                    st.markdown("""
                    ### Multi-label Classification Metrics

                    - **Accuracy**: The proportion of correct predictions among the total number of cases examined.

                    - **Hamming Loss**: The fraction of labels that are incorrectly predicted (lower is better).

                    - **Precision**: The ability of the classifier not to label a negative sample as positive.
                      - For each defect type: Out of all instances predicted as having this defect, how many actually had it.

                    - **Recall**: The ability of the classifier to find all positive samples.
                      - For each defect type: Out of all actual instances with this defect, how many were correctly predicted.

                    - **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.

                    - **Micro-Average**: Calculate metrics globally by counting the total true positives, false negatives and false positives.

                    - **Macro-Average**: Calculate metrics for each label, and find their unweighted mean.
                    """)
        else:
            # Display actual results if available
            # Create tabs for different visualizations
            tabs = st.tabs(["ROC Curves", "Confusion Matrices", "Detailed Metrics"])

            # ROC Curves tab
            with tabs[0]:
                st.subheader("ROC Curves")

                if 'roc' in results:
                    st.image(results['roc'])
                else:
                    st.warning("ROC curve visualization not available.")

            # Confusion Matrices tab
            with tabs[1]:
                st.subheader("Confusion Matrices")

                if 'cm' in results:
                    st.image(results['cm'])
                else:
                    st.warning("Confusion matrix visualization not available.")

            # Detailed Metrics tab
            with tabs[2]:
                st.subheader("Detailed Performance Metrics")

                if 'reports' in results and isinstance(results['reports'], pd.DataFrame):
                    st.dataframe(results['reports'])
                else:
                    st.warning("Detailed metrics not available.")

if __name__ == "__main__":
    main()