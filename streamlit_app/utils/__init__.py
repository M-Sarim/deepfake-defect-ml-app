"""
Utility modules for the Streamlit app.
"""

from .audio_features import (
    extract_features, 
    extract_mfcc, 
    extract_features_batch, 
    detect_deepfake_features, 
    display_audio_visualizations,
    LIBROSA_AVAILABLE,
    MAX_LEN,
    N_MFCC
)

from .model_utils import (
    MultiLabelDNN,
    DeepNeuralNetwork,
    safe_load_pickle,
    predict_deepfake,
    predict_defects
)

from .model_loader import load_models

from .visualization_utils import (
    plot_feature_importance,
    plot_radar_chart,
    plot_confusion_matrix
)

from .evaluation_utils import load_evaluation_results

from .defect_utils import (
    preprocess_data,
    get_defect_types,
    create_sample_dataset
)
