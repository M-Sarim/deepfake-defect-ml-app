import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

def preprocess_data(df):
    """
    Preprocess the defect dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing audio features and defect labels
        
    Returns:
    --------
    X : numpy.ndarray
        Preprocessed features
    y : numpy.ndarray
        Preprocessed labels
    """
    # Assuming df has columns for features and a 'defects' column for labels
    # The 'defects' column might contain comma-separated defect types
    
    # Extract features (all columns except 'defects')
    feature_cols = [col for col in df.columns if col != 'defects']
    X = df[feature_cols].values
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Process labels
    # Assuming 'defects' column contains comma-separated defect types
    defects = df['defects'].str.split(',').tolist()
    
    # Convert to multi-hot encoding
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(defects)
    
    return X, y

def get_defect_types():
    """
    Get the list of possible audio defect types.
    
    Returns:
    --------
    defect_types : list
        List of defect types
    """
    return [
        'noise',
        'distortion',
        'clipping',
        'dropout',
        'hum',
        'click',
        'pop',
        'hiss',
        'wow_flutter',
        'phase_issues'
    ]

def create_sample_dataset(n_samples=1000, output_path='../data/dataset.csv'):
    """
    Create a sample dataset for audio defect classification.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    output_path : str
        Path to save the dataset
        
    Returns:
    --------
    df : pandas.DataFrame
        The generated dataset
    """
    # Define feature names (these would be audio features)
    feature_names = [
        'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean', 'mfcc4_mean', 'mfcc5_mean',
        'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
        'zero_crossing_rate', 'energy', 'tempo'
    ]
    
    # Generate random features
    np.random.seed(42)
    features = np.random.randn(n_samples, len(feature_names))
    
    # Create DataFrame with features
    df = pd.DataFrame(features, columns=feature_names)
    
    # Generate random defect labels
    defect_types = get_defect_types()
    defects = []
    
    for _ in range(n_samples):
        # Randomly select 0-3 defect types for each sample
        n_defects = np.random.randint(0, 4)
        if n_defects == 0:
            defects.append('none')
        else:
            sample_defects = np.random.choice(defect_types, size=n_defects, replace=False)
            defects.append(','.join(sample_defects))
    
    df['defects'] = defects
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df
