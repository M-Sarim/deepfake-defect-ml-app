import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf

# Flag to check if librosa is available
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa module not available. Audio processing features will be limited.")

# Constants for audio processing
MAX_LEN = 200  # For audio feature extraction
N_MFCC = 13    # Number of MFCC coefficients

def extract_features(audio_path, sr=22050, n_mfcc=13, n_chroma=12, n_spectral=7):
    """
    Extract audio features from an audio file.

    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    sr : int
        Sample rate
    n_mfcc : int
        Number of MFCC coefficients to extract
    n_chroma : int
        Number of chroma features to extract
    n_spectral : int
        Number of spectral features to extract

    Returns:
    --------
    features : numpy.ndarray
        Extracted features
    """
    if not LIBROSA_AVAILABLE:
        print("Cannot extract features: librosa module not available")
        return None

    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)

        # Extract features
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]

        # Temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

        # Energy features
        rmse = librosa.feature.rms(y=y)[0]

        # Combine all features
        features = np.concatenate([
            mfccs_mean, mfccs_std,
            chroma_mean, chroma_std,
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(spectral_contrast), np.std(spectral_contrast),
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate),
            np.mean(rmse), np.std(rmse)
        ])

        return features

    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def extract_mfcc(file_path, max_len=MAX_LEN, n_mfcc=N_MFCC, use_compact=True):
    """Extract MFCC features from an audio file

    Args:
        file_path (str): Path to the audio file
        max_len (int): Maximum length of the MFCC features
        n_mfcc (int): Number of MFCC coefficients to extract
        use_compact (bool): If True, return a compact feature set (mean and std of MFCCs)
                           If False, return the full flattened MFCC matrix

    Returns:
        numpy.ndarray: MFCC features or None if extraction fails
    """
    if not LIBROSA_AVAILABLE:
        print("Cannot extract features: librosa module not available")
        return None

    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        if use_compact:
            # Extract statistical features (mean and std) for each coefficient
            # This creates a more compact representation (2*n_mfcc features)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # Add some additional features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]

            # Combine features
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                [np.mean(spectral_centroid), np.std(spectral_centroid)],
                [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)]
            ])

            return features
        else:
            # Pad or truncate to max_len
            if mfcc.shape[1] < max_len:
                pad_width = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :max_len]

            return mfcc.flatten()
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def extract_features_batch(audio_paths):
    """
    Extract features from a batch of audio files.

    Parameters:
    -----------
    audio_paths : list
        List of paths to audio files

    Returns:
    --------
    features : numpy.ndarray
        Extracted features for all audio files
    """
    features_list = []

    for path in audio_paths:
        features = extract_features(path)
        if features is not None:
            features_list.append(features)

    return np.array(features_list)

def detect_deepfake_features(audio_path):
    """
    Extract features specifically for deepfake detection.

    Parameters:
    -----------
    audio_path : str
        Path to the audio file

    Returns:
    --------
    features : numpy.ndarray
        Extracted features
    """
    # This function can be customized to extract features
    # that are particularly useful for deepfake detection

    # For now, we'll use the same feature extraction as above
    # but in a real application, you might want to add more specialized features
    return extract_features(audio_path)

def display_audio_visualizations(audio_path):
    """Create visualizations of the audio file

    Args:
        audio_path (str): Path to the audio file

    Returns:
        matplotlib.figure.Figure: Figure with waveform and spectrogram or None if visualization fails
    """
    if not LIBROSA_AVAILABLE:
        print("Cannot display visualizations: librosa module not available")
        return None

    try:
        y, sr = librosa.load(audio_path, sr=None)

        fig, ax = plt.subplots(2, 1, figsize=(10, 6))

        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=ax[0])
        ax[0].set_title('Waveform')

        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', ax=ax[1])
        ax[1].set_title('Spectrogram')
        fig.colorbar(img, ax=ax[1], format='%+2.0f dB')

        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return None
