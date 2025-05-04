# Urdu Deepfake Audio Detection: Model Evaluation Report

## Project Overview

This project focuses on developing and evaluating machine learning models for detecting deepfake audio in the Urdu language. We trained and compared four different models:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Perceptron
4. Deep Neural Network (DNN)

The models were trained on the CSALT Deepfake Detection Dataset for Urdu, which contains both bonafide (real) and deepfake audio samples.

## Dataset

The dataset used is the [CSALT Deepfake Detection Dataset for Urdu](https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu), which contains 6,794 audio samples. The dataset is balanced between bonafide and deepfake samples.

For feature extraction, we used Mel-Frequency Cepstral Coefficients (MFCCs), which are commonly used in speech and audio processing tasks. Each audio sample was processed to extract 13 MFCCs, which were then padded or truncated to a fixed length of 200 frames, resulting in a feature vector of 2,600 dimensions.

## Model Training

We split the dataset into training (70%), validation (10%), and test (20%) sets. The features were standardized using StandardScaler to have zero mean and unit variance.

The models were trained with the following configurations:

1. **Logistic Regression**: Standard implementation with L2 regularization and maximum 1000 iterations.
2. **SVM**: Radial Basis Function (RBF) kernel with probability estimation enabled.
3. **Perceptron**: Standard implementation with maximum 1000 iterations.
4. **Deep Neural Network**: A 3-layer neural network with ReLU activation, dropout regularization, and sigmoid output. Trained using Adam optimizer and Binary Cross-Entropy loss.

## Evaluation Results

### Overall Performance

| Model               | Accuracy | Deepfake F1 | Bonafide F1 | Macro Avg F1 |
| ------------------- | -------- | ----------- | ----------- | ------------ |
| Deep Neural Network | 0.9684   | 0.9684      | 0.9684      | 0.9684       |
| SVM                 | 0.9617   | 0.9615      | 0.9619      | 0.9617       |
| Logistic Regression | 0.9007   | 0.9005      | 0.9008      | 0.9007       |
| Perceptron          | 0.8999   | 0.8994      | 0.9004      | 0.8999       |

### ROC Curves

The ROC curves show the trade-off between true positive rate and false positive rate at various threshold settings. The Area Under the Curve (AUC) is a measure of the model's ability to distinguish between classes.

All models performed well, with the DNN and SVM achieving the highest AUC values, followed by Logistic Regression and Perceptron.

### Confusion Matrices

The confusion matrices provide a detailed breakdown of correct and incorrect predictions for each class:

1. **Deep Neural Network**: Achieved the highest accuracy with very few misclassifications in both classes.
2. **SVM**: Performed nearly as well as the DNN, with slightly more misclassifications.
3. **Logistic Regression**: Had more misclassifications but still maintained a balanced performance across both classes.
4. **Perceptron**: Similar performance to Logistic Regression, with a slight imbalance in class-specific performance.

## Key Findings

1. **Deep Neural Network** performed the best across all metrics, achieving an accuracy of 96.84% and an F1-score of 0.9684 for both classes.
2. **SVM** was a close second, with an accuracy of 96.17% and an F1-score of 0.9617.
3. **Logistic Regression** and **Perceptron** had similar performance, with accuracies around 90%.
4. All models showed balanced performance across both classes, indicating that they are equally effective at identifying both bonafide and deepfake audio.

## Conclusion

The results demonstrate that machine learning models, particularly Deep Neural Networks and SVMs, can effectively detect deepfake audio in the Urdu language with high accuracy. The DNN model achieved the best performance, making it the recommended choice for deployment.

The high performance across different model architectures suggests that the MFCC features provide strong discriminative power for this task. This is encouraging for real-world applications where deepfake detection is becoming increasingly important.

## Interactive Web Application

We developed a Streamlit web application to demonstrate the models' capabilities in real-time. The application features:

1. **Clean, Intuitive Interface**: A user-friendly dashboard with a navigation panel for easy access to different sections.
2. **Deepfake Detection**: Upload audio files and get predictions from multiple models with confidence scores.
3. **Model Analytics**: Visualize model performance metrics and compare different models.
4. **Performance Metrics**: View detailed evaluation results including ROC curves and confusion matrices.

The application is designed to be accessible to non-technical users while providing detailed information for those interested in the technical aspects of the models.

## Future Work

1. **Feature Engineering**: Explore additional audio features beyond MFCCs, such as spectrograms, pitch features, or raw waveforms.
2. **Model Architecture**: Experiment with more complex neural network architectures, such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs).
3. **Transfer Learning**: Investigate the use of pre-trained audio models for feature extraction or fine-tuning.
4. **Robustness Testing**: Evaluate the models on more challenging datasets or with adversarial examples to test their robustness.
5. **Deployment**: Develop a production-ready system for real-time deepfake audio detection.
6. **User Interface Enhancements**: Further improve the Streamlit application with additional visualizations and user interaction features.

## Appendix: Implementation Details

The project was implemented in Python using the following libraries:

- scikit-learn for Logistic Regression, SVM, and Perceptron models
- PyTorch for the Deep Neural Network
- librosa for audio processing and feature extraction
- Streamlit for the interactive web application
- Matplotlib and Seaborn for visualization

### Project Structure

The project is organized into the following directories:

- `models/`: Contains all trained model files (.pkl and .pt)
- `visualizations/`: Contains all visualization assets (ROC curves, confusion matrices, etc.)
- `results/`: Contains evaluation metrics and reports
- `reports/`: Contains textual evaluation reports
- `streamlit_app/`: Contains the Streamlit web application

### Code Organization

The code is organized into several Python scripts:

- `sarim.py`: Data preprocessing and feature extraction
- `train_models.py`: Model training and evaluation
- `save_models.py`: Saving trained models for later use
- `evaluate_models.py`: Generating ROC curves and confusion matrices
- `visualize_evaluation.py`: Interactive visualization of evaluation results
- `app.py`: Streamlit web application for model demonstration

### Visualization Assets

All visualization assets are centralized in the `visualizations/` directory for better organization:

- ROC curves for each model (overall and individual)
- Confusion matrices for each model
- Model performance comparison charts
- Training progress visualizations for the DNN model
