# 🧠🔍 Deepfake Audio & Software Defect Detection  
### Using Classical ML and Deep Neural Networks

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-ff4b4b?logo=streamlit)
[![Dataset](https://img.shields.io/badge/Dataset-Urdu%20Deepfakes-8A2BE2)](https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Overview

This repository tackles **two critical machine learning tasks**:

1. 🎙️ **Urdu Deepfake Audio Detection**  
   Detects synthetic (deepfake) vs. real (bonafide) Urdu speech using:
   - MFCC feature extraction (`librosa`)
   - Classical ML (Logistic Regression, SVM, Perceptron)
   - Deep Learning (PyTorch DNN)

2. 🐞 **Multi-Label Software Defect Prediction**  
   Predicts multiple possible defects from feature vectors of software modules using:
   - Logistic Regression (OvR), SVM (Binary Relevance), Perceptron
   - Deep Neural Network with Sigmoid + BCE Loss

💡 Both tasks are integrated into a user-friendly **Streamlit web app**, where users can:
- Upload audio for classification.
- Input software feature vectors.
- Select models at runtime (SVM, LR, DNN).
- View predictions with confidence scores.

---

## 📂 Project Structure

```text
deepfake-defect-ml-app/
├── data/                  # Raw & processed data
│   └── audio/             # Urdu audio samples
├── models/                # Trained models (.pkl / .pt)
├── notebooks/             # Jupyter notebooks
├── reports/               # Model evaluation reports
├── results/               # Output predictions
├── streamlit_app/         # Streamlit app code
│   └── utils/             # Utility scripts
└── visualizations/        # Confusion matrix, ROC, PR curves
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/M-Sarim/deepfake-defect-ml-app.git
cd deepfake-defect-ml-app

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## 📓 Running the Notebooks

Launch Jupyter and open the desired notebook from the `notebooks/` directory:

```bash
jupyter notebook
```

- `deepfake_audio_detection.ipynb` – For Urdu audio classification  
- `software_defect_prediction.ipynb` – For multi-label bug prediction

Ensure the required data is in `data/`.

---

## 🚀 Launch the Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```

🖼️ The browser UI allows:
- Drag-and-drop audio upload for deepfake detection
- Feature input for software module classification
- Runtime model selection + probability outputs

---

## 📊 Results Summary

### 🎙️ Deepfake Audio Detection (Binary)

| Model              | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 92.3%   | 91.8%     | 93.0%  | 92.4%    | 0.96    |
| SVM                 | 93.1%   | 92.6%     | 93.5%  | 93.0%    | 0.97    |
| Perceptron          | 89.4%   | 88.7%     | 90.0%  | 89.3%    | 0.94    |
| DNN (PyTorch)       | **95.2%** | **94.8%** | **95.7%** | **95.2%** | **0.98** |

### 🐞 Software Defect Prediction (Multi-Label)

| Model                       | Hamming Loss | Micro-F1 | Macro-F1 | Precision@k |
|----------------------------|--------------|----------|----------|-------------|
| Logistic Regression (OvR)  | 0.18         | 0.76     | 0.73     | 0.79        |
| SVM (Binary Relevance)     | 0.17         | 0.78     | 0.74     | 0.81        |
| Perceptron (Online)        | 0.21         | 0.72     | 0.69     | 0.75        |
| DNN (Sigmoid + BCE Loss)   | **0.14**     | **0.83** | **0.80** | **0.85**    |

---

## 📈 Visualizations

Located in the `visualizations/` folder:
- ✅ Confusion Matrices
- 📉 ROC Curves
- 📊 Precision-Recall Curves
- 📌 Label Distribution Graphs

---

## 🧰 Tools & Technologies

- **Frameworks:** PyTorch, Streamlit
- **Languages:** Python 3.9
- **Libraries:** NumPy, Pandas, Scikit-learn, Librosa, Matplotlib, Seaborn
- **Data Sources:**
  - [CSALT Urdu Deepfake Dataset (HuggingFace)](https://huggingface.co/datasets/CSALT/deepfake_detection_dataset_urdu)
  - Custom software module CSVs

---

## 🧠 Key Learnings

- 📢 MFCC features effectively capture audio signals for binary speech classification.
- 🔍 DNNs deliver strong performance with proper regularization and tuning.
- 📦 Streamlit provides quick deployment for model demos and stakeholder feedback.
- 🎯 Tackling label imbalance is essential for multi-label reliability.

---

## 🚧 Future Improvements

- 🌐 Expand deepfake detection to include multilingual datasets.
- 🔁 Integrate real-time audio augmentation for robust training.
- 🧠 Add explainability (e.g., SHAP, LIME) for bug prediction insights.
- ☁️ Cloud deployment (Heroku, AWS, or HuggingFace Spaces).

---

## 👤 Author

**Muhammad Sarim**  
📧 [muhammad2004sarim@gmail.com](mailto:muhammad2004sarim@gmail.com)  
💼 [LinkedIn](https://www.linkedin.com/in/imuhammadsarim/)  
🐱 [GitHub](https://github.com/M-Sarim)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
