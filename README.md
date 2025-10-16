# ❤️ Heart Disease Prediction Project

## 📘 Overview
This project aims to predict heart disease presence using machine learning techniques.  
It includes both **supervised** (classification) and **unsupervised** (clustering) models, with feature reduction via PCA and hyperparameter tuning for optimization.

---

## 📂 Project Structure
```
Heart_Disease_Project/
│
├── data/
│   └── heart_disease.csv                 # Dataset used for training and testing
│
├── notebooks/                            # Jupyter notebooks for analysis and experiments
│   ├── 01_data_preprocessing.ipynb       # Data cleaning and preprocessing
│   ├── 02_pca_analysis.ipynb             # Principal Component Analysis (PCA)
│   ├── 03_feature_selection.ipynb        # Feature selection techniques
│   ├── 04_supervised_learning.ipynb      # Model training using supervised learning
│   ├── 05_unsupervised_learning.ipynb    # Clustering and other unsupervised methods
│   └── 06_hyperparameter_tuning.ipynb    # Model optimization
│
├── models/
│   └── final_model.pkl                   # Saved trained model
│
├── ui/
│   └── app.py                            # Streamlit or Flask app for model deployment
│
├── deployment/
│   └── ngrok_setup.txt                   # Instructions or setup for deploying via ngrok
│
├── results/
│   └── evaluation_metrics.txt            # Model performance results and metrics
│
├── README.md                             # Project documentation
├── requirements.txt                      # List of dependencies
└── .gitignore                            # Files to be ignored by Git
```


## 🚀 Features
- **Data Preprocessing:** Cleaning, encoding, and scaling
- **PCA Analysis:** Dimensionality reduction
- **Feature Selection:** Statistical and model-based selection
- **Supervised Learning:** Classification models (Logistic Regression, Random Forest, XGBoost)
- **Unsupervised Learning:** Clustering for pattern discovery
- **Hyperparameter Tuning:** GridSearchCV / RandomizedSearchCV
- **Deployment:** Streamlit-based UI for predictions

---

## 🧠 Models Used
- Logistic Regression  
- Random Forest  
- XGBoost  
- KMeans  
- PCA  

---

## 🧪 Evaluation Metrics
See [`results/evaluation_metrics.txt`](results/evaluation_metrics.txt)

---

## 💻 Deployment
To run the Streamlit UI:
```bash
streamlit run ui/app.py
```

---

## 🚀 Live Demo
Check out the live app here: [Heart Disease Prediction App](https://heart-disease-uci-8zkvt3ugghtpctd8v9kuu8.streamlit.app/)

---

## 💻 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Youssef-Ebrahim0/Heart_Disease_Project.git
   cd Heart_Disease_Project
---

## ⚠️ Disclaimer
This project uses a machine learning model to predict the likelihood of heart disease based on input data.  
It is intended **for educational and demonstration purposes only** and **should not be used for medical decisions**.  
Always consult a licensed medical professional for proper diagnosis and treatment.

*Results may not always be accurate or reliable.*

====================================
## 👨‍💻 Author

**Youssef Ebrahim**  
Artificial Intelligence & Data Science Student at Zagazig University  
Machine Learning Engineer | IEEE Member  

[🌐 LinkedIn](https://www.linkedin.com/in/youssef-ebrahim01) | [💻 GitHub](https://github.com/Youssef-Ebrahim0)
