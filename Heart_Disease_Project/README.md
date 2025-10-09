# ❤️ Heart Disease Prediction Project

## 📘 Overview
This project aims to predict heart disease presence using machine learning techniques.  
It includes both **supervised** (classification) and **unsupervised** (clustering) models, with feature reduction via PCA and hyperparameter tuning for optimization.

---

## 📂 Project Structure
Heart_Disease_Project/
│── data/
│ └── heart_disease.csv
│── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_pca_analysis.ipynb
│ ├── 03_feature_selection.ipynb
│ ├── 04_supervised_learning.ipynb
│ ├── 05_unsupervised_learning.ipynb
│ └── 06_hyperparameter_tuning.ipynb
│── models/
│ └── final_model.pkl
│── ui/
│ └── app.py
│── deployment/
│ └── ngrok_setup.txt
│── results/
│ └── evaluation_metrics.txt
│── README.md
│── requirements.txt
│── .gitignore
---

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

## 💻 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YoussefEbrahim/Heart_Disease_Project.git
   cd Heart_Disease_Project
====================================
## 👨‍💻 Author

**Youssef Ebrahim**  
Artificial Intelligence & Data Science Student at Zagazig University  
Machine Learning Engineer | IEEE Member  

[🌐 LinkedIn](https://www.linkedin.com/in/youssef-ebrahim) | [💻 GitHub](https://github.com/YoussefEbrahim)
