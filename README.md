# Intrusion Detection System using XGBoost (AI)

An **AI-based Intrusion Detection System (IDS)** that uses **XGBoost** and the **UNSW-NB15 dataset** to perform **binary classification** of network traffic as **Normal** or **Attack**.  
The project implements a complete machine learning pipeline including preprocessing, class imbalance handling, model training, evaluation, and visualization.

---

## 📌 Project Overview

- **Model**: XGBoost Classifier  
- **Task**: Binary Classification (Normal vs Attack)  
- **Dataset**: UNSW-NB15  
- **Language**: Python  
- **Techniques Used**:
  - One-Hot Encoding
  - Feature Scaling (StandardScaler)
  - Class Imbalance Handling (SMOTE)
  - Performance Evaluation & Visualization

---

## 🧠 Key Features

✔ Loads and merges training & testing datasets  
✔ Encodes categorical features automatically  
✔ Scales numerical features  
✔ Handles class imbalance using **SMOTE**  
✔ Trains an optimized **XGBoost** model  
✔ Evaluates with Accuracy, Precision, Recall, F1-Score  
✔ Generates Confusion Matrix, ROC Curve & Feature Importance plots  

---

## 📁 Project Structure

```

Intrusion-Detection-System-using-AI/
│
├── main.py
├── UNSW_NB15_training-set.csv
├── UNSW_NB15_testing-set.csv
├── requirements.txt
├── results/
│   ├── metrics.txt
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
└── README.md

````

---

## 📊 Dataset Information

This project uses the **UNSW-NB15 dataset**, a modern and widely used benchmark dataset for intrusion detection research.

- **Label Definition**:
  - `0` → Normal Traffic
  - `1` → Attack Traffic

Both training and testing datasets are combined before preprocessing to maintain feature consistency.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/somendra-n/Intrusion-Detection-System-using-AI.git
cd Intrusion-Detection-System-using-AI
````

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

Ensure the dataset CSV files are present in the project root directory, then run:

```bash
python main.py
```

The script will:

* Preprocess the data
* Train the XGBoost model
* Evaluate performance
* Save metrics and plots in the `results/` directory

---

## 📈 Model Evaluation Metrics

The following evaluation metrics are computed:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC Score

Results are saved in:

```
results/metrics.txt
```

---

## 📉 Visual Outputs

Automatically generated plots include:

* Confusion Matrix Heatmap
* ROC Curve
* Top 20 Feature Importances (XGBoost)

All visual outputs are saved in:

```
results/
```

---

## 🛠️ Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)
* Matplotlib & Seaborn

---

## 🚀 Future Enhancements

* Multi-class attack classification
* Real-time network traffic analysis
* Deep learning models (LSTM, CNN)
* Web-based monitoring dashboard

---

## 📄 License & Copyright

### © 2026 Somendra N. All Rights Reserved.

This project is protected under **copyright law**.

#### Usage Policy:

* ✅ Allowed for **academic, educational, and research purposes**
* ❌ Commercial use is **not permitted** without prior written permission
* ❌ Redistribution or modification without attribution is prohibited

You **must provide proper credit** to the author when using or referencing this project.

For permission requests, please contact the repository owner.

---

## 👤 Author

**Somendra N**
Intrusion Detection using AI & Machine Learning
GitHub: [https://github.com/somendra-n](https://github.com/somendra-n)
