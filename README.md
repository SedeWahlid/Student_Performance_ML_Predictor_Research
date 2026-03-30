# 🎓 Student Performance ML Predictor & Research

> A supervised learning model and research project powered by **Linear Regression** and **Random Forest Regressor** to predict student exam scores. This research demonstrates that standard demographic data (e.g., lunch type, race) lacks meaningful correlation to academic targets. Instead, it proves that accurate predictions rely heavily on historical academic indicators (prior test scores).

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?style=for-the-badge\&logo=streamlit\&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge\&logo=python)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)   
![joblib](https://img.shields.io/badge/joblib-%23D94F1F.svg?style=for-the-badge&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  
![scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)      
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

---

## 🎮 DEMO

Playground: coming soon...

## ✨ Features

* 📊 **Automated ML Pipeline**: Clean separation of data preprocessing, training, and evaluation.
* 🤖 **Algorithm Comparison**: Evaluates both Linear Regression and Random Forest Regressor.
* 🎯 **Multi vs. Single Target Analysis**: Compares the accuracy of predicting all scores simultaneously vs. predicting one score using the others as features.
* 🎛️ **Interactive Playground**: Real-time Streamlit UI to input student data and instantly generate score predictions.
* 🔄 **Dynamic Feature Alignment**: Automatically handles dummy variable mapping between single-row user inputs and the trained model's expected features.

---

## 🖼️ Preview

> Clean, minimal interface focused on data research and real-time inference.

```
Train Models → Analyze Reports → Input Student Data → Predict Scores
```

---

## 🛠️ Tech Stack

* 🐍 **Python** (Core Logic)
* ⚡ **Streamlit** (Web Interface)
* 🧠 **Scikit-Learn** (Machine Learning Models & Metrics)
* 🐼 **Pandas** (Data Manipulation & Encoding)
* 💾 **Joblib** (Model Serialization)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/SedeWahlid/Student_Performance_ML_Predictor_Research.git
cd Student_Performance_ML_Predictor_Research
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

**Step 1: Train the models and generate reports**  
*You can run this first to create the `.joblib` model files and `.json` reports but they are already provided in the models and reports folder.*
```bash
python train.py
```

**Step 2: Run the Streamlit web application**
```bash
streamlit run main.py
```

Then open your browser to:
1. Read the generated model performance reports (R² and MSE).
2. Review the research conclusions.
3. Use the interactive playground to test custom predictions! 🎉

---

## 🧠 How It Works

* **`train.py`** loads the `StudentsPerformance.csv` dataset and applies One-Hot Encoding (`pd.get_dummies`) to categorical variables.
* It trains models in two phases: predicting all 3 scores at once, and predicting 1 score using the other 2 as features.
* Trained models are serialized and saved to the `models/` directory using `joblib`.
* **`main.py`** loads these pre-trained models and renders the Streamlit UI.
* When a user submits data, the app dynamically aligns the user's input features with the model's expected training features using Pandas `.reindex()`, preventing the "dummy variable trap" during inference.

---

## 📁 Project Structure

```
📦 Student_Performance_ML_Predictor_Research
 ┣ 📂 data
 ┃ ┗ 📜 StudentsPerformance.csv
 ┣ 📂 models
 ┃ ┗ 📜 (Generated .joblib model files)
 ┣ 📂 reports
 ┃ ┗ 📜 all_reports.json
 ┣ 📜 main.py
 ┣ 📜 train.py
 ┣ 📜 README.md
 ┣ 📜 LICENSE
 ┣ 📜 .gitignore
 ┗ 📜 requirements.txt
```

---

## ⚠️ Notes

* 🛑 **Prerequisite**:
* Ensure the `StudentsPerformance.csv` file is correctly placed inside the `data/` directory.

---

## 📄 License

This project is licensed under the MIT License.

---
