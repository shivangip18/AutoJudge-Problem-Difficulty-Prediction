# AutoJudge: Problem Difficulty Prediction

## 📌 Project Overview
AutoJudge is a machine learning–based system designed to automatically predict the difficulty of competitive programming problems.  
The project performs:
- **Classification** of problems into *Easy / Medium / Hard*
- **Regression** to estimate a continuous difficulty score

A web-based interface is provided to allow users to input a problem description and receive predictions in real time.

---

## 📊 Dataset Used
The dataset used in this project is provided in JSONL format as:

```
data/problems.jsonl
```

Each record contains problem-related textual and numerical metadata.  
The dataset is directly used for feature extraction and model training without external dependencies.

---

## ⚙️ Approach and Models Used

### 🔹 Data Preprocessing
- Handling missing values
- Normalization of numerical features
- Text preprocessing for feature extraction

### 🔹 Feature Engineering
Custom features are extracted using `feature_utils.py`, including:
- Problem description length
- Keyword frequency
- Metadata-based numerical features

### 🔹 Models
- **Classification Model:** Predicts difficulty class (Easy / Medium / Hard)
- **Regression Model:** Predicts a numerical difficulty score

Both models are trained using scikit-learn and saved for reuse.

---

## 📈 Evaluation Metrics
- **Classification:** Accuracy, Confusion Matrix
- **Regression:** Mean Absolute Error (MAE), Root Mean Square Error (RMSE)

All reported results correspond directly to the trained models included in the repository.

---

## 🌐 Web Interface
A Flask-based web application allows users to:
- Enter a programming problem description
- Receive predicted difficulty level and score

The UI is implemented using HTML templates and runs locally.

---

## ▶️ Demo Video
🎥 [Click here to watch the demo](https://youtu.be/T8cjYp5D8lc?si=82cpl-topJZmuHIv)


The demo showcases:
- Brief project explanation
- Model approach
- Working web interface with predictions

---

## 🚀 Steps to Run the Project Locally

### 1. Clone the repository
```bash
git clone <repository-link>
cd AutoJudge-Problem-Difficulty-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the web application
```bash
python app_web.py
```

### 4. Open in browser
```
http://127.0.0.1:5000/
```

---

## 👤 Author Details
**Name:** Devansh Pandey  
**Enrollment Number:** 23113049  
**Branch:** Civil Engineering, 3rd Year  

---

## 📌 Notes
- All trained models are included in the `models/` directory.
- The project runs fully offline without errors.