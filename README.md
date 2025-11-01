## 💼 **Expected CTC Prediction App ( LaunchED Capstone Project)**  

Predicting **Expected CTC (salary)** for job applicants using advanced **Machine Learning**.  
This project builds a regression model that assists HR professionals in offering **fair and consistent salaries** based on candidate profiles — minimizing **human bias** and improving **data-driven compensation decisions**.

---

### 🏷️ **Project Status**
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![NumPy](https://img.shields.io/badge/Library-NumPy-lightblue?logo=numpy)
![Pandas](https://img.shields.io/badge/Library-Pandas-yellow?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Viz-Matplotlib-green?logo=plotly)
![Seaborn](https://img.shields.io/badge/Viz-Seaborn-teal?logo=seaborn)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red?logo=xgboost)
![Streamlit](https://img.shields.io/badge/App-Streamlit-magenta?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🧠 **Project Overview**

The **Expected CTC Prediction App** predicts the salary that a job applicant expects based on their education, skills, experience, and past performance data.  
The model uses multiple **machine learning algorithms** — including **Linear Regression**, **Random Forest**, and **XGBoost** — with **XGBoost** emerging as the best-performing model.  

It’s deployed as an **interactive Streamlit web app** that enables recruiters to input applicant details and instantly view a predicted expected salary.

---

## 🎯 **Objective**
- Develop an ML regression model to predict **Expected CTC (Cost to Company)** for candidates.  
- Leverage key profile features like experience, education, appraisal ratings, and certifications.  
- Help HR and recruitment teams make unbiased, data-supported salary offers.  
- Deploy the model via Streamlit for real-time, user-friendly access.

---

## 📂 **Dataset Summary**

The dataset includes **25,000 records** and **29 attributes** describing applicant profiles.  

| Feature Type | Examples |
|---------------|-----------|
| 👩‍💼 Job Details | Department, Role, Industry, Organization |
| 🎓 Education | Graduation, Post-Graduation, PhD Specializations |
| 💼 Experience | Total Experience, Field Experience, Current CTC |
| ⭐ Achievements | Publications, Certifications, Appraisal Rating |
| 🗺️ Location | Current Location, Preferred Location |
| 🎯 Target | **Expected_CTC (Salary to be predicted)** |

---

## ⚙️ **Tech Stack and Libraries**

### 🧩 **Programming Language**
- **Python 3.10**

### 🧮 **Core Libraries**
- **NumPy** → Numerical operations and array manipulations  
- **Pandas** → Data handling, cleaning, preprocessing  
- **Matplotlib** & **Seaborn** → Exploratory Data Analysis (EDA) & data visualization  

### 🤖 **Machine Learning**
- **Scikit-Learn** → Data preprocessing, model training, evaluation metrics  
- **XGBoost** → Tuned gradient boosting model for high accuracy  
- **Random Forest & Linear Regression** → Baseline regression models for comparison  

### 🚀 **Deployment**
- **Streamlit** → Interactive web app for real-time CTC prediction  
- **Pickle** → Model serialization for deployment  
- **VS Code / Google Colab** → Development environment  

---

## 🧩 **Project Workflow**

### **1️⃣ Data Preprocessing & EDA**
- Inspected dataset structure (`df.info()`, `df.describe()`)  
- Handled **missing values** using mean/median/mode imputation  
- Detected and **capped outliers** using the IQR method  
- Encoded **categorical variables** using one-hot encoding  
- Performed **feature scaling** using `StandardScaler`  
- Conducted **univariate & bivariate analysis** using Seaborn & Matplotlib  

### **2️⃣ Model Building**
Tested multiple regression models:
- 📈 **Linear Regression**
- 🌳 **Random Forest Regressor**
- 🚀 **XGBoost Regressor** *(Tuned model chosen for deployment)*  

**Best Model Results (XGBoost):**
| Metric | Score |
|---------|-------|
| R² Score | 0.9997 |
| MAE | ₹9,754 |
| RMSE | ₹20,281 |

### **3️⃣ Model Deployment (Streamlit)**
- Created interactive UI for candidate data input  
- Included dropdowns for Department, Role, Industry, Education, Location, etc.  
- Used saved model and scaler (`expected_ctc_model.pkl`, `scaler.pkl`)  
- Displayed live predictions with download option  

---

## 💻 **How to Run the Project Locally**

### 🪄 1. Clone the Repository
```bash
git clone https://github.com/safwaan-exe/LaunchED-Capstone-Project.git
cd LaunchED-Capstone-Project
```

---

### 🏫 **Project Context**
This project was developed as part of the **LaunchED Capstone Project**, focusing on real-world applications of machine learning in HR analytics.

---


### 💬 **Quote**
> *“Data-driven hiring decisions lead to fairer outcomes — and machine learning makes that possible.”*

---

### 👤 **Author**
**Safwaan Siddiqui**  
[GitHub Profile](https://github.com/safwaan-exe)
