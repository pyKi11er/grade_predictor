# Student Academic Performance Prediction: Regression & Classification with PyTorch MLPs

An end-to-end machine learning pipeline built using **PyTorch** and **NumPy** to predict secondary school students' academic outcomes from demographic, social, and financial features.

The project solves two tasks using dedicated Multi-Layer Perceptron (MLP) architectures:

1. **Continuous Grade Regression:** Predicts the final numerical grade ($G_3 \in [0, 20]$).
2. **Performance Tier Classification:** Classifies student performance into three distinct achievement levels (Low, Moderate, High).

---

## Dataset Overview

Based on the **UCI Machine Learning Repository Student Performance Dataset**, covering social, educational, and demographic attributes:

* **Sample Size:** 395 student profiles.
* **Attributes (33 base variables):** Parental education/jobs, study time, alcohol consumption, absences, past academic failures, and intermediate term grades ($G_1, G_2$).
* **Missing Value Handling:** Missing entries in categorical features (`Mjob`, `Fjob`, `guardian`) are handled by appending binary indicator flags (presence/absence) and zero-imputing raw `NaN` values, expanding the feature space to 35 inputs.

---

## Architecture & Task Formulation

```
                                [35 Preprocessed Features]
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
            Task 1: Regression                              Task 2: Classification
       (Target: Continuous Final Grade)              (Target: Academic Tier Class [0, 1, 2])
                     │                                               │
           Linear(35 → 25) + ReLU                          Linear(33 → 30) + ReLU
                     │                                               │
           Linear(25 → 25) + ReLU                          Linear(30 → 20) + ReLU
                     │                                               │
              Linear(25 → 1)                                  Linear(20 → 3)
                     │                                               │
              Loss: MSELoss                                   Loss: CrossEntropyLoss

```

### 1. Regression Task ($G_3$ Prediction)

* **Inputs:** 35 features (all demographic and historical features, including $G_1, G_2$ and imputation masks).
* **Target:** Continuous final grade $G_3$ ($0\text{--}20$).
* **Loss:** Mean Squared Error ($\text{MSE}$).
* **Architecture:** `35 → 25 (ReLU) → 25 (ReLU) → 1`.

### 2. Multi-Class Classification (Performance Tiers)

Intermediate grades ($G_1, G_2, G_3$) are excluded from model inputs and mapped into categorical labels:

* **Class 0 (Low-Performing):** At least two period grades $< 11$.
* **Class 1 (Moderately Performing):** Average grade $< 14$ (excluding Class 0).
* **Class 2 (High-Performing):** Remaining top-tier students.
* **Loss:** Cross-Entropy Loss ($\text{CE}$).
* **Architecture:** `33 → 30 (ReLU) → 20 (ReLU) → 3`.

---

## Project Structure

```text
student-grade-predictor/
├── data/
│   └── student_data.csv          # Raw / formatted semicolon-delimited dataset
├── student_grade_prediction.ipynb # Unified data prep, regression & classification pipeline
├── .gitignore                     # Excludes __pycache__/, *.pyc, test harnesses
└── README.md

```

---

## Training Setup & Regularization

* **Splits:** 50% Train, 25% Validation, 25% Test (stratified random permutation).
* **Batch Size:** 32 (via PyTorch `DataLoader` with custom `Dataset` wrappers).
* **Optimizer:** Adam ($\text{lr} = 0.001$).
* **Early Stopping:** Configured with patience ($P = 5$) monitoring validation loss to prevent overfitting and restore best checkpoint weights.

---

## Results & Evaluation

* **Regression ($G_3$):** Achieves strong correlation with actual test grades using term history ($G_1, G_2$) as high-signal predictors.
* **Classification:** Evaluated using test accuracy and a $3 \times 3$ confusion matrix across all three performance tiers.

---

## Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/student-grade-predictor.git
cd student-grade-predictor
pip install torch numpy matplotlib

```

### 2. Run the Notebook

```bash
jupyter notebook student_grade_prediction.ipynb

```
