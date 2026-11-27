# PROJECT INSTRUCTIONS — PDA FINAL PROJECT

## 1. Project Identity

Project: Breast Tumor Classification System

Research topic:
Pengembangan Sistem Klasifikasi Tumor Payudara Berdasarkan Karakteristik Inti Sel Menggunakan Random Forest dan Support Vector Machine.

Primary technologies:
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Plotly
- SQLite

The project is an existing research codebase. Your responsibility is to complete and integrate the existing implementation into a working Streamlit application.

Do not rebuild the project from scratch.

## 2. Existing Architecture

Existing modules:

- app.py
- train.py
- data/breast_cancer.csv
- preprocessing/
- random_forest/
- svm/
- evaluasi/
- interpretasi_shap/
- basis_data/

Inspect the actual source code before assuming what each module does.

## 3. Research Integrity

The original research methodology must be preserved.

Do not independently change:
- Model algorithms
- Model hyperparameters
- Feature selection
- Preprocessing methodology
- Training and testing split
- Cross-validation strategy
- Evaluation metrics
- Positive class definition
- SHAP methodology

Any proposed methodological change must be explained and approved first.

Never fabricate evaluation results or model performance.

Never introduce data leakage.

All preprocessing transformations must be fitted on training data only and reused consistently during inference.

## 4. Coding Principles

- Follow PEP 8.
- Prefer simple and modular code.
- Reuse existing functions.
- Avoid duplicate implementations.
- Keep business logic separate from Streamlit UI.
- Validate input data.
- Handle errors explicitly.
- Preserve consistent feature names and ordering.
- Avoid unnecessary dependencies.
- Do not retrain models on every Streamlit rerun.
- Use caching only where it is safe and appropriate.

## 5. File Safety

Before modifying a file:
1. Read the existing file.
2. Understand its dependencies.
3. Explain the intended change.
4. Identify possible side effects.

Never delete, rename, or overwrite files without approval.

Never modify the dataset unless explicitly instructed.

Do not replace existing research implementations with simplified alternatives.

## 6. Development Workflow

Work in small, verifiable stages:

1. Project audit
2. Backend integration
3. Model persistence
4. Streamlit interface
5. SHAP integration
6. Database integration
7. Testing
8. Documentation

At the end of every stage, report:
- Files inspected
- Files modified
- Functions added or changed
- Commands executed
- Test results
- Remaining issues

Do not proceed to the next major stage without approval.

## 7. Testing

Verify:
- Data loading
- Missing-value handling
- Feature validation
- Feature ordering
- Model loading
- Random Forest prediction
- SVM prediction
- Probability output
- Evaluation metrics
- SHAP output
- Database insertion and retrieval
- Streamlit application startup

Do not claim a test passed unless you actually ran it.

## 8. Medical Disclaimer

This application is a research and educational prototype.

It is not a clinical diagnostic tool and must not replace professional medical evaluation.

## 9. Communication

Explain technical decisions in Indonesian.

Use actual file names and function names when reporting findings.

Clearly distinguish:
- Verified facts
- Assumptions
- Unknown information
- Recommendations

Prioritize correctness, reproducibility, and research integrity over implementation speed.