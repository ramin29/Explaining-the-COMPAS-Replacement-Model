# COMPAS Recidivism Prediction: Explainability Analysis

## Purpose of the Analysis

This project analyzes the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) recidivism prediction algorithm using machine learning explainability methods.

The analysis addresses three main objectives:

1. Model Transparency: Understanding which features drive risk score predictions using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations)

2. Fairness Assessment: Evaluating whether the model exhibits differential behavior across racial and gender groups by examining feature attributions and error rate disparities

3. Counterfactual Analysis: Identifying minimal changes required to flip predictions using DiCE (Diverse Counterfactual Explanations) and flagging cases where immutable features (race, sex) are implicated

Key research questions:
- What features most strongly influence COMPAS risk scores?
- Do SHAP and LIME agree on feature importance? Where do they diverge?
- Are protected attributes (race, sex) required to change predictions?
- What are the governance implications for algorithmic accountability?

---

## Python Libraries Used

Core Data Manipulation:
- pandas (version 1.5.0 or higher) - Data manipulation and analysis
- numpy (version 1.23.0 or higher) - Numerical computations

Visualization:
- matplotlib (version 3.6.0 or higher) - Base plotting library
- seaborn (version 0.12.0 or higher) - Statistical visualizations

Statistical Modeling:
- statsmodels (version 0.13.0 or higher) - Logistic regression with formula interface
- scipy (version 1.9.0 or higher) - Statistical functions

Machine Learning:
- scikit-learn (version 1.1.0 or higher) - Model building, preprocessing, evaluation

Explainability:
- shap (version 0.42.0 or higher) - SHAP values and visualizations
- lime (version 0.2.0 or higher) - Local interpretable explanations
- dice-ml (version 0.10 or higher) - Counterfactual explanations

---

## Instructions for Reproducing the Results

Step 1: Environment Setup

Install required libraries:

    pip install pandas numpy matplotlib seaborn statsmodels scipy scikit-learn shap lime dice-ml

Or check if libraries are already installed:

    pip list

Step 2: Dataset

The analysis uses the COMPAS recidivism dataset from ProPublica.

Load data directly from URL:

    import pandas as pd
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    data = pd.read_csv(url)

Step 3: Running the Analysis

Option A - Jupyter Notebook (Google Colab):
1. Upload the notebook to Google Colab
2. Run the installation cell first: !pip install shap lime dice-ml
3. Execute cells sequentially from top to bottom

Option B - Local Jupyter Notebook:
1. Install Jupyter: pip install jupyter
2. Launch: jupyter notebook
3. Open the .ipynb file and run cells sequentially

Option C - Python Script:
1. Run: python compas_explainability_analysis.py

Step 4: Expected Outputs

The analysis produces:

1. SHAP Analysis:
   - Beeswarm summary plot showing global feature importance
   - Waterfall plots for 4 individuals (highest/lowest risk per racial group)

2. LIME Analysis:
   - Local explanation plots for the same 4 individuals
   - Feature importance comparison with SHAP

3. DiCE Counterfactuals:
   - Minimal feature changes required to flip each prediction
   - Flags for any counterfactuals requiring immutable feature changes

4. Comparison Tables:
   - SHAP vs LIME top feature comparison
   - Counterfactual summary with immutable feature flags

5. Governance Memo:
   - 300-word summary of findings for court auditor

---

## Project Structure

compas-explainability/
    README.md                           - This file
    requirements.txt                    - Python dependencies
    compas_explainability_analysis.py   - Main Python script
    compas_explainability_analysis.ipynb - Jupyter notebook version
    governance_memo.docx                - Governance memo for auditor
    outputs/
        shap_beeswarm.png               - SHAP beeswarm plot
        shap_waterfall.png              - SHAP waterfall plots
        lime_explanations.png           - LIME explanation plots

---

## Notes on Reproducibility

1. Random Seed: Train/test split uses random_state=42 for reproducibility

2. Data Filtering: Analysis filters to African-American and Caucasian defendants only

3. Model: Logistic regression with sklearn pipeline (StandardScaler + OneHotEncoder)

4. Threshold: Classification threshold is 0.5 for predicted probabilities

5. Library Versions: Minor numerical differences may occur with different library versions
