# Loan Approval Prediction System (corrected model package)

This package keeps the supplied Streamlit loan approval interface and aligns its model artifacts.

## Files
- `app.py`: Streamlit interface.
- `train_model.py`: reproducible model training and artifact export.
- `best_model.pkl`, `scaler.pkl`, `encoder.pkl`, `feature_columns.pkl`, `best_model_name.pkl`: matching trained artifacts.
- `metrics.json`: held-out test metrics for the three candidate models.
- `loan_approval_dataset.csv`: supplied dataset.
- `notebook.ipynb`: supplied notebook.

## Install and run
```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain and regenerate all artifacts:
```bash
python train_model.py
```

## Training result
The supplied notebook's split and model settings reproduce the following held-out test results:
- Random Forest: accuracy 97.78%, precision 97.50%, recall 96.59%, F1 97.05%, ROC-AUC 0.9977.
- Decision Tree: accuracy 97.19%, precision 99.02%, recall 93.50%, F1 96.18%, ROC-AUC 0.9937.
- Logistic Regression: accuracy 92.27%, precision 91.59%, recall 87.62%, F1 89.56%, ROC-AUC 0.9745.

`train_model.py` selects by F1 score, then refits the selected model and scaler on the full dataset for deployment. The metrics remain calculated on the held-out test split.

**Important:** This is an educational project, not a real lending decision system. Test performance on this dataset does not guarantee performance on new applicants. Do not use it for actual loan decisions.
