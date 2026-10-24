import joblib

# Save trained models to disk
joblib.dump(rf_cc, '../models/rf_cc_model.joblib')
joblib.dump(rf_fraud, '../models/rf_fraud_model.joblib')

# Optionally, save logistic regression models as well
joblib.dump(lr_cc, '../models/lr_cc_model.joblib')
joblib.dump(lr_fraud, '../models/lr_fraud_model.joblib')
