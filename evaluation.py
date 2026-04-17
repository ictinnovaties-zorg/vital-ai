import numpy as np
import pandas as pd
import os
import warnings

"""
### Instructie
- `testset (with label).csv`:
  - Bevat voor elke patiënt de referentie sepsislabels (0 = geen sepsis, 1 = sepsis).
  - Kolommen (ten minste): `Patient_ID`, `SepsisLabel`.

  - `predictions.csv`:
  - Bevat jullie voorspellingen (0 = geen sepsis, 1 = sepsis) voor elke patiënt.
  - Kolommen (ten minste): `Patient_ID`, `SepsisLabel`.

**Belangrijk:**
- De voorspellingen moeten binair zijn.
- Een model dat de referentielabels perfect kan voorspellen, levert niet de maximale score op. De utility score werkt namelijk cumulatief.
Wanneer je tussen uur -12 en -6 de sepsis kan voorspellen, krijg je een hogere score dan wanneer je de sepsis perfect op tijdstip -6 voorspelt.
"""

def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=0, dt_late=3.0, max_u_tp=1, min_u_fn=-2, u_fp=-0.2, u_tn=0, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    return np.sum(u)

def evaluate_sepsis_score(label_csv, prediction_csv,
                               patient_col='Patient_ID',
                               label_header='SepsisLabel',
                               prediction_header='SepsisLabel'):
    """
    Evaluate sepsis prediction performance using flat CSV files.

    Parameters
    ----------
    label_csv : str
        Path to CSV with columns [patient_col, label_header].
    prediction_csv : str
        Path to CSV with columns [patient_col, prediction_header, probability_header].
    patient_col : str
        Column name used to identify individual patients.
    label_header : str
        Column name for ground-truth sepsis labels (0 or 1).
    prediction_header : str
        Column name for predicted labels (0 or 1).

    Returns
    -------
    auroc, auprc, accuracy, f_measure, normalized_observed_utility
    """
    # Scoring parameters.
    dt_early   = -12
    dt_optimal = -6
    dt_late    = 3

    max_u_tp = 1
    min_u_fn = -2
    u_fp     = -0.2
    u_tn     = 0

    # ----- Load data -----
    labels_df = pd.read_csv(label_csv)
    preds_df  = pd.read_csv(prediction_csv)

    required_label_cols = [patient_col, label_header]
    required_pred_cols  = [patient_col, prediction_header]

    for col in required_label_cols:
        if col not in labels_df.columns:
            raise Exception(f'Column "{col}" not found in label CSV.')
    for col in required_pred_cols:
        if col not in preds_df.columns:
            raise Exception(f'Column "{col}" not found in prediction CSV.')

    label_patients = sorted(labels_df[patient_col].unique())
    pred_patients  = sorted(preds_df[patient_col].unique())

    if label_patients != pred_patients:
        raise Exception('Patient IDs in label and prediction CSVs must match exactly.')

    num_files = len(label_patients)

    cohort_labels        = []
    cohort_predictions   = []
    cohort_probabilities = []
 
    for patient_id in label_patients:
        labels        = labels_df.loc[labels_df[patient_col] == patient_id, label_header].to_numpy(dtype=float)
        predictions   = preds_df.loc[preds_df[patient_col] == patient_id, prediction_header].to_numpy(dtype=float)
 
        if not (len(labels) == len(predictions)):
            raise Exception(f'Row counts for patient {patient_id} differ between label and prediction CSVs.')
 
        num_rows = len(labels)
 
        # Validate labels and predictions.
        for i in range(num_rows):
            if labels[i] not in (0, 1):
                raise Exception(f'Label must be 0 or 1 (patient {patient_id}, row {i}).')
            if predictions[i] not in (0, 1):
                raise Exception(f'Prediction must be 0 or 1 (patient {patient_id}, row {i}).')
 
        cohort_labels.append(labels)
        cohort_predictions.append(predictions)
 
    # ----- Utility -----
    observed_utilities = np.zeros(num_files)
    best_utilities     = np.zeros(num_files)
    worst_utilities    = np.zeros(num_files)
    inaction_utilities = np.zeros(num_files)
 
    for k in range(num_files):
        labels = cohort_labels[k]
        num_rows = len(labels)
        observed_predictions = cohort_predictions[k]
 
        best_predictions    = np.zeros(num_rows)
        worst_predictions   = np.zeros(num_rows)
        inaction_predictions = np.zeros(num_rows)
 
        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            best_predictions[max(0, t_sepsis + dt_early): min(t_sepsis + dt_late + 1, num_rows)] = 1
        worst_predictions = 1 - best_predictions
 
        observed_utilities[k] = compute_prediction_utility(
            labels, observed_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        best_utilities[k] = compute_prediction_utility(
            labels, best_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        worst_utilities[k] = compute_prediction_utility(
            labels, worst_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
        inaction_utilities[k] = compute_prediction_utility(
            labels, inaction_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
 
    unnormalized_observed_utility = np.sum(observed_utilities)
    unnormalized_best_utility     = np.sum(best_utilities)
    unnormalized_inaction_utility = np.sum(inaction_utilities)
    
    print(unnormalized_observed_utility)
    print(unnormalized_best_utility)
    print(unnormalized_inaction_utility)
    
    normalized_observed_utility = (
        (unnormalized_observed_utility - unnormalized_inaction_utility) /
        (unnormalized_best_utility     - unnormalized_inaction_utility)
    )
 
    return normalized_observed_utility

if __name__ == '__main__':
    utility = evaluate_sepsis_score('testset (with label).csv', 'predictions.csv')
    print(utility)