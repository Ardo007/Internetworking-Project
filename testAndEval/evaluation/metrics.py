from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_eval_metrics(y_true, y_pred):
    """
    Calculates key evaluation metrics for binary classification models.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'true_positive': int(tp),
        'false_positive': int(fp),
        'true_negative': int(tn),
        'false_negative': int(fn)
    }