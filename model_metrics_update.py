
# Метрики качества модели (обновлено: 2025-05-25 00:14:18)
MODEL_METRICS = {
    'last_updated': '2025-05-25T00:14:18.807500',
    'model_name': 'XGBoost',
    'accuracy': 0.7116,
    'precision': 0.6586,
    'recall': 0.5479,
    'f1_score': 0.5982,
    'auc_roc': 0.7587,
    'total_features': 70,
    'most_important_features': [{'feature': 'stable_high_consumption', 'importance': 0.06626666337251663}, {'feature': 'high_heating_consumption', 'importance': 0.039019625633955}, {'feature': 'has_zero_months', 'importance': 0.030588330700993538}, {'feature': 'is_city', 'importance': 0.02746652252972126}, {'feature': 'heating_summer_ratio', 'importance': 0.023611215874552727}, {'feature': 'zero_consumption_months', 'importance': 0.022778695449233055}, {'feature': 'is_major_city', 'importance': 0.022476989775896072}, {'feature': 'residentsCount', 'importance': 0.0185711532831192}, {'feature': 'month_5', 'importance': 0.018425162881612778}, {'feature': 'autumn_winter_ratio', 'importance': 0.018293628469109535}],
    'confusion_matrix': [[2398, 537], [855, 1036]],
    'samples_analyzed': 4826,
    'fraud_rate_detected': 0.392
}
