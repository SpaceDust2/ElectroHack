"""
⚙️ Конфигурация ML Fraud Detection System
"""

# Пути к данным
DATA_PATHS = {
    'train': 'data/dataset_train.json',
    'test': 'data/dataset_test.json'
}

# Параметры моделей
MODEL_PARAMS = {
    'catboost': {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 3,
        'task_type': 'GPU',  # GPU ускорение
        'random_seed': 42,
        'verbose': False
    },
    'xgboost': {
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'gpu_hist',  # GPU ускорение
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'random_state': 42,
        'n_jobs': -1
    }
}

# Правила детекции мошенников
FRAUD_RULES = {
    'rule1': {
        'description': 'Минимальное потребление > 50 кВт·ч',
        'condition': lambda df: df['min_consumption'] > 50
    },
    'rule2': {
        'description': 'Высокое стабильное потребление',
        'condition': lambda df: (df['avg_consumption'] > 400) & (df['cv'] < 0.5)
    },
    'rule3': {
        'description': 'Отсутствие сезонности + высокое потребление',
        'condition': lambda df: (df['summer_winter_ratio'].between(0.8, 1.2)) & (df['avg_consumption'] > 300)
    }
}

# Уровни риска
RISK_LEVELS = {
    'HIGH': {'threshold': 0.8, 'color': '#ff0000', 'icon': '🚨'},
    'MEDIUM': {'threshold': 0.6, 'color': '#ff9900', 'icon': '⚠️'},
    'LOW': {'threshold': 0.4, 'color': '#ffcc00', 'icon': '📍'},
    'MINIMAL': {'threshold': 0.0, 'color': '#00cc00', 'icon': '✅'}
}

# Визуализация
PLOT_THEME = {
    'template': 'plotly_white',
    'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'font_size': 12
}
