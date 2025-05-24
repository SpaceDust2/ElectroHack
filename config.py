"""
⚙️ Конфигурация системы обнаружения мошенников
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

# Правила детекции нарушителей (обновлены на основе кейса)
FRAUD_RULES = {
    'rule1': {
        'description': 'Высокое потребление в отопительный сезон > 3000 кВт·ч (ключевой критерий)',
        'condition': lambda df: df['heating_season'] > 3000
    },
    'rule2': {
        'description': 'Очень высокое минимальное потребление > 500 кВт·ч',
        'condition': lambda df: df['min_consumption'] > 500
    },
    'rule3': {
        'description': 'Отсутствие сезонности + высокое потребление > 1000 кВт·ч',
        'condition': lambda df: (df['summer_winter_ratio'].between(0.7, 1.3)) & (df['avg_consumption'] > 1000)
    },
    'rule4': {
        'description': 'Сверхстабильное потребление (CV < 0.15) + высокое потребление',
        'condition': lambda df: (df['cv'] < 0.15) & (df['avg_consumption'] > 800)
    },
    'rule5': {
        'description': 'Экстремально высокое потребление на жителя > 1000 кВт·ч',
        'condition': lambda df: df['consumption_per_resident'] > 1000
    }
}

# Уровни риска
RISK_LEVELS = {
    'HIGH': {
        'threshold': 0.8,
        'color': '#ff0000',
        'icon': '🚨',
        'name_ru': 'ВЫСОКИЙ',
        'description': 'Очень высокая вероятность коммерческого использования'
    },
    'MEDIUM': {
        'threshold': 0.6,
        'color': '#ff9900',
        'icon': '⚠️',
        'name_ru': 'СРЕДНИЙ',
        'description': 'Требуется дополнительная проверка'
    },
    'LOW': {
        'threshold': 0.4,
        'color': '#ffcc00',
        'icon': '📍',
        'name_ru': 'НИЗКИЙ',
        'description': 'Некоторые подозрительные признаки'
    },
    'MINIMAL': {
        'threshold': 0.0,
        'color': '#00cc00',
        'icon': '✅',
        'name_ru': 'МИНИМАЛЬНЫЙ',
        'description': 'Соответствует бытовому использованию'
    }
}

# Настройки визуализации
PLOT_THEME = {
    'template': 'plotly_white',
    'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'font_size': 12
}

# 🛡️ НАСТРОЙКИ КОНСЕРВАТИВНОЙ КЛАССИФИКАЦИИ ДЛЯ ЗАЩИТЫ ЧЕСТНЫХ ЖИТЕЛЕЙ
CLASSIFICATION_THRESHOLDS = {
    'default': 0.5,           # Стандартный порог
    'conservative': 0.7,      # Консервативный - меньше ложных обвинений
    'ultra_conservative': 0.85,  # Ультра-консервативный - максимальная защита
    'custom': 0.8             # Пользовательский порог
}

# 🤝 НАСТРОЙКИ АНСАМБЛЯ И КОНСЕНСУСА
ENSEMBLE_SETTINGS = {
    'require_consensus': True,        # Требовать согласия моделей
    # Доля моделей которые должны согласиться (2 из 3)
    'consensus_threshold': 0.67,
    'min_models_agree': 2,           # Минимум моделей которые должны согласиться
    # Усреднять вероятности или использовать голосование
    'use_probability_averaging': True,
    'individual_thresholds': {       # Пороги для каждой модели отдельно
        'CatBoost': 0.75,
        'XGBoost': 0.8,
        'RandomForest': 0.85
    }
}

# 🏠 ЗАЩИЩЕННЫЕ КАТЕГОРИИ (дополнительная осторожность)
PROTECTED_CATEGORIES = {
    'enable_protection': True,
    'categories': {
        'elderly_threshold_age': 65,           # Возраст для льготной категории
        'large_family_threshold': 4,           # Количество жителей для многодетной семьи
        # Площадь маленькой квартиры (м²)
        'small_apartment_threshold': 35,
        'low_income_consumption': 100          # Низкое потребление (льготники)
    },
    # Увеличиваем порог на 20% для защищенных
    'protection_multiplier': 1.2,
    'additional_threshold': 0.05              # Дополнительные 5% к порогу
}

# ⚖️ НАСТРОЙКИ СТОИМОСТИ ОШИБОК (Cost-Sensitive Learning)
ERROR_COSTS = {
    'false_positive_cost': 100,      # Стоимость ложного обвинения честного
    'false_negative_cost': 10,       # Стоимость пропуска мошенника
    'true_positive_reward': -20,     # Награда за поимку мошенника
    'true_negative_reward': 0        # Награда за правильное определение честного
}

# 🔍 УРОВНИ ОСТОРОЖНОСТИ КЛАССИФИКАЦИИ
CAUTION_LEVELS = {
    'aggressive': {
        'threshold': 0.5,
        'description': 'Агрессивная детекция - больше находим, больше ошибаемся',
        'expected_precision': 0.75
    },
    'balanced': {
        'threshold': 0.65,
        'description': 'Сбалансированный подход - компромисс точности и полноты',
        'expected_precision': 0.85
    },
    'conservative': {
        'threshold': 0.8,
        'description': 'Консервативная детекция - меньше ошибок, больше пропусков',
        'expected_precision': 0.92
    },
    'ultra_safe': {
        'threshold': 0.9,
        'description': 'Максимальная защита честных - только очевидные случаи',
        'expected_precision': 0.97
    }
}
