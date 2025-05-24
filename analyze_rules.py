"""
🔍 Анализ данных для формирования правил обнаружения нарушителей
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_and_analyze_dataset(file_path: str):
    """Загрузка и первичный анализ датасета"""
    print(f"📁 Загружаем данные из {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ Загружено {len(data)} записей")

    # Преобразуем в DataFrame для анализа
    rows = []
    for item in data:
        row = {
            'accountId': item.get('accountId'),
            'buildingType': item.get('buildingType'),
            'roomsCount': item.get('roomsCount'),
            'residentsCount': item.get('residentsCount'),
            'totalArea': item.get('totalArea'),
            'isCommercial': item.get('isCommercial', False)
        }

        # Извлекаем потребление по месяцам
        consumption = item.get('consumption', {})
        for month in range(1, 13):
            row[f'month_{month}'] = consumption.get(str(month), 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def calculate_consumption_features(df):
    """Вычисление характеристик потребления"""
    print("🔧 Вычисляем характеристики потребления...")

    # Месячные данные
    month_cols = [f'month_{i}' for i in range(1, 13)]

    # Основные статистики
    df['avg_consumption'] = df[month_cols].mean(axis=1)
    df['min_consumption'] = df[month_cols].min(axis=1)
    df['max_consumption'] = df[month_cols].max(axis=1)
    df['std_consumption'] = df[month_cols].std(axis=1)
    df['cv'] = df['std_consumption'] / \
        df['avg_consumption']  # Коэффициент вариации

    # Сезонные характеристики
    df['winter_avg'] = df[['month_12', 'month_1', 'month_2']].mean(
        axis=1)  # Декабрь-Февраль
    df['spring_avg'] = df[['month_3', 'month_4', 'month_5']].mean(
        axis=1)   # Март-Май
    df['summer_avg'] = df[['month_6', 'month_7', 'month_8']].mean(
        axis=1)   # Июнь-Август
    df['autumn_avg'] = df[['month_9', 'month_10', 'month_11']].mean(
        axis=1)  # Сентябрь-Ноябрь

    # Отопительный сезон (октябрь-апрель) - ключевой критерий!
    df['heating_season'] = df[['month_10', 'month_11', 'month_12',
                               'month_1', 'month_2', 'month_3', 'month_4']].mean(axis=1)

    # Соотношения
    df['summer_winter_ratio'] = df['summer_avg'] / (df['winter_avg'] + 1e-8)
    df['heating_summer_ratio'] = df['heating_season'] / \
        (df['summer_avg'] + 1e-8)

    # Нулевые месяцы
    df['zero_months'] = (df[month_cols] == 0).sum(axis=1)
    df['months_with_data'] = (df[month_cols] > 0).sum(axis=1)

    # Потребление на жителя и площадь
    df['consumption_per_resident'] = df['avg_consumption'] / \
        (df['residentsCount'] + 1e-8)
    df['consumption_per_area'] = df['avg_consumption'] / \
        (df['totalArea'].fillna(df['roomsCount'] * 17.5) + 1e-8)

    # ДОПОЛНИТЕЛЬНЫЕ НЕОЧЕВИДНЫЕ ПРИЗНАКИ

    # Тренды и паттерны
    df['q1_avg'] = df[['month_1', 'month_2', 'month_3']].mean(axis=1)
    df['q2_avg'] = df[['month_4', 'month_5', 'month_6']].mean(axis=1)
    df['q3_avg'] = df[['month_7', 'month_8', 'month_9']].mean(axis=1)
    df['q4_avg'] = df[['month_10', 'month_11', 'month_12']].mean(axis=1)

    # Межквартальная стабильность
    quarter_cols = ['q1_avg', 'q2_avg', 'q3_avg', 'q4_avg']
    df['quarter_stability'] = df[quarter_cols].std(
        axis=1) / (df[quarter_cols].mean(axis=1) + 1e-8)

    # Пиковые месяцы    avg_threshold_high = df['avg_consumption'].values[:, np.newaxis] * 1.5    avg_threshold_low = df['avg_consumption'].values[:, np.newaxis] * 0.5    df['peak_months'] = (df[month_cols].values > avg_threshold_high).sum(axis=1)    df['low_months'] = (df[month_cols].values < avg_threshold_low).sum(axis=1)

    # Последовательные высокие месяцы
    high_threshold = df['avg_consumption'] * 1.2
    df['consecutive_high'] = 0
    for idx in df.index:
        max_consecutive = 0
        current_consecutive = 0
        for month in range(1, 13):
            if df.loc[idx, f'month_{month}'] > high_threshold.loc[idx]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        df.loc[idx, 'consecutive_high'] = max_consecutive

    # Экстремальные соотношения
    df['max_min_ratio'] = df['max_consumption'] / \
        (df['min_consumption'] + 1e-8)
    df['median_mean_ratio'] = df[month_cols].median(
        axis=1) / (df['avg_consumption'] + 1e-8)

    # Энтропия потребления (мера предсказуемости)
    def calculate_entropy(row):
        values = row[month_cols].values + 1e-8  # Избегаем log(0)
        probabilities = values / values.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    df['consumption_entropy'] = df.apply(calculate_entropy, axis=1)

    return df


def discover_hidden_patterns(df):
    """Поиск скрытых паттернов и неочевидных правил"""
    print("\n🕵️ ПОИСК СКРЫТЫХ ПАТТЕРНОВ")
    print("=" * 60)

    commercial = df[df['isCommercial'] == True]
    residential = df[df['isCommercial'] == False]

    hidden_patterns = {}

    # 1. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
    print("\n🔗 Корреляционный анализ:")

    # Выбираем числовые признаки
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['accountId']]

    correlation_with_target = df[numeric_cols + ['isCommercial']
                                 ].corr()['isCommercial'].abs().sort_values(ascending=False)

    print("   Топ-10 признаков по корреляции с мошенничеством:")
    for i, (feature, corr) in enumerate(correlation_with_target.head(11).items()):
        if feature != 'isCommercial':
            print(f"   {i}. {feature}: {corr:.3f}")

    hidden_patterns['top_correlations'] = correlation_with_target.head(
        11).to_dict()

    # 2. МАШИННОЕ ОБУЧЕНИЕ ДЛЯ ПОИСКА ВАЖНЫХ ПРИЗНАКОВ
    print("\n🤖 ML-анализ важности признаков:")

    # Подготовка данных
    X = df[numeric_cols].fillna(0)
    y = df['isCommercial'].astype(int)

    # Random Forest для определения важности
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("   Топ-10 признаков по важности (Random Forest):")
    for i, row in enumerate(feature_importance.head(10).iterrows()):
        feature, importance = row[1]['feature'], row[1]['importance']
        print(f"   {i+1}. {feature}: {importance:.3f}")

    hidden_patterns['rf_importance'] = feature_importance.head(
        15).to_dict('records')

    # 3. СТАТИСТИЧЕСКИЙ АНАЛИЗ РАЗЛИЧИЙ
    print("\n📊 Статистические различия между группами:")

    from scipy import stats

    significant_features = []
    for feature in numeric_cols:
        if feature not in ['isCommercial']:
            comm_values = commercial[feature].dropna()
            res_values = residential[feature].dropna()

            if len(comm_values) > 0 and len(res_values) > 0:
                # t-test
                t_stat, p_value = stats.ttest_ind(
                    comm_values, res_values, equal_var=False)

                if p_value < 0.01:  # Значимость 1%
                    effect_size = (comm_values.mean() - res_values.mean()) / \
                        np.sqrt((comm_values.var() + res_values.var()) / 2)
                    significant_features.append({
                        'feature': feature,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'comm_mean': comm_values.mean(),
                        'res_mean': res_values.mean()
                    })

    significant_features = sorted(
        significant_features, key=lambda x: x['p_value'])

    print("   Наиболее значимые различия:")
    for i, feat in enumerate(significant_features[:10]):
        print(
            f"   {i+1}. {feat['feature']}: p={feat['p_value']:.2e}, effect={feat['effect_size']:.2f}")

    hidden_patterns['significant_differences'] = significant_features[:15]

    # 4. КОМБИНАЦИОННЫЕ ПРАВИЛА
    print("\n🔄 Поиск комбинационных правил:")

    # Попробуем найти эффективные комбинации правил
    combo_rules = []

    # Правило: высокое + стабильное
    mask1 = (df['avg_consumption'] >
             df['avg_consumption'].quantile(0.8)) & (df['cv'] < 0.3)
    precision1, recall1 = calculate_rule_metrics(df, mask1)
    combo_rules.append({
        'rule': 'Высокое потребление + стабильность',
        'condition': 'avg_consumption > 80% квантиль AND cv < 0.3',
        'precision': precision1,
        'recall': recall1,
        'count': mask1.sum()
    })

    # Правило: отопительный сезон + нет пиков
    mask2 = (df['heating_season'] > 2000) & (df['peak_months'] < 2)
    precision2, recall2 = calculate_rule_metrics(df, mask2)
    combo_rules.append({
        'rule': 'Высокий отопительный сезон + нет пиков',
        'condition': 'heating_season > 2000 AND peak_months < 2',
        'precision': precision2,
        'recall': recall2,
        'count': mask2.sum()
    })

    # Правило: энтропия + последовательность
    mask3 = (df['consumption_entropy'] < df['consumption_entropy'].quantile(
        0.3)) & (df['consecutive_high'] > 6)
    precision3, recall3 = calculate_rule_metrics(df, mask3)
    combo_rules.append({
        'rule': 'Низкая энтропия + долгие высокие периоды',
        'condition': 'consumption_entropy < 30% квантиль AND consecutive_high > 6',
        'precision': precision3,
        'recall': recall3,
        'count': mask3.sum()
    })

    combo_rules = sorted(
        combo_rules, key=lambda x: x['precision'] * x['recall'], reverse=True)

    print("   Лучшие комбинационные правила:")
    for i, rule in enumerate(combo_rules):
        print(f"   {i+1}. {rule['rule']}")
        print(
            f"      Точность: {rule['precision']:.2%}, Покрытие: {rule['recall']:.2%}")
        print(f"      Условие: {rule['condition']}")

    hidden_patterns['combination_rules'] = combo_rules

    # 5. АНОМАЛЬНЫЕ ПАТТЕРНЫ
    print("\n🚨 Аномальные паттерны:")

    anomaly_patterns = []

    # Аномалия: нулевое потребление + высокие другие месяцы
    zero_but_high = df[(df['zero_months'] > 0) &
                       (df['max_consumption'] > 1000)]
    if len(zero_but_high) > 0:
        fraud_rate = zero_but_high['isCommercial'].mean()
        anomaly_patterns.append({
            'pattern': 'Нулевые месяцы + высокие пики',
            'count': len(zero_but_high),
            'fraud_rate': fraud_rate,
            'description': 'Объекты с нулевым потреблением в некоторые месяцы, но высоким в другие'
        })

    # Аномалия: очень низкая энтропия
    low_entropy = df[df['consumption_entropy'] <
                     df['consumption_entropy'].quantile(0.1)]
    if len(low_entropy) > 0:
        fraud_rate = low_entropy['isCommercial'].mean()
        anomaly_patterns.append({
            'pattern': 'Сверхпредсказуемое потребление',
            'count': len(low_entropy),
            'fraud_rate': fraud_rate,
            'description': 'Крайне предсказуемые паттерны потребления'
        })

    for pattern in anomaly_patterns:
        print(
            f"   📍 {pattern['pattern']}: {pattern['count']} объектов, {pattern['fraud_rate']:.1%} мошенники")

    hidden_patterns['anomaly_patterns'] = anomaly_patterns

    return hidden_patterns


def calculate_rule_metrics(df, mask):
    """Вычисление точности и покрытия правила"""
    if mask.sum() == 0:
        return 0, 0

    tp = (mask & df['isCommercial']).sum()
    fp = (mask & ~df['isCommercial']).sum()
    fn = (~mask & df['isCommercial']).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


def analyze_commercial_patterns(df):
    """Анализ паттернов коммерческого использования"""
    print("\n🔍 АНАЛИЗ КОММЕРЧЕСКИХ ПАТТЕРНОВ")
    print("=" * 60)

    commercial = df[df['isCommercial'] == True]
    residential = df[df['isCommercial'] == False]

    print(f"📊 Коммерческих объектов: {len(commercial)}")
    print(f"📊 Жилых объектов: {len(residential)}")
    print(f"📊 Доля коммерческих: {len(commercial)/len(df)*100:.1f}%")

    print("\n🔥 КЛЮЧЕВЫЕ РАЗЛИЧИЯ:")
    print("-" * 40)

    features_to_analyze = [
        'avg_consumption', 'min_consumption', 'max_consumption',
        'heating_season', 'summer_avg', 'winter_avg',
        'cv', 'summer_winter_ratio', 'heating_summer_ratio',
        'consumption_per_resident', 'consumption_per_area',
        'zero_months', 'quarter_stability', 'consumption_entropy',
        'consecutive_high', 'peak_months'
    ]

    results = {}

    for feature in features_to_analyze:
        if feature in df.columns:
            comm_mean = commercial[feature].mean()
            res_mean = residential[feature].mean()
            comm_median = commercial[feature].median()
            res_median = residential[feature].median()

            print(f"\n📈 {feature}:")
            print(
                f"   Коммерческие: среднее={comm_mean:.1f}, медиана={comm_median:.1f}")
            print(
                f"   Жилые:        среднее={res_mean:.1f}, медиана={res_median:.1f}")
            print(f"   Разница:      {(comm_mean/res_mean - 1)*100:+.1f}%")

            results[feature] = {
                'commercial_mean': comm_mean,
                'residential_mean': res_mean,
                'commercial_median': comm_median,
                'residential_median': res_median,
                'ratio': comm_mean / res_mean if res_mean > 0 else float('inf')
            }

    return results


def find_optimal_thresholds(df):
    """Поиск оптимальных пороговых значений"""
    print("\n🎯 ПОИСК ОПТИМАЛЬНЫХ ПОРОГОВЫХ ЗНАЧЕНИЙ")
    print("=" * 60)

    commercial = df[df['isCommercial'] == True]
    residential = df[df['isCommercial'] == False]

    rules = {}

    # 1. Правило по отопительному сезону (ключевое!)
    heating_values = df['heating_season'].values
    comm_heating = commercial['heating_season'].values
    res_heating = residential['heating_season'].values

    # Попробуем разные пороги
    thresholds = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000]
    best_threshold = None
    best_score = 0

    print("\n🌡️ Анализ порога для отопительного сезона:")
    for threshold in thresholds:
        tp = np.sum((comm_heating >= threshold))  # True Positive
        fp = np.sum((res_heating >= threshold))   # False Positive
        fn = np.sum((comm_heating < threshold))   # False Negative

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        print(
            f"   {threshold:4d} кВт·ч: точность={precision:.3f}, покрытие={recall:.3f}, F1={f1:.3f}")

        if f1 > best_score:
            best_score = f1
            best_threshold = threshold

    rules['heating_season_threshold'] = {
        'value': best_threshold,
        'description': f'Среднее потребление в отопительный сезон > {best_threshold} кВт·ч',
        'precision': precision,
        'recall': recall,
        'f1': best_score
    }

    # 2. Автоматический поиск порогов для других признаков
    features_to_optimize = [
        'min_consumption', 'consumption_per_resident', 'cv',
        'consumption_entropy', 'consecutive_high', 'quarter_stability'
    ]

    for feature in features_to_optimize:
        if feature in df.columns:
            best_thresh, best_f1, best_prec, best_rec = optimize_threshold(
                df, feature)
            rules[f'{feature}_threshold'] = {
                'value': best_thresh,
                'description': f'{feature} оптимальный порог',
                'precision': best_prec,
                'recall': best_rec,
                'f1': best_f1
            }

    return rules


def optimize_threshold(df, feature):
    """Оптимизация порога для признака"""
    commercial = df[df['isCommercial'] == True]
    residential = df[df['isCommercial'] == False]

    comm_values = commercial[feature].dropna()
    res_values = residential[feature].dropna()

    if len(comm_values) == 0 or len(res_values) == 0:
        return 0, 0, 0, 0

    # Пробуем пороги от 10% до 90% квантилей
    percentiles = np.arange(10, 91, 5)
    best_f1 = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0

    for p in percentiles:
        threshold = np.percentile(df[feature].dropna(), p)

        # Для некоторых признаков логика обратная (меньше = подозрительно)
        if feature in ['cv', 'consumption_entropy', 'quarter_stability']:
            mask = df[feature] <= threshold
        else:
            mask = df[feature] >= threshold

        precision, recall = calculate_rule_metrics(df, mask)
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_threshold, best_f1, best_precision, best_recall


def test_rules_effectiveness(df, rules):
    """Тестирование эффективности правил"""
    print("\n🧪 ТЕСТИРОВАНИЕ ЭФФЕКТИВНОСТИ ПРАВИЛ")
    print("=" * 60)

    commercial = df[df['isCommercial'] == True]
    residential = df[df['isCommercial'] == False]

    # Правило 1: Отопительный сезон
    rule1_mask = df['heating_season'] >= rules['heating_season_threshold']['value']
    tp1 = np.sum(rule1_mask & (df['isCommercial'] == True))
    fp1 = np.sum(rule1_mask & (df['isCommercial'] == False))
    precision1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0
    recall1 = tp1 / len(commercial) if len(commercial) > 0 else 0

    print(
        f"📏 Правило 1 - Отопительный сезон > {rules['heating_season_threshold']['value']} кВт·ч:")
    print(f"   Найдено коммерческих: {tp1}/{len(commercial)} ({recall1:.1%})")
    print(
        f"   Ложных срабатываний: {fp1}/{len(residential)} ({fp1/len(residential):.1%})")
    print(f"   Точность: {precision1:.1%}")

    effectiveness = {
        'rule1': {'precision': precision1, 'recall': recall1, 'caught': tp1}
    }

    # Добавляем другие оптимизированные правила
    rule_counter = 2
    for rule_name, rule_info in rules.items():
        if rule_name != 'heating_season_threshold' and '_threshold' in rule_name:
            feature = rule_name.replace('_threshold', '')
            if feature in df.columns:
                # Применяем правило
                if feature in ['cv', 'consumption_entropy', 'quarter_stability']:
                    mask = df[feature] <= rule_info['value']
                    comparison = '<='
                else:
                    mask = df[feature] >= rule_info['value']
                    comparison = '>='

                tp = np.sum(mask & (df['isCommercial'] == True))
                fp = np.sum(mask & (df['isCommercial'] == False))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / len(commercial) if len(commercial) > 0 else 0

                print(
                    f"\n📏 Правило {rule_counter} - {feature} {comparison} {rule_info['value']:.2f}:")
                print(
                    f"   Найдено коммерческих: {tp}/{len(commercial)} ({recall:.1%})")
                print(
                    f"   Ложных срабатываний: {fp}/{len(residential)} ({fp/len(residential):.1%})")
                print(f"   Точность: {precision:.1%}")

                effectiveness[f'rule{rule_counter}'] = {
                    'precision': precision, 'recall': recall, 'caught': tp,
                    'feature': feature, 'threshold': rule_info['value']
                }
                rule_counter += 1

    return effectiveness


def generate_config_code(rules, hidden_patterns):
    """Генерация кода для config.py с учетом скрытых паттернов"""
    print("\n💾 ГЕНЕРАЦИЯ ОБНОВЛЕННОГО КОДА ДЛЯ CONFIG.PY")
    print("=" * 60)

    config_code = f"""
# Обновленные правила детекции на основе глубокого анализа данных
FRAUD_RULES = {{
    'rule1': {{
        'description': 'Высокое потребление в отопительный сезон > {rules['heating_season_threshold']['value']} кВт·ч (ключевой критерий)',
        'condition': lambda df: df['heating_season'] > {rules['heating_season_threshold']['value']}
    }},"""

    rule_counter = 2
    for rule_name, rule_info in rules.items():
        if rule_name != 'heating_season_threshold' and '_threshold' in rule_name:
            feature = rule_name.replace('_threshold', '')
            if feature in ['cv', 'consumption_entropy', 'quarter_stability']:
                operator = '<'
                comparison = 'lambda df: df[\'{}\'] < {:.3f}'.format(
                    feature, rule_info['value'])
            else:
                operator = '>'
                comparison = 'lambda df: df[\'{}\'] > {:.3f}'.format(
                    feature, rule_info['value'])

            config_code += f"""
    'rule{rule_counter}': {{
        'description': '{feature} {operator} {rule_info["value"]:.3f} (F1={rule_info["f1"]:.3f})',
        'condition': {comparison}
    }},"""
            rule_counter += 1

    # Добавляем комбинационные правила из скрытых паттернов
    if 'combination_rules' in hidden_patterns:
        # Топ-3
        for i, combo_rule in enumerate(hidden_patterns['combination_rules'][:3]):
            config_code += f"""
    'combo_rule{i+1}': {{
        'description': '{combo_rule["rule"]} (точность={combo_rule["precision"]:.2%})',
        'condition': lambda df: {combo_rule["condition"].replace('AND', '&').replace('OR', '|')}
    }},"""

    config_code += """
}

# Метрики качества модели (будут обновляться при обучении)
MODEL_METRICS = {
    'last_updated': None,
    'accuracy': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'f1_score': 0.0,
    'auc_roc': 0.0,
    'total_features': 0,
    'most_important_features': [],
    'confusion_matrix': None
}

# Скрытые паттерны, обнаруженные в данных
HIDDEN_PATTERNS = """ + str(hidden_patterns) + """
"""

    print(config_code)
    return config_code


def main():
    """Основная функция анализа"""
    print("🔍 ГЛУБОКИЙ АНАЛИЗ ДАННЫХ ДЛЯ ФОРМИРОВАНИЯ ПРАВИЛ ОБНАРУЖЕНИЯ")
    print("=" * 80)

    # Пути к данным
    train_path = "data/dataset_train.json"

    if not Path(train_path).exists():
        print(f"❌ Файл {train_path} не найден!")
        print("📥 Пожалуйста, скачайте данные по ссылкам из кейса и поместите в папку data/")
        return

    try:
        # 1. Загрузка данных
        df = load_and_analyze_dataset(train_path)

        # 2. Вычисление характеристик (включая новые признаки)
        df = calculate_consumption_features(df)

        # 3. Поиск скрытых паттернов !!!
        hidden_patterns = discover_hidden_patterns(df)

        # 4. Анализ основных паттернов
        patterns = analyze_commercial_patterns(df)

        # 5. Поиск оптимальных порогов
        rules = find_optimal_thresholds(df)

        # 6. Тестирование правил
        effectiveness = test_rules_effectiveness(df, rules)

        # 7. Генерация обновленного кода
        config_code = generate_config_code(rules, hidden_patterns)

        # 8. Сохранение результатов
        results = {
            'rules': rules,
            'effectiveness': effectiveness,
            'patterns': patterns,
            'hidden_patterns': hidden_patterns,
            'config_code': config_code,
            'dataset_stats': {
                'total_objects': len(df),
                'commercial_objects': df['isCommercial'].sum(),
                'fraud_rate': df['isCommercial'].mean(),
                'features_analyzed': len(df.select_dtypes(include=[np.number]).columns)
            }
        }

        with open('rules_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n✅ Результаты сохранены в rules_analysis_results.json")
        print(f"📊 Всего проанализировано: {len(df)} объектов")
        print(f"🎯 Найдено оптимальных правил: {len(rules)}")
        print(f"🕵️ Обнаружено скрытых паттернов: {len(hidden_patterns)}")
        print(
            f"🧮 Проанализировано признаков: {len(df.select_dtypes(include=[np.number]).columns)}")

    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
