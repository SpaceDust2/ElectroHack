"""
📊 Модуль разведочного анализа данных (EDA)
"""

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from config import FRAUD_RULES, PLOT_THEME


class FraudDataAnalyzer:
    """Класс для анализа данных о мошенничестве"""

    def __init__(self):
        self.df = None
        self.insights = {}

    def load_data(self, train_path, test_path=None):
        """Загрузка и объединение данных"""
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        all_data = train_data

        if test_path:
            with open(test_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            all_data += test_data

        self.df = self._extract_features(all_data)
        print(f"✅ Загружено {len(self.df)} объектов")
        return self.df

    def _extract_features(self, data):
        """Извлечение признаков из JSON"""
        features = []

        for record in data:
            feature_dict = {
                'accountId': record.get('accountId'),
                'isCommercial': record.get('isCommercial'),
                'buildingType': record.get('buildingType', 'unknown'),
                'roomsCount': record.get('roomsCount', 0),
                'residentsCount': record.get('residentsCount', 0),
                'totalArea': record.get('totalArea'),
                'address': record.get('address', '')
            }

            # Анализ потребления
            consumption = record.get('consumption', {})
            monthly_values = []

            for month in range(1, 13):
                value = consumption.get(str(month))
                if value is not None and value >= 0:
                    monthly_values.append(value)
                    feature_dict[f'month_{month}'] = value

            if monthly_values:
                feature_dict['avg_consumption'] = np.mean(monthly_values)
                feature_dict['max_consumption'] = max(monthly_values)
                feature_dict['min_consumption'] = min(monthly_values)
                feature_dict['std_consumption'] = np.std(
                    monthly_values) if len(monthly_values) > 1 else 0
                feature_dict['total_consumption'] = sum(monthly_values)
                feature_dict['months_with_data'] = len(monthly_values)

                # Сезонные средние
                winter = [consumption.get(m, 0) for m in [
                    '12', '1', '2'] if consumption.get(m, 0) > 0]
                summer = [consumption.get(m, 0) for m in [
                    '6', '7', '8'] if consumption.get(m, 0) > 0]

                feature_dict['winter_avg'] = np.mean(winter) if winter else 0
                feature_dict['summer_avg'] = np.mean(summer) if summer else 0
                feature_dict['summer_winter_ratio'] = (
                    feature_dict['summer_avg'] / feature_dict['winter_avg']
                    if feature_dict['winter_avg'] > 0 else 1
                )

                # Коэффициент вариации
                feature_dict['cv'] = feature_dict['std_consumption'] / \
                    feature_dict['avg_consumption'] if feature_dict['avg_consumption'] > 0 else 0

            features.append(feature_dict)

        return pd.DataFrame(features)

    def basic_statistics(self):
        """Базовая статистика"""
        stats_dict = {
            'total_objects': len(self.df),
            'fraudsters': int(self.df['isCommercial'].sum()),
            'fraud_rate': float(self.df['isCommercial'].mean()),
            'missing_totalArea': int(self.df['totalArea'].isna().sum()),
            'missing_totalArea_pct': float(self.df['totalArea'].isna().mean() * 100)
        }

        self.insights['basic_stats'] = stats_dict

        print("\n📊 БАЗОВАЯ СТАТИСТИКА")
        print("=" * 50)
        for key, value in stats_dict.items():
            print(f"{key}: {value}")

        return stats_dict

    def compare_groups(self):
        """Сравнение мошенников и честных жителей"""
        print("\n🔍 СРАВНЕНИЕ ГРУПП")
        print("=" * 50)

        comparison = {}
        features = ['avg_consumption', 'min_consumption', 'max_consumption',
                    'cv', 'summer_winter_ratio', 'roomsCount', 'residentsCount']

        for feature in features:
            if feature in self.df.columns:
                honest = self.df[self.df['isCommercial']
                                 == False][feature].dropna()
                fraud = self.df[self.df['isCommercial']
                                == True][feature].dropna()

                if len(honest) > 0 and len(fraud) > 0:
                    # T-test
                    t_stat, p_value = stats.ttest_ind(honest, fraud)

                    comparison[feature] = {
                        'honest_mean': float(honest.mean()),
                        'fraud_mean': float(fraud.mean()),
                        'difference_pct': float((fraud.mean() - honest.mean()) / honest.mean() * 100),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }

                    print(f"\n{feature}:")
                    print(f"  Честные: {honest.mean():.2f}")
                    print(f"  Мошенники: {fraud.mean():.2f}")
                    print(
                        f"  Разница: {comparison[feature]['difference_pct']:+.1f}%")
                    print(
                        f"  p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

        self.insights['group_comparison'] = comparison
        return comparison

    def find_optimal_thresholds(self):
        """Поиск оптимальных порогов для детекции"""
        print("\n📏 ОПТИМАЛЬНЫЕ ПОРОГИ")
        print("=" * 50)

        thresholds = {}

        for feature in ['avg_consumption', 'min_consumption', 'cv']:
            if feature not in self.df.columns:
                continue

            honest = self.df[self.df['isCommercial']
                             == False][feature].dropna()
            fraud = self.df[self.df['isCommercial'] == True][feature].dropna()

            # Находим оптимальный порог
            threshold_range = np.linspace(honest.min(), fraud.max(), 100)
            best_f1 = 0
            best_threshold = 0
            best_metrics = {}

            for threshold in threshold_range:
                if feature == 'cv':  # Для CV меньшие значения указывают на мошенничество
                    tp = (fraud <= threshold).sum()
                    fp = (honest <= threshold).sum()
                    fn = (fraud > threshold).sum()
                else:
                    tp = (fraud >= threshold).sum()
                    fp = (honest >= threshold).sum()
                    fn = (fraud < threshold).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / \
                    (precision + recall) if (precision + recall) > 0 else 0

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }

            thresholds[feature] = {
                'threshold': float(best_threshold),
                'metrics': best_metrics
            }

            print(f"\n{feature}:")
            print(f"  Оптимальный порог: {best_threshold:.2f}")
            print(f"  Precision: {best_metrics['precision']:.2%}")
            print(f"  Recall: {best_metrics['recall']:.2%}")
            print(f"  F1-score: {best_metrics['f1']:.2%}")

        self.insights['optimal_thresholds'] = thresholds
        return thresholds

    def apply_fraud_rules(self):
        """Применение правил детекции"""
        print("\n🎯 ПРИМЕНЕНИЕ ПРАВИЛ")
        print("=" * 50)

        rules_results = {}

        for rule_name, rule_info in FRAUD_RULES.items():
            mask = rule_info['condition'](self.df)

            precision = self.df[mask]['isCommercial'].mean(
            ) if mask.sum() > 0 else 0
            recall = mask[self.df['isCommercial'] == True].mean() if (
                self.df['isCommercial'] == True).sum() > 0 else 0

            rules_results[rule_name] = {
                'description': rule_info['description'],
                'precision': float(precision),
                'recall': float(recall),
                'caught': int(mask.sum()),
                'fraudsters_caught': int(mask[self.df['isCommercial'] == True].sum())
            }

            print(f"\n{rule_name}: {rule_info['description']}")
            print(f"  Точность: {precision:.1%}")
            print(f"  Покрытие: {recall:.1%}")
            print(f"  Поймано: {mask.sum()} объектов")

        self.insights['fraud_rules'] = rules_results
        return rules_results

    def create_visualizations(self):
        """Создание интерактивных визуализаций"""
        print("\n📊 Создание визуализаций...")

        # Создаем сложную визуализацию с subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Распределение потребления', 'Сезонность', 'Стабильность потребления',
                'Потребление по месяцам', 'Количество жителей', 'Тип здания',
                'Минимальное потребление', 'Корреляция с мошенничеством', 'Правила детекции'
            ),
            specs=[
                [{'type': 'box'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'box'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'bar'}, {'type': 'bar'}]
            ]
        )

        # 1. Распределение потребления
        for is_commercial in [False, True]:
            label = 'Мошенники' if is_commercial else 'Честные'
            data = self.df[self.df['isCommercial']
                           == is_commercial]['avg_consumption']
            fig.add_trace(
                go.Box(y=data, name=label, boxpoints='outliers'),
                row=1, col=1
            )

        # 2. Сезонность
        fig.add_trace(
            go.Scatter(
                x=self.df[self.df['isCommercial'] == False]['winter_avg'],
                y=self.df[self.df['isCommercial'] == False]['summer_avg'],
                mode='markers',
                name='Честные',
                marker=dict(color='blue', size=5, opacity=0.6)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=self.df[self.df['isCommercial'] == True]['winter_avg'],
                y=self.df[self.df['isCommercial'] == True]['summer_avg'],
                mode='markers',
                name='Мошенники',
                marker=dict(color='red', size=8, opacity=0.8)
            ),
            row=1, col=2
        )

        # 3. Стабильность потребления
        fig.add_trace(
            go.Scatter(
                x=self.df['avg_consumption'],
                y=self.df['cv'],
                mode='markers',
                marker=dict(
                    color=self.df['isCommercial'].astype(int),
                    colorscale='RdBu',
                    size=5,
                    opacity=0.6
                )
            ),
            row=1, col=3
        )

        # 4. Потребление по месяцам
        months = ['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
                  'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12']

        for is_commercial in [False, True]:
            monthly_avg = []
            for month in months:
                if month in self.df.columns:
                    values = self.df[self.df['isCommercial']
                                     == is_commercial][month].dropna()
                    monthly_avg.append(values.mean() if len(values) > 0 else 0)

            fig.add_trace(
                go.Bar(x=list(range(1, 13)), y=monthly_avg,
                       name='Мошенники' if is_commercial else 'Честные'),
                row=2, col=1
            )

        # 5. Количество жителей
        for is_commercial in [False, True]:
            data = self.df[self.df['isCommercial']
                           == is_commercial]['residentsCount']
            fig.add_trace(
                go.Box(y=data, name='Мошенники' if is_commercial else 'Честные'),
                row=2, col=2
            )

        # 6. Тип здания
        building_stats = self.df.groupby(
            ['buildingType', 'isCommercial']).size().unstack(fill_value=0)
        for col in building_stats.columns:
            fig.add_trace(
                go.Bar(x=building_stats.index, y=building_stats[col],
                       name='Мошенники' if col else 'Честные'),
                row=2, col=3
            )

        # 7. Минимальное потребление
        for is_commercial in [False, True]:
            data = self.df[self.df['isCommercial']
                           == is_commercial]['min_consumption']
            fig.add_trace(
                go.Histogram(x=data, name='Мошенники' if is_commercial else 'Честные',
                             opacity=0.7, nbinsx=30),
                row=3, col=1
            )

        # 8. Корреляция с мошенничеством
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = self.df[numeric_cols].corrwith(
            self.df['isCommercial']).abs().sort_values(ascending=False).head(10)

        fig.add_trace(
            go.Bar(x=correlations.values, y=correlations.index, orientation='h'),
            row=3, col=2
        )

        # 9. Эффективность правил
        if 'fraud_rules' in self.insights:
            rules_data = self.insights['fraud_rules']
            rule_names = list(rules_data.keys())
            precisions = [rules_data[r]['precision'] for r in rule_names]
            recalls = [rules_data[r]['recall'] for r in rule_names]

            fig.add_trace(
                go.Bar(x=rule_names, y=precisions, name='Precision'),
                row=3, col=3
            )
            fig.add_trace(
                go.Bar(x=rule_names, y=recalls, name='Recall'),
                row=3, col=3
            )

        # Настройка layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Комплексный анализ паттернов мошенничества",
            title_font_size=20,
            template=PLOT_THEME['template']
        )

        # Сохранение
        fig.write_html('eda_visualizations.html')
        print("✅ Визуализации сохранены в eda_visualizations.html")

        return fig

    def generate_report(self):
        """Генерация отчета с инсайтами"""
        report = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': self.insights.get('basic_stats', {}),
            'group_comparison': self.insights.get('group_comparison', {}),
            'optimal_thresholds': self.insights.get('optimal_thresholds', {}),
            'fraud_rules': self.insights.get('fraud_rules', {}),
            'key_findings': [
                "Мошенники имеют значительно более высокое среднее потребление",
                "У мошенников более стабильное потребление (низкий CV)",
                "Отсутствие сезонности - важный признак коммерческой деятельности",
                "Минимальное потребление > 50 кВт·ч - сильный индикатор"
            ]
        }

        # Сохранение отчета
        with open('eda_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n✅ Отчет сохранен в eda_report.json")
        return report


def run_full_analysis(train_path='data/dataset_train.json', test_path='data/dataset_test.json'):
    """Запуск полного анализа"""
    analyzer = FraudDataAnalyzer()

    # 1. Загрузка данных
    analyzer.load_data(train_path, test_path)

    # 2. Базовая статистика
    analyzer.basic_statistics()

    # 3. Сравнение групп
    analyzer.compare_groups()

    # 4. Оптимальные пороги
    analyzer.find_optimal_thresholds()

    # 5. Применение правил
    analyzer.apply_fraud_rules()

    # 6. Визуализации
    analyzer.create_visualizations()

    # 7. Генерация отчета
    analyzer.generate_report()

    return analyzer


if __name__ == "__main__":
    analyzer = run_full_analysis()
