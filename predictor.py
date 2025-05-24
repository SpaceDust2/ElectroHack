"""
🔮 Модуль предсказаний и интерпретации результатов
"""

import json
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Union
from config import RISK_LEVELS


class FraudPredictor:
    """Класс для предсказаний и интерпретации результатов"""

    def __init__(self, model_path: str):
        """Инициализация с загрузкой модели"""
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.feature_names = self.model_data['feature_names']
        self.feature_engineer = self.model_data['feature_engineer']
        self.model_name = self.model_data.get('model_name', 'Unknown')

    def predict_from_file(self, data_path: str) -> List[Dict]:
        """Предсказание для данных из файла"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return self.predict(data)

    def predict(self, data: Union[List[Dict], Dict]) -> List[Dict]:
        """Предсказание для одного или нескольких объектов"""
        # Преобразуем в список если передан один объект
        if isinstance(data, dict):
            data = [data]

        # Извлечение признаков
        df = self.feature_engineer.extract_features(data)

        # Предсказание
        X = df[self.feature_names]
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)

        # Формирование результатов
        results = []
        for i, record in enumerate(data):
            fraud_prob = float(probabilities[i])
            is_fraud = bool(predictions[i])

            result = {
                'accountId': record.get('accountId', f'UNKNOWN_{i}'),
                'isCommercial': is_fraud,
                'fraud_probability': fraud_prob,
                'fraud_probability_percent': f"{fraud_prob * 100:.1f}%",
                'risk_level': self._get_risk_level(fraud_prob),
                'interpretation': self._get_interpretation(is_fraud, fraud_prob),
                'patterns': self._analyze_patterns(df.iloc[i])
            }

            results.append(result)

        return results

    def _get_risk_level(self, probability: float) -> str:
        """Определение уровня риска"""
        for level, info in RISK_LEVELS.items():
            if probability >= info['threshold']:
                return level
        return 'MINIMAL'

    def _get_interpretation(self, is_fraud: bool, probability: float) -> str:
        """Текстовая интерпретация результата"""
        if is_fraud:
            if probability > 0.9:
                return "🚨 МОШЕННИК с очень высокой вероятностью!"
            elif probability > 0.8:
                return "⚠️ МОШЕННИК с высокой вероятностью"
            else:
                return "📍 Вероятный МОШЕННИК"
        else:
            if probability < 0.2:
                return "✅ Честный житель с высокой вероятностью"
            elif probability < 0.4:
                return "✅ Скорее всего честный житель"
            else:
                return "❓ Требует дополнительной проверки"

    def _analyze_patterns(self, features: pd.Series) -> Dict:
        """Анализ паттернов в данных объекта"""
        patterns = {
            'consumption_level': self._get_consumption_level(features),
            'seasonality': self._get_seasonality_status(features),
            'stability': self._get_stability_status(features),
            'anomalies': self._get_anomalies(features)
        }

        return patterns

    def _get_consumption_level(self, features: pd.Series) -> str:
        """Определение уровня потребления"""
        avg_consumption = features.get('avg_consumption', 0)

        if avg_consumption > 500:
            return "Очень высокое"
        elif avg_consumption > 300:
            return "Высокое"
        elif avg_consumption > 150:
            return "Среднее"
        else:
            return "Низкое"

    def _get_seasonality_status(self, features: pd.Series) -> str:
        """Определение наличия сезонности"""
        ratio = features.get('summer_winter_ratio', 1)

        if 0.8 <= ratio <= 1.2:
            return "Отсутствует (подозрительно)"
        elif ratio < 0.6:
            return "Выраженная (типично для жилья)"
        else:
            return "Умеренная"

    def _get_stability_status(self, features: pd.Series) -> str:
        """Определение стабильности потребления"""
        cv = features.get('cv', 0)

        if cv < 0.2:
            return "Очень стабильное (подозрительно)"
        elif cv < 0.3:
            return "Стабильное"
        elif cv < 0.5:
            return "Умеренно стабильное"
        else:
            return "Нестабильное (типично для жилья)"

    def _get_anomalies(self, features: pd.Series) -> List[str]:
        """Выявление аномалий"""
        anomalies = []

        if features.get('high_min_consumption', 0):
            anomalies.append("Высокое минимальное потребление")

        if features.get('stable_high_consumption', 0):
            anomalies.append("Стабильно высокое потребление")

        if features.get('no_seasonality', 0) and features.get('avg_consumption', 0) > 300:
            anomalies.append("Нет сезонности при высоком потреблении")

        if features.get('zero_consumption_months', 0) == 0 and features.get('months_with_data', 0) >= 10:
            anomalies.append("Нет месяцев с нулевым потреблением")

        if features.get('consumption_per_resident', 0) > 250:
            anomalies.append("Очень высокое потребление на жителя")

        return anomalies

    def generate_report(self, results: List[Dict]) -> Dict:
        """Генерация сводного отчета по результатам"""
        total = len(results)
        fraudsters = sum(1 for r in results if r['isCommercial'])

        # Группировка по уровням риска
        risk_distribution = {}
        for level in RISK_LEVELS.keys():
            count = sum(1 for r in results if r['risk_level'] == level)
            risk_distribution[level] = {
                'count': count,
                'percentage': count / total * 100 if total > 0 else 0
            }

        # Статистика по паттернам
        pattern_stats = {
            'no_seasonality': sum(1 for r in results if 'Отсутствует' in r['patterns']['seasonality']),
            'very_stable': sum(1 for r in results if 'Очень стабильное' in r['patterns']['stability']),
            'high_consumption': sum(1 for r in results if 'высокое' in r['patterns']['consumption_level'].lower())
        }

        report = {
            'summary': {
                'total_analyzed': total,
                'fraudsters_detected': fraudsters,
                'fraud_rate': fraudsters / total * 100 if total > 0 else 0,
                'model_used': self.model_name
            },
            'risk_distribution': risk_distribution,
            'pattern_statistics': pattern_stats,
            'top_suspicious': sorted(results, key=lambda x: x['fraud_probability'], reverse=True)[:10]
        }

        return report

    def export_results(self, results: List[Dict], format: str = 'json', filename: str = 'predictions') -> str:
        """Экспорт результатов в различных форматах"""
        if format == 'json':
            filepath = f'{filename}.json'
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        elif format == 'csv':
            filepath = f'{filename}.csv'
            # Упрощаем структуру для CSV
            simplified = []
            for r in results:
                row = {
                    'accountId': r['accountId'],
                    'isCommercial': r['isCommercial'],
                    'fraud_probability': r['fraud_probability'],
                    'risk_level': r['risk_level'],
                    'interpretation': r['interpretation']
                }
                simplified.append(row)

            df = pd.DataFrame(simplified)
            df.to_csv(filepath, index=False, encoding='utf-8')

        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")

        print(f"✅ Результаты сохранены в {filepath}")
        return filepath


def predict_fraud(data_path: str, model_path: str = 'fraud_detection_model.joblib') -> str:
    """Быстрая функция для предсказания"""
    predictor = FraudPredictor(model_path)
    results = predictor.predict_from_file(data_path)

    # Генерация отчета
    report = predictor.generate_report(results)

    print(f"\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("=" * 50)
    print(f"Проанализировано: {report['summary']['total_analyzed']}")
    print(
        f"Выявлено мошенников: {report['summary']['fraudsters_detected']} ({report['summary']['fraud_rate']:.1f}%)")
    print(f"\nРаспределение по рискам:")
    for level, info in report['risk_distribution'].items():
        print(f"  {level}: {info['count']} ({info['percentage']:.1f}%)")

    # Экспорт результатов
    predictor.export_results(results, format='json')

    return json.dumps(results, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Пример использования
    results = predict_fraud('test_data.json')
