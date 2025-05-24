"""
🤝 Ансамбль предиктор с консенсусом для защиты от ложных срабатываний
"""

import json
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Union, Optional
from config import (CLASSIFICATION_THRESHOLDS, ENSEMBLE_SETTINGS,
                    PROTECTED_CATEGORIES, CAUTION_LEVELS, RISK_LEVELS)


class EnsemblePredictor:
    """Ансамбль предиктор с консенсусом и настраиваемыми порогами для защиты честных жителей"""

    def __init__(self, ensemble_path: str, caution_level: str = 'conservative'):
        """
        Инициализация ансамбля моделей

        Args:
            ensemble_path: Путь к файлу с ансамблем моделей
            caution_level: Уровень осторожности ('aggressive', 'balanced', 'conservative', 'ultra_safe')
        """
        self.ensemble_data = joblib.load(ensemble_path)
        self.models = self.ensemble_data['models']
        self.feature_engineer = self.ensemble_data['feature_engineer']
        self.feature_names = self.ensemble_data.get('feature_names', [])
        self.best_model = self.ensemble_data.get('best_model', 'Unknown')

        # Настройки осторожности
        self.caution_level = caution_level
        self.caution_settings = CAUTION_LEVELS.get(
            caution_level, CAUTION_LEVELS['conservative'])
        self.base_threshold = self.caution_settings['threshold']

        print(f"🤝 Загружен ансамбль из {len(self.models)} моделей")
        print(f"🛡️ Уровень осторожности: {caution_level}")
        print(f"📊 Базовый порог: {self.base_threshold:.2f}")
        print(
            f"🎯 Ожидаемая точность: {self.caution_settings['expected_precision']:.1%}")

    def predict(self, data: Union[List[Dict], Dict],
                custom_threshold: Optional[float] = None,
                require_consensus: bool = None,
                detailed: bool = True) -> List[Dict]:
        """
        Предсказание с консенсусом моделей и защитой честных жителей
        detailed: если False — только исходные поля + isCommercial
        """
        # Преобразуем в список если передан один объект
        if isinstance(data, dict):
            data = [data]

        # Извлечение признаков
        df = self.feature_engineer.extract_features(data)
        X = df[self.feature_names]

        # Настройки консенсуса
        if require_consensus is None:
            require_consensus = ENSEMBLE_SETTINGS['require_consensus']

        results = []
        for i, record in enumerate(data):
            # Получаем предсказания от всех моделей
            model_predictions = self._get_all_model_predictions(X.iloc[i:i+1])

            # Применяем консенсус
            final_prediction = self._apply_consensus(
                model_predictions,
                custom_threshold or self.base_threshold,
                require_consensus
            )

            # Проверяем защищенные категории
            adjusted_prediction = self._check_protected_categories(
                final_prediction, record, df.iloc[i]
            )

            # Формируем результат
            result = self._format_result(
                record, i, adjusted_prediction, model_predictions, detailed=detailed
            )

            results.append(result)

        return results

    def _get_all_model_predictions(self, X_sample: pd.DataFrame) -> Dict:
        """Получение предсказаний от всех моделей"""
        predictions = {
            'probabilities': {},
            'binary_predictions': {},
            'individual_decisions': {}
        }

        for model_name, model_data in self.models.items():
            try:
                model = model_data['model']

                # Вероятности
                prob = model.predict_proba(X_sample)[0, 1]
                predictions['probabilities'][model_name] = float(prob)

                # Индивидуальные пороги для каждой модели
                individual_threshold = ENSEMBLE_SETTINGS['individual_thresholds'].get(
                    model_name, self.base_threshold
                )

                # Бинарное предсказание
                binary_pred = prob >= individual_threshold
                predictions['binary_predictions'][model_name] = binary_pred

                # Решение с учетом порога
                predictions['individual_decisions'][model_name] = {
                    'probability': prob,
                    'threshold_used': individual_threshold,
                    'decision': binary_pred,
                    'confidence': abs(prob - 0.5) * 2  # Уверенность от 0 до 1
                }

            except Exception as e:
                print(f"⚠️ Ошибка при предсказании модели {model_name}: {e}")
                predictions['probabilities'][model_name] = 0.0
                predictions['binary_predictions'][model_name] = False
                predictions['individual_decisions'][model_name] = {
                    'probability': 0.0,
                    'threshold_used': self.base_threshold,
                    'decision': False,
                    'confidence': 0.0
                }

        return predictions

    def _apply_consensus(self, model_predictions: Dict, threshold: float,
                         require_consensus: bool) -> Dict:
        """Применение консенсуса между моделями"""
        probabilities = list(model_predictions['probabilities'].values())
        binary_decisions = list(
            model_predictions['binary_predictions'].values())

        if not probabilities:
            return {
                'final_probability': 0.0,
                'final_decision': False,
                'consensus_reached': False,
                'agreement_level': 0.0,
                'decision_method': 'error'
            }

        # Усредненная вероятность
        avg_probability = np.mean(probabilities)

        # Уровень согласия
        positive_votes = sum(binary_decisions)
        total_models = len(binary_decisions)
        agreement_level = positive_votes / total_models if total_models > 0 else 0

        # Проверка консенсуса
        consensus_threshold = ENSEMBLE_SETTINGS['consensus_threshold']
        min_models_agree = ENSEMBLE_SETTINGS['min_models_agree']

        consensus_reached = (
            agreement_level >= consensus_threshold and
            positive_votes >= min_models_agree
        )

        # Финальное решение
        if require_consensus:
            # Строгий консенсус - требуем согласия
            if consensus_reached:
                final_decision = avg_probability >= threshold
                decision_method = 'strict_consensus'
            else:
                # Нет консенсуса - по умолчанию считаем честным (защита от ложных обвинений!)
                final_decision = False
                decision_method = 'no_consensus_safe'
        else:
            # Мягкий подход - используем усредненную вероятность
            final_decision = avg_probability >= threshold
            decision_method = 'averaged_probability'

        return {
            'final_probability': avg_probability,
            'final_decision': final_decision,
            'consensus_reached': consensus_reached,
            'agreement_level': agreement_level,
            'positive_votes': positive_votes,
            'total_models': total_models,
            'decision_method': decision_method,
            'individual_probabilities': model_predictions['probabilities']
        }

    def _check_protected_categories(self, prediction: Dict, record: Dict,
                                    features: pd.Series) -> Dict:
        """Проверка защищенных категорий граждан"""
        if not PROTECTED_CATEGORIES['enable_protection']:
            return prediction

        # Создаем копию предсказания
        adjusted = prediction.copy()
        protection_applied = []

        # Проверяем различные категории защиты
        categories = PROTECTED_CATEGORIES['categories']

        # 1. Многодетная семья
        residents = record.get('residentsCount', 1)
        if residents >= categories['large_family_threshold']:
            protection_applied.append('large_family')

        # 2. Маленькая квартира (возможно льготники)
        area = record.get('totalArea', 100)
        if area and area <= categories['small_apartment_threshold']:
            protection_applied.append('small_apartment')

        # 3. Низкое потребление (возможно льготники)
        avg_consumption = features.get('avg_consumption', 0)
        if avg_consumption <= categories['low_income_consumption']:
            protection_applied.append('low_consumption')

        # 4. Пенсионеры (если есть данные о возрасте - пока пропускаем)

        # Применяем защиту если есть основания
        if protection_applied:
            # Увеличиваем порог для защищенных категорий
            multiplier = PROTECTED_CATEGORIES['protection_multiplier']
            additional = PROTECTED_CATEGORIES['additional_threshold']

            original_threshold = self.base_threshold
            protected_threshold = (
                original_threshold * multiplier) + additional

            # Пересчитываем решение с повышенным порогом
            if adjusted['final_probability'] < protected_threshold:
                adjusted['final_decision'] = False
                adjusted['protection_applied'] = True
                adjusted['protection_reasons'] = protection_applied
                adjusted['original_threshold'] = original_threshold
                adjusted['protected_threshold'] = protected_threshold
                adjusted['decision_method'] = f"{adjusted['decision_method']}_protected"
            else:
                adjusted['protection_applied'] = False
        else:
            adjusted['protection_applied'] = False

        return adjusted

    def _format_result(self, record: Dict, index: int, prediction: Dict,
                       model_predictions: Dict, detailed: bool = True) -> Dict:
        """Форматирование результата"""
        fraud_prob = prediction['final_probability']
        is_fraud = prediction['final_decision']

        if not detailed:
            # Короткий формат: все исходные поля + isCommercial после accountId
            result = {}
            for k, v in record.items():
                result[k] = v
                if k == 'accountId':
                    result['isCommercial'] = is_fraud
            if 'accountId' not in record:
                result = {'isCommercial': is_fraud, **record}
            return result

        # Подробный формат (старое поведение)
        result = {
            'accountId': record.get('accountId', f'UNKNOWN_{index}'),
            'isCommercial': is_fraud,
            'fraud_probability': fraud_prob,
            'fraud_probability_percent': f"{fraud_prob * 100:.1f}%",
            'risk_level': self._get_risk_level(fraud_prob),
            'interpretation': self._get_interpretation(is_fraud, fraud_prob, prediction),
            'consensus_details': {
                'agreement_level': f"{prediction['agreement_level']:.1%}",
                'positive_votes': f"{prediction['positive_votes']}/{prediction['total_models']}",
                'consensus_reached': prediction['consensus_reached'],
                'decision_method': prediction['decision_method']
            },
            'individual_models': model_predictions['individual_decisions'],
            'protection_applied': prediction.get('protection_applied', False),
            'protection_reasons': prediction.get('protection_reasons', []),
            'caution_level': self.caution_level,
            'threshold_used': prediction.get('protected_threshold', self.base_threshold)
        }
        return result

    def _get_risk_level(self, probability: float) -> str:
        """Определение уровня риска"""
        for level, info in RISK_LEVELS.items():
            if probability >= info['threshold']:
                return level
        return 'MINIMAL'

    def _get_interpretation(self, is_fraud: bool, probability: float,
                            prediction_details: Dict) -> str:
        """Расширенная интерпретация результата"""
        consensus = prediction_details.get('consensus_reached', False)
        method = prediction_details.get('decision_method', 'unknown')
        protected = prediction_details.get('protection_applied', False)

        if is_fraud:
            if consensus and probability > 0.9:
                base = "🚨 КОНСЕНСУС МОДЕЛЕЙ: Очень высокая вероятность нарушения"
            elif consensus and probability > 0.8:
                base = "⚠️ КОНСЕНСУС МОДЕЛЕЙ: Высокая вероятность нарушения"
            elif probability > 0.8:
                base = "📍 Вероятное нарушение (без полного консенсуса)"
            else:
                base = "❓ Возможное нарушение - требует проверки"

            if protected:
                base += " | 🛡️ ПРОВЕРЕН защитными фильтрами"
        else:
            if method == 'no_consensus_safe':
                base = "✅ НЕТ КОНСЕНСУСА - по умолчанию считается честным (защитная мера)"
            elif protected:
                base = "✅ Защищенная категория - дополнительная осторожность применена"
            elif probability < 0.2:
                base = "✅ Добросовестный потребитель (все модели согласны)"
            elif probability < 0.4:
                base = "✅ Скорее всего добросовестный потребитель"
            else:
                base = "❓ Пограничный случай - но порог не достигнут"

        return base

    def get_model_explanations(self, results: List[Dict]) -> Dict:
        """Получение объяснений решений моделей"""
        explanations = {
            'caution_settings': self.caution_settings,
            'ensemble_settings': ENSEMBLE_SETTINGS,
            'protection_settings': PROTECTED_CATEGORIES,
            'models_performance': {}
        }

        # Добавляем информацию о производительности моделей
        for model_name, model_data in self.models.items():
            explanations['models_performance'][model_name] = {
                'cv_score': model_data.get('cv_score', 0),
                'individual_threshold': ENSEMBLE_SETTINGS['individual_thresholds'].get(
                    model_name, self.base_threshold
                )
            }

        return explanations

    def update_caution_level(self, new_level: str):
        """Обновление уровня осторожности"""
        if new_level in CAUTION_LEVELS:
            self.caution_level = new_level
            self.caution_settings = CAUTION_LEVELS[new_level]
            self.base_threshold = self.caution_settings['threshold']
            print(f"🛡️ Уровень осторожности обновлен: {new_level}")
            print(f"📊 Новый порог: {self.base_threshold:.2f}")
        else:
            print(f"❌ Неизвестный уровень: {new_level}")
            print(f"✅ Доступные: {list(CAUTION_LEVELS.keys())}")


def predict_with_ensemble(data_path: str,
                          ensemble_path: str = 'fraud_detection_model_ensemble.joblib',
                          caution_level: str = 'conservative') -> str:
    """Быстрая функция для предсказания с ансамблем"""
    predictor = EnsemblePredictor(ensemble_path, caution_level)

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = predictor.predict(data)

    # Статистика
    total = len(results)
    fraudsters = sum(1 for r in results if r['isCommercial'])
    protected = sum(1 for r in results if r.get('protection_applied', False))
    consensus_cases = sum(
        1 for r in results if r['consensus_details']['consensus_reached'])

    print(f"\n🤝 РЕЗУЛЬТАТЫ АНСАМБЛЯ (уровень: {caution_level.upper()})")
    print("=" * 60)
    print(f"Проанализировано: {total}")
    print(f"Выявлено нарушителей: {fraudsters} ({fraudsters/total*100:.1f}%)")
    print(
        f"Консенсус достигнут: {consensus_cases} ({consensus_cases/total*100:.1f}%)")
    print(f"Защита применена: {protected} ({protected/total*100:.1f}%)")

    return json.dumps(results, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Пример использования
    results = predict_with_ensemble(
        'test_data.json', caution_level='conservative')
