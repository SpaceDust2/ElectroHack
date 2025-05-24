"""
🤖 Модуль обучения ML моделей с детальными метриками
"""

from datetime import datetime
from feature_engineering import FeatureEngineer
from config import MODEL_PARAMS
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import (roc_auc_score, accuracy_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_curve,
                             precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib
from typing import Dict, List, Tuple, Any
import warnings
import os
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Класс для обучения и оценки моделей с детальными метриками"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.models = self._init_models()
        self.feature_engineer = FeatureEngineer()
        self.best_model = None
        self.best_score = 0
        self.results = {}
        self.detailed_metrics = {}

    def _init_models(self) -> Dict[str, Any]:
        """Инициализация моделей с параметрами из конфига"""
        models = {}

        # CatBoost
        catboost_params = MODEL_PARAMS['catboost'].copy()
        if not self.use_gpu:
            catboost_params['task_type'] = 'CPU'
        models['CatBoost'] = CatBoostClassifier(**catboost_params)

        # XGBoost
        xgboost_params = MODEL_PARAMS['xgboost'].copy()
        if not self.use_gpu:
            xgboost_params['tree_method'] = 'hist'
        models['XGBoost'] = XGBClassifier(**xgboost_params)

        # Random Forest
        models['RandomForest'] = RandomForestClassifier(
            **MODEL_PARAMS['random_forest'])

        return models

    def load_data(self, train_path: str, test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Загрузка и подготовка данных"""
        print("📁 Загрузка данных...")

        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        # Извлечение признаков
        print("🔧 Feature engineering...")
        train_df = self.feature_engineer.extract_features(train_data)

        test_df = None
        if test_path:
            with open(test_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            test_df = self.feature_engineer.extract_features(test_data)

        print(f"✅ Загружено: {len(train_df)} train")
        if test_df is not None:
            print(f"✅ Загружено: {len(test_df)} test")
        print(f"✅ Признаков: {len(self.feature_engineer.feature_names)}")

        # Анализ дисбаланса классов
        fraud_rate = train_df['isCommercial'].mean()
        print(f"📊 Доля мошенников в обучающей выборке: {fraud_rate:.2%}")
        print(
            f"📊 Баланс классов - Честные: {(1-fraud_rate)*100:.1f}%, Мошенники: {fraud_rate*100:.1f}%")

        return train_df, test_df

    def train_models(self, train_df: pd.DataFrame, cv_folds: int = 5) -> Dict[str, Dict]:
        """Обучение всех моделей с детальными метриками"""
        print("\n🚀 ОБУЧЕНИЕ МОДЕЛЕЙ С ДЕТАЛЬНЫМ АНАЛИЗОМ")
        print("=" * 60)

        # Проверка на пустые данные
        if len(train_df) == 0:
            raise ValueError("❌ Обучающие данные пусты!")

        if 'isCommercial' not in train_df.columns:
            raise ValueError("❌ Отсутствует столбец 'isCommercial' в данных!")

        # Подготовка данных
        X = train_df[self.feature_engineer.feature_names].fillna(0)
        y = train_df['isCommercial'].astype(int)

        # Проверка на достаточность данных
        if len(y.unique()) < 2:
            raise ValueError(
                "❌ Недостаточно классов для обучения (нужны и мошенники, и честные)!")

        # Стратифицированная кросс-валидация
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for model_name, model in self.models.items():
            print(f"\n🤖 Обучение {model_name}")
            print("-" * 40)

            # Кросс-валидация с детальными метриками
            cv_scores = []
            cv_predictions = []
            cv_probabilities = []
            cv_true_labels = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                # Обучение на фолде
                model_copy = self._clone_model(model)
                model_copy.fit(X_train_fold, y_train_fold)

                # Предсказания на валидации
                y_pred_proba = model_copy.predict_proba(X_val_fold)[:, 1]
                y_pred = model_copy.predict(X_val_fold)

                # Метрики
                auc = roc_auc_score(y_val_fold, y_pred_proba)
                cv_scores.append(auc)

                # Сохраняем для общего анализа
                cv_predictions.extend(y_pred)
                cv_probabilities.extend(y_pred_proba)
                cv_true_labels.extend(y_val_fold)

                print(f"   Fold {fold+1}: AUC = {auc:.4f}")

            # Финальное обучение на всех данных
            model.fit(X, y)

            # Детальные метрики по кросс-валидации
            cv_predictions = np.array(cv_predictions)
            cv_probabilities = np.array(cv_probabilities)
            cv_true_labels = np.array(cv_true_labels)

            detailed_cv_metrics = self._calculate_detailed_metrics(
                cv_true_labels, cv_predictions, cv_probabilities)

            # Сохранение результатов
            cv_scores_array = np.array(cv_scores)
            self.results[model_name] = {
                'model': model,
                'cv_scores': cv_scores_array,
                'cv_mean': cv_scores_array.mean(),
                'cv_std': cv_scores_array.std(),
                'feature_names': self.feature_engineer.feature_names,
                'detailed_metrics': detailed_cv_metrics
            }

            # Анализ важности признаков
            feature_importance = self._get_model_feature_importance(
                model, model_name)
            self.results[model_name]['feature_importance'] = feature_importance

            # Вывод результатов
            print(f"\n📊 РЕЗУЛЬТАТЫ {model_name}:")
            print(
                f"   AUC (CV): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"   Точность: {detailed_cv_metrics['accuracy']:.4f}")
            print(f"   Precision: {detailed_cv_metrics['precision']:.4f}")
            print(f"   Recall: {detailed_cv_metrics['recall']:.4f}")
            print(f"   F1-Score: {detailed_cv_metrics['f1']:.4f}")

            # Confusion Matrix
            print(f"\n   Confusion Matrix:")
            cm = detailed_cv_metrics['confusion_matrix']
            print(f"   TN={cm[0, 0]:4d}  FP={cm[0, 1]:4d}")
            print(f"   FN={cm[1, 0]:4d}  TP={cm[1, 1]:4d}")

            # Топ-5 важных признаков
            if len(feature_importance) > 0:
                print(f"\n   Топ-5 важных признаков:")
                for i, (feature, importance) in enumerate(feature_importance.head(5).values):
                    print(f"   {i+1}. {feature}: {importance:.4f}")

            # Обновление лучшей модели
            if np.mean(cv_scores) > self.best_score:
                self.best_score = np.mean(cv_scores)
                self.best_model = model_name

        print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {self.best_model}")
        print(f"🎯 ЛУЧШИЙ AUC: {self.best_score:.4f}")

        return self.results

    def _clone_model(self, model):
        """Клонирование модели"""
        if isinstance(model, CatBoostClassifier):
            return CatBoostClassifier(**model.get_params())
        elif isinstance(model, XGBClassifier):
            return XGBClassifier(**model.get_params())
        else:
            return RandomForestClassifier(**model.get_params())

    def _calculate_detailed_metrics(self, y_true, y_pred, y_prob):
        """Расчет детальных метрик"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

    def _get_model_feature_importance(self, model, model_name):
        """Получение важности признаков для модели"""
        try:
            feature_names = self.feature_engineer.feature_names

            if not feature_names:
                print(f"⚠️ Нет списка признаков для {model_name}")
                return pd.DataFrame()

            importances = None

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()
            else:
                print(f"⚠️ {model_name} не поддерживает важность признаков")
                return pd.DataFrame()

            if importances is None or len(importances) == 0:
                print(f"⚠️ Пустая важность признаков для {model_name}")
                return pd.DataFrame()

            if len(importances) != len(feature_names):
                print(
                    f"⚠️ Несоответствие размеров: {len(importances)} важностей vs {len(feature_names)} признаков")
                return pd.DataFrame()

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            return importance_df

        except Exception as e:
            print(
                f"❌ Ошибка при получении важности признаков для {model_name}: {e}")
            return pd.DataFrame()

    def evaluate_on_test(self, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """Детальная оценка моделей на тестовом датасете"""
        if not self.results:
            raise ValueError("Сначала обучите модели!")

        print("\n📏 ДЕТАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОМ ДАТАСЕТЕ")
        print("=" * 60)

        X_test = test_df[self.feature_engineer.feature_names].fillna(0)
        y_test = test_df['isCommercial'].astype(int)

        test_results = {}

        for model_name, model_info in self.results.items():
            print(f"\n🔬 Тестирование {model_name}")
            print("-" * 40)

            model = model_info['model']

            # Предсказания
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # Детальные метрики
            detailed_metrics = self._calculate_detailed_metrics(
                y_test, y_pred, y_pred_proba)

            # Дополнительный анализ
            classification_rep = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0)

            test_results[model_name] = {
                **detailed_metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_rep
            }

            # Вывод результатов
            print(f"   AUC: {detailed_metrics['auc']:.4f}")
            print(f"   Точность: {detailed_metrics['accuracy']:.4f}")
            print(f"   Precision: {detailed_metrics['precision']:.4f}")
            print(f"   Recall: {detailed_metrics['recall']:.4f}")
            print(f"   F1-Score: {detailed_metrics['f1']:.4f}")

            # Confusion Matrix
            print(f"   Confusion Matrix:")
            cm = detailed_metrics['confusion_matrix']
            print(f"   TN={cm[0, 0]:4d}  FP={cm[0, 1]:4d}")
            print(f"   FN={cm[1, 0]:4d}  TP={cm[1, 1]:4d}")

            # Анализ ошибок
            if len(y_test) > 0:
                false_positives = np.sum((y_test == 0) & (y_pred == 1))
                false_negatives = np.sum((y_test == 1) & (y_pred == 0))
                print(
                    f"   Ложные срабатывания: {false_positives} ({false_positives/len(y_test)*100:.1f}%)")
                print(
                    f"   Пропущенные мошенники: {false_negatives} ({false_negatives/len(y_test)*100:.1f}%)")

        return test_results

    def generate_comprehensive_report(self, test_results=None) -> Dict:
        """Генерация комплексного отчета с реальными метриками"""
        print("\n📋 ГЕНЕРАЦИЯ КОМПЛЕКСНОГО ОТЧЕТА")
        print("=" * 60)

        try:
            if not self.results:
                raise ValueError(
                    "❌ Нет результатов обучения для генерации отчета!")

            report = {
                'training_completed_at': datetime.now().isoformat(),
                'best_model': self.best_model if self.best_model else 'Unknown',
                'best_score': float(self.best_score) if self.best_score else 0.0,
                'total_features': len(self.feature_engineer.feature_names) if self.feature_engineer.feature_names else 0,
                'gpu_used': self.use_gpu,
                'models_results': {}
            }

            # Результаты по каждой модели
            for model_name, model_info in self.results.items():
                try:
                    model_result = {
                        'cv_mean_auc': float(model_info.get('cv_mean', 0)),
                        'cv_std_auc': float(model_info.get('cv_std', 0)),
                        'cv_scores': model_info.get('cv_scores', []).tolist() if hasattr(model_info.get('cv_scores', []), 'tolist') else [],
                        'detailed_cv_metrics': {}
                    }

                    # Безопасная обработка детальных метрик
                    if 'detailed_metrics' in model_info and model_info['detailed_metrics']:
                        for k, v in model_info['detailed_metrics'].items():
                            try:
                                if np.isscalar(v):
                                    model_result['detailed_cv_metrics'][k] = float(
                                        v)
                                elif hasattr(v, 'tolist'):
                                    model_result['detailed_cv_metrics'][k] = v.tolist(
                                    )
                                else:
                                    model_result['detailed_cv_metrics'][k] = str(
                                        v)
                            except Exception as e:
                                print(
                                    f"⚠️ Ошибка при обработке метрики {k}: {e}")
                                model_result['detailed_cv_metrics'][k] = None

                    # Добавляем важность признаков
                    if 'feature_importance' in model_info and len(model_info['feature_importance']) > 0:
                        try:
                            model_result['top_features'] = model_info['feature_importance'].head(
                                15).to_dict('records')
                        except Exception as e:
                            print(
                                f"⚠️ Ошибка при обработке важности признаков для {model_name}: {e}")
                            model_result['top_features'] = []

                    report['models_results'][model_name] = model_result

                except Exception as e:
                    print(
                        f"⚠️ Ошибка при обработке результатов модели {model_name}: {e}")
                    continue

            # Результаты тестирования
            if test_results:
                report['test_results'] = {}
                for model_name, test_result in test_results.items():
                    try:
                        report['test_results'][model_name] = {
                            'auc': float(test_result.get('auc', 0)),
                            'accuracy': float(test_result.get('accuracy', 0)),
                            'precision': float(test_result.get('precision', 0)),
                            'recall': float(test_result.get('recall', 0)),
                            'f1': float(test_result.get('f1', 0)),
                            'confusion_matrix': test_result.get('confusion_matrix', [[0, 0], [0, 0]]).tolist() if hasattr(test_result.get('confusion_matrix', [[0, 0], [0, 0]]), 'tolist') else [[0, 0], [0, 0]],
                            'classification_report': test_result.get('classification_report', {})
                        }
                    except Exception as e:
                        print(
                            f"⚠️ Ошибка при обработке тестовых результатов для {model_name}: {e}")
                        continue

            # Лучшая модель - детальная информация
            if self.best_model and self.best_model in self.results:
                try:
                    best_model_info = self.results[self.best_model]
                    report['best_model_details'] = {}

                    # Важность признаков
                    if 'feature_importance' in best_model_info and len(best_model_info['feature_importance']) > 0:
                        try:
                            report['best_model_details']['feature_importance'] = best_model_info['feature_importance'].head(
                                20).to_dict('records')
                        except Exception as e:
                            print(
                                f"⚠️ Ошибка при обработке важности признаков лучшей модели: {e}")
                            report['best_model_details']['feature_importance'] = []

                    # Детальные метрики CV
                    if 'detailed_metrics' in best_model_info:
                        report['best_model_details']['cv_detailed_metrics'] = {}
                        for k, v in best_model_info['detailed_metrics'].items():
                            try:
                                if np.isscalar(v):
                                    report['best_model_details']['cv_detailed_metrics'][k] = float(
                                        v)
                                elif hasattr(v, 'tolist'):
                                    report['best_model_details']['cv_detailed_metrics'][k] = v.tolist(
                                    )
                                else:
                                    report['best_model_details']['cv_detailed_metrics'][k] = str(
                                        v)
                            except Exception as e:
                                print(
                                    f"⚠️ Ошибка при обработке детальной метрики {k}: {e}")

                except Exception as e:
                    print(
                        f"⚠️ Ошибка при обработке деталей лучшей модели: {e}")
                    report['best_model_details'] = {}

            # Анализ данных
            try:
                class_distribution = self._analyze_class_distribution()
                total_samples = 0

                if self.best_model and self.best_model in self.results:
                    cm = self.results[self.best_model].get(
                        'detailed_metrics', {}).get('confusion_matrix')
                    if cm is not None:
                        total_samples = int(cm.sum())

                report['data_analysis'] = {
                    'total_samples': total_samples,
                    'class_distribution': class_distribution
                }
            except Exception as e:
                print(f"⚠️ Ошибка при анализе данных: {e}")
                report['data_analysis'] = {
                    'total_samples': 0,
                    'class_distribution': {}
                }

            print(f"✅ Отчет сгенерирован для {len(self.results)} моделей")
            return report

        except Exception as e:
            print(f"❌ Критическая ошибка при генерации отчета: {e}")
            # Возвращаем минимальный отчет
            return {
                'training_completed_at': datetime.now().isoformat(),
                'best_model': 'Unknown',
                'best_score': 0.0,
                'total_features': 0,
                'gpu_used': self.use_gpu,
                'models_results': {},
                'error': str(e)
            }

    def _analyze_class_distribution(self):
        """Анализ распределения классов"""
        try:
            if not self.best_model or self.best_model not in self.results:
                print("⚠️ Нет лучшей модели для анализа распределения классов")
                return {}

            model_info = self.results[self.best_model]

            if 'detailed_metrics' not in model_info:
                print("⚠️ Нет детальных метрик для анализа классов")
                return {}

            if 'confusion_matrix' not in model_info['detailed_metrics']:
                print("⚠️ Нет матрицы ошибок для анализа классов")
                return {}

            cm = model_info['detailed_metrics']['confusion_matrix']

            if cm is None:
                print("⚠️ Матрица ошибок пуста")
                return {}

            total = cm.sum()

            if total == 0:
                print("⚠️ Матрица ошибок содержит только нули")
                return {}

            return {
                'total_samples': int(total),
                'negative_samples': int(cm[0].sum()),
                'positive_samples': int(cm[1].sum()),
                'positive_rate': float(cm[1].sum() / total) if total > 0 else 0.0
            }

        except Exception as e:
            print(f"⚠️ Ошибка при анализе распределения классов: {e}")
            return {}

    def update_config_with_metrics(self, report):
        """Обновление config.py с реальными метриками"""
        print("\n⚙️ ОБНОВЛЕНИЕ КОНФИГУРАЦИИ")
        print("=" * 40)

        try:
            if not self.best_model:
                print("⚠️ Нет лучшей модели для обновления конфигурации")
                return

            if 'models_results' not in report or self.best_model not in report['models_results']:
                print(f"⚠️ Нет результатов для модели {self.best_model}")
                return

            best_metrics = report['models_results'][self.best_model].get(
                'detailed_cv_metrics', {})

            # Безопасное получение метрик с значениями по умолчанию
            accuracy = best_metrics.get('accuracy', 0.0)
            precision = best_metrics.get('precision', 0.0)
            recall = best_metrics.get('recall', 0.0)
            f1 = best_metrics.get('f1', 0.0)
            confusion_matrix = best_metrics.get(
                'confusion_matrix', [[0, 0], [0, 0]])

            # Безопасное получение других данных
            total_features = report.get('total_features', 0)
            best_score = report.get('best_score', 0.0)

            # Важность признаков
            feature_importance = []
            if 'best_model_details' in report and 'feature_importance' in report['best_model_details']:
                feature_importance = report['best_model_details']['feature_importance'][:10]

            # Анализ данных
            total_samples = 0
            positive_rate = 0.0

            if 'data_analysis' in report:
                total_samples = report['data_analysis'].get('total_samples', 0)
                if 'class_distribution' in report['data_analysis']:
                    positive_rate = report['data_analysis']['class_distribution'].get(
                        'positive_rate', 0.0)

            # Формируем обновление для MODEL_METRICS в config.py
            metrics_update = f"""
# Метрики качества модели (обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
MODEL_METRICS = {{
    'last_updated': '{datetime.now().isoformat()}',
    'model_name': '{self.best_model}',
    'accuracy': {accuracy:.4f},
    'precision': {precision:.4f},
    'recall': {recall:.4f},
    'f1_score': {f1:.4f},
    'auc_roc': {best_score:.4f},
    'total_features': {total_features},
    'most_important_features': {feature_importance},
    'confusion_matrix': {confusion_matrix},
    'samples_analyzed': {total_samples},
    'fraud_rate_detected': {positive_rate:.3f}
}}
"""

            # Сохраняем в отдельный файл для обновления config.py
            with open('model_metrics_update.py', 'w', encoding='utf-8') as f:
                f.write(metrics_update)

            print("✅ Метрики сохранены в model_metrics_update.py")
            print("📝 Скопируйте содержимое в config.py для обновления UI")

        except Exception as e:
            print(f"❌ Ошибка при обновлении конфигурации: {e}")
            print("⚠️ Конфигурация не обновлена, но обучение завершено успешно")

    def save_model(self, model_name: str = None, filename: str = 'model.joblib') -> str:
        """Сохранение модели с метаданными"""
        try:
            if model_name is None:
                model_name = self.best_model

            if not model_name:
                raise ValueError("❌ Не указано имя модели для сохранения!")

            if model_name not in self.results:
                raise ValueError(
                    f"❌ Модель {model_name} не найдена в результатах обучения!")

            model_info = self.results[model_name]

            # Проверяем наличие основных компонентов
            if 'model' not in model_info:
                raise ValueError(
                    f"❌ Отсутствует обученная модель для {model_name}!")

            # Подготовка объекта для сохранения с защитой от ошибок
            model_data = {
                'model': model_info['model'],
                'feature_names': model_info.get('feature_names', []),
                'model_name': model_name,
                'cv_score': float(model_info.get('cv_mean', 0.0)),
                'feature_engineer': self.feature_engineer,
                'training_date': datetime.now().isoformat(),
                'detailed_metrics': model_info.get('detailed_metrics', {})
            }

            # Безопасное добавление важности признаков
            try:
                if 'feature_importance' in model_info and len(model_info['feature_importance']) > 0:
                    model_data['feature_importance'] = model_info['feature_importance'].to_dict(
                        'records')
                else:
                    model_data['feature_importance'] = []
            except Exception as e:
                print(f"⚠️ Ошибка при сохранении важности признаков: {e}")
                model_data['feature_importance'] = []

            # Сохранение
            joblib.dump(model_data, filename)
            print(f"✅ Модель {model_name} сохранена в {filename}")
            print(
                f"📊 Включены метрики и важность {len(model_data['feature_names'])} признаков")

            return filename

        except Exception as e:
            print(f"❌ Ошибка при сохранении модели: {e}")
            raise e

    def save_ensemble(self, filename: str = 'fraud_ensemble.joblib') -> str:
        """Сохранение ансамбля всех моделей для консенсуса"""
        try:
            if not self.results:
                raise ValueError(
                    "❌ Нет обученных моделей для сохранения ансамбля!")

            print("💾 Сохранение ансамбля всех моделей...")

            # Подготовка данных ансамбля
            ensemble_data = {
                'models': {},
                'best_model': self.best_model,
                'best_score': float(self.best_score),
                'feature_engineer': self.feature_engineer,
                'training_date': datetime.now().isoformat(),
                'ensemble_type': 'consensus_based',
                'total_models': len(self.results)
            }

            # Сохраняем все модели с их метриками
            for model_name, model_info in self.results.items():
                try:
                    model_data = {
                        'model': model_info['model'],
                        'cv_score': float(model_info.get('cv_mean', 0.0)),
                        'cv_std': float(model_info.get('cv_std', 0.0)),
                        'detailed_metrics': model_info.get('detailed_metrics', {}),
                        'feature_names': model_info.get('feature_names', [])
                    }

                    # Добавляем важность признаков
                    if 'feature_importance' in model_info and len(model_info['feature_importance']) > 0:
                        try:
                            model_data['feature_importance'] = model_info['feature_importance'].to_dict(
                                'records')
                        except Exception as e:
                            print(
                                f"⚠️ Ошибка при сохранении важности для {model_name}: {e}")
                            model_data['feature_importance'] = []
                    else:
                        model_data['feature_importance'] = []

                    ensemble_data['models'][model_name] = model_data
                    print(
                        f"  ✅ Добавлена модель {model_name} (AUC: {model_data['cv_score']:.4f})")

                except Exception as e:
                    print(f"⚠️ Ошибка при подготовке модели {model_name}: {e}")
                    continue

            # Общие признаки для всех моделей
            if self.feature_engineer and self.feature_engineer.feature_names:
                ensemble_data['feature_names'] = self.feature_engineer.feature_names
            else:
                ensemble_data['feature_names'] = []

            # Сохранение ансамбля
            joblib.dump(ensemble_data, filename)

            print(
                f"🎉 Ансамбль из {len(ensemble_data['models'])} моделей сохранен в {filename}")
            print(f"📊 Модели: {', '.join(ensemble_data['models'].keys())}")
            print(f"🏆 Лучшая модель: {ensemble_data['best_model']}")

            return filename

        except Exception as e:
            print(f"❌ Ошибка при сохранении ансамбля: {e}")
            raise e


def train_pipeline(train_path: str, test_path: str = None,
                   model_filename: str = 'fraud_model.joblib',
                   use_gpu: bool = True) -> Dict:
    """Полный пайплайн обучения с детальными метриками"""

    print("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ОБУЧЕНИЯ ML")
    print("=" * 80)

    try:
        # Проверка входных файлов
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"❌ Файл обучающих данных не найден: {train_path}")

        if test_path and not os.path.exists(test_path):
            print(
                f"⚠️ Файл тестовых данных не найден: {test_path}, продолжаем без тестирования")
            test_path = None

        # Инициализация
        print("🔧 Инициализация системы...")
        trainer = ModelTrainer(use_gpu=use_gpu)

        # Загрузка данных
        print("📁 Загрузка и подготовка данных...")
        train_df, test_df = trainer.load_data(train_path, test_path)

        if len(train_df) == 0:
            raise ValueError("❌ Обучающие данные пусты после загрузки!")

        # Обучение моделей
        print("🤖 Обучение моделей...")
        trainer.train_models(train_df)

        if not trainer.results:
            raise ValueError("❌ Ни одна модель не была обучена успешно!")

        # Оценка на тесте
        test_results = None
        if test_df is not None and len(test_df) > 0:
            print("🧪 Оценка на тестовых данных...")
            try:
                test_results = trainer.evaluate_on_test(test_df)
            except Exception as e:
                print(f"⚠️ Ошибка при тестировании: {e}")
                print("⚠️ Продолжаем без тестовых результатов")

        # Сохранение лучшей модели
        print("💾 Сохранение модели...")
        try:
            trainer.save_model(filename=model_filename)
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении модели: {e}")
            print("⚠️ Модель не сохранена, но обучение завершено")

        # Сохранение ансамбля для консенсуса
        print("🤝 Сохранение ансамбля моделей...")
        try:
            ensemble_filename = model_filename.replace(
                '.joblib', '_ensemble.joblib')
            trainer.save_ensemble(filename=ensemble_filename)
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении ансамбля: {e}")
            print("⚠️ Ансамбль не сохранен, но обучение завершено")

        # Генерация комплексного отчета
        print("📋 Генерация отчета...")
        report = trainer.generate_comprehensive_report(test_results)

        if not report or 'error' in report:
            print("⚠️ Ошибки при генерации отчета, создаем упрощенный отчет")
            report = {
                'training_completed_at': datetime.now().isoformat(),
                'best_model': trainer.best_model if trainer.best_model else 'Unknown',
                'best_score': float(trainer.best_score) if trainer.best_score else 0.0,
                'total_features': len(trainer.feature_engineer.feature_names) if trainer.feature_engineer.feature_names else 0,
                'gpu_used': use_gpu,
                'models_results': {},
                'status': 'completed_with_warnings'
            }

        # Обновление конфигурации
        print("⚙️ Обновление конфигурации...")
        try:
            trainer.update_config_with_metrics(report)
        except Exception as e:
            print(f"⚠️ Ошибка при обновлении конфигурации: {e}")

        # Сохранение отчета
        print("📄 Сохранение отчета...")
        try:
            with open('training_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении отчета: {e}")

        print(f"\n🎉 ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print(
            f"🏆 Лучшая модель: {report.get('best_model', 'Unknown')} (AUC: {report.get('best_score', 0):.4f})")
        print(f"📊 Отчет сохранен в training_report.json")
        print(f"💾 Модель сохранена в {model_filename}")
        print(f"⚙️ Метрики для UI сохранены в model_metrics_update.py")

        return report

    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА В ПАЙПЛАЙНЕ: {e}")
        print("📋 Генерация аварийного отчета...")

        # Создаем минимальный отчет об ошибке
        error_report = {
            'training_completed_at': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(e),
            'best_model': None,
            'best_score': 0.0,
            'total_features': 0,
            'gpu_used': use_gpu,
            'models_results': {}
        }

        try:
            with open('training_error_report.json', 'w', encoding='utf-8') as f:
                json.dump(error_report, f, ensure_ascii=False, indent=2)
            print("📄 Отчет об ошибке сохранен в training_error_report.json")
        except:
            pass

        raise e


if __name__ == "__main__":
    # Запуск обучения
    report = train_pipeline(
        train_path='data/dataset_train.json',
        test_path='data/dataset_test.json',
        model_filename='fraud_detection_model.joblib',
        use_gpu=True
    )
