"""
🤖 Модуль обучения ML моделей
"""

from feature_engineering import FeatureEngineer
from config import MODEL_PARAMS
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import joblib
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Класс для обучения и оценки моделей"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.models = self._init_models()
        self.feature_engineer = FeatureEngineer()
        self.best_model = None
        self.best_score = 0
        self.results = {}

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

        return train_df, test_df

    def train_models(self, train_df: pd.DataFrame, cv_folds: int = 5) -> Dict[str, Dict]:
        """Обучение всех моделей с кросс-валидацией"""
        print("\n🚀 Обучение моделей...")

        # Подготовка данных
        X = train_df[self.feature_engineer.feature_names]
        y = train_df['isCommercial']

        # Стратифицированная кросс-валидация
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for model_name, model in self.models.items():
            print(f"\n📊 Обучение {model_name}...")

            # Кросс-валидация
            cv_scores = cross_val_score(
                model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1
            )

            # Обучение на полном датасете
            model.fit(X, y)

            # Сохранение результатов
            self.results[model_name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_names': self.feature_engineer.feature_names
            }

            print(f"  AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # Обновление лучшей модели
            if cv_scores.mean() > self.best_score:
                self.best_score = cv_scores.mean()
                self.best_model = model_name

        print(
            f"\n🏆 Лучшая модель: {self.best_model} (AUC: {self.best_score:.4f})")

        return self.results

    def evaluate_on_test(self, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """Оценка моделей на тестовом датасете"""
        if not self.results:
            raise ValueError("Сначала обучите модели!")

        print("\n📏 Оценка на тестовом датасете...")

        X_test = test_df[self.feature_engineer.feature_names]
        y_test = test_df['isCommercial']

        test_results = {}

        for model_name, model_info in self.results.items():
            model = model_info['model']

            # Предсказания
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # Метрики
            auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)

            test_results[model_name] = {
                'auc': auc,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            print(f"\n{model_name}:")
            print(f"  AUC: {auc:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")

        return test_results

    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """Получение важности признаков"""
        if model_name is None:
            model_name = self.best_model

        if model_name not in self.results:
            raise ValueError(f"Модель {model_name} не найдена!")

        model = self.results[model_name]['model']
        feature_names = self.results[model_name]['feature_names']

        # Получение важности
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            return pd.DataFrame()

        # Создание DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, model_name: str = None, filename: str = 'model.joblib') -> str:
        """Сохранение модели"""
        if model_name is None:
            model_name = self.best_model

        if model_name not in self.results:
            raise ValueError(f"Модель {model_name} не найдена!")

        # Подготовка объекта для сохранения
        model_data = {
            'model': self.results[model_name]['model'],
            'feature_names': self.results[model_name]['feature_names'],
            'model_name': model_name,
            'cv_score': self.results[model_name]['cv_mean'],
            'feature_engineer': self.feature_engineer
        }

        # Сохранение
        joblib.dump(model_data, filename)
        print(f"✅ Модель {model_name} сохранена в {filename}")

        return filename

    def generate_training_report(self) -> Dict:
        """Генерация отчета об обучении"""
        report = {
            'best_model': self.best_model,
            'best_score': float(self.best_score),
            'total_features': len(self.feature_engineer.feature_names),
            'models_results': {}
        }

        for model_name, model_info in self.results.items():
            report['models_results'][model_name] = {
                'cv_mean_auc': float(model_info['cv_mean']),
                'cv_std_auc': float(model_info['cv_std']),
                'cv_scores': model_info['cv_scores'].tolist()
            }

        # Топ важных признаков
        if self.best_model:
            importance_df = self.get_feature_importance()
            if not importance_df.empty:
                report['top_features'] = importance_df.head(
                    10).to_dict('records')

        return report


def train_pipeline(train_path: str, test_path: str = None,
                   model_filename: str = 'fraud_model.joblib',
                   use_gpu: bool = True) -> Dict:
    """Полный пайплайн обучения"""

    # Инициализация
    trainer = ModelTrainer(use_gpu=use_gpu)

    # Загрузка данных
    train_df, test_df = trainer.load_data(train_path, test_path)

    # Обучение моделей
    trainer.train_models(train_df)

    # Оценка на тесте
    test_results = None
    if test_df is not None:
        test_results = trainer.evaluate_on_test(test_df)

    # Сохранение лучшей модели
    trainer.save_model(filename=model_filename)

    # Генерация отчета
    report = trainer.generate_training_report()
    if test_results:
        report['test_results'] = {
            name: {
                'auc': float(results['auc']),
                'accuracy': float(results['accuracy'])
            }
            for name, results in test_results.items()
        }

    # Сохранение отчета
    with open('training_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n✅ Обучение завершено!")
    print(f"📊 Отчет сохранен в training_report.json")

    return report


if __name__ == "__main__":
    # Запуск обучения
    report = train_pipeline(
        train_path='data/dataset_train.json',
        test_path='data/dataset_test.json',
        model_filename='fraud_detection_model.joblib',
        use_gpu=True
    )
