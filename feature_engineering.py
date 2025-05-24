"""
🔧 Модуль feature engineering для ML моделей
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class FeatureEngineer:
    """Класс для создания и преобразования признаков"""

    def __init__(self):
        self.feature_names = []
        self.area_estimator = 17.5  # м² на комнату

    def extract_features(self, data: List[Dict]) -> pd.DataFrame:
        """Извлечение всех признаков из сырых данных"""
        features_list = []

        for record in data:
            # Базовые признаки
            features = self._extract_basic_features(record)

            # Признаки потребления
            consumption_features = self._extract_consumption_features(record)
            features.update(consumption_features)

            # Сезонные признаки
            seasonal_features = self._extract_seasonal_features(record)
            features.update(seasonal_features)

            # Признаки стабильности
            stability_features = self._extract_stability_features(
                consumption_features)
            features.update(stability_features)

            # Признаки аномалий
            anomaly_features = self._extract_anomaly_features(
                record, consumption_features)
            features.update(anomaly_features)

            # Географические признаки
            geo_features = self._extract_geo_features(record)
            features.update(geo_features)

            features_list.append(features)

        df = pd.DataFrame(features_list)
        self.feature_names = [col for col in df.columns if col not in [
            'accountId', 'isCommercial']]

        return df

    def _extract_basic_features(self, record: Dict) -> Dict:
        """Базовые признаки"""
        features = {
            'accountId': record.get('accountId'),
            'isCommercial': record.get('isCommercial'),
            'roomsCount': record.get('roomsCount', 0),
            'residentsCount': record.get('residentsCount', 0)
        }

        # Обработка площади
        total_area = record.get('totalArea')
        if total_area is None or total_area <= 0:
            features['totalArea'] = features['roomsCount'] * \
                self.area_estimator
            features['area_is_estimated'] = 1
        else:
            features['totalArea'] = total_area
            features['area_is_estimated'] = 0

        # Производные признаки
        features['area_per_resident'] = (
            features['totalArea'] / features['residentsCount']
            if features['residentsCount'] > 0 else features['totalArea']
        )
        features['residents_per_room'] = (
            features['residentsCount'] / features['roomsCount']
            if features['roomsCount'] > 0 else 0
        )

        # One-hot encoding для типа здания
        building_type = record.get('buildingType', 'Unknown')
        features['building_type_Apartment'] = int(building_type == 'Apartment')
        features['building_type_House'] = int(building_type == 'House')
        features['building_type_Other'] = int(
            building_type not in ['Apartment', 'House'])

        return features

    def _extract_consumption_features(self, record: Dict) -> Dict:
        """Признаки потребления электроэнергии"""
        features = {}
        consumption = record.get('consumption', {})

        # Собираем месячные значения
        monthly_values = []
        zero_months = 0
        missing_months = 0

        for month in range(1, 13):
            value = consumption.get(str(month))
            if value is None:
                missing_months += 1
                features[f'month_{month}'] = 0
            else:
                features[f'month_{month}'] = value
                if value >= 0:
                    monthly_values.append(value)
                    if value == 0:
                        zero_months += 1

        # Статистические признаки
        if monthly_values:
            features['avg_consumption'] = np.mean(monthly_values)
            features['max_consumption'] = max(monthly_values)
            features['min_consumption'] = min(monthly_values)
            features['std_consumption'] = np.std(
                monthly_values) if len(monthly_values) > 1 else 0
            features['median_consumption'] = np.median(monthly_values)
            features['total_consumption'] = sum(monthly_values)

            # Квартили
            features['q1_consumption'] = np.percentile(monthly_values, 25)
            features['q3_consumption'] = np.percentile(monthly_values, 75)
            features['iqr_consumption'] = features['q3_consumption'] - \
                features['q1_consumption']

            # Дополнительные признаки
            features['consumption_range'] = features['max_consumption'] - \
                features['min_consumption']
            features['months_with_data'] = len(monthly_values)
            features['zero_consumption_months'] = zero_months
            features['missing_data_months'] = missing_months
            features['has_zero_months'] = int(zero_months > 0)

            # Относительные признаки
            features['min_to_max_ratio'] = (
                features['min_consumption'] / features['max_consumption']
                if features['max_consumption'] > 0 else 0
            )
            features['min_to_avg_ratio'] = (
                features['min_consumption'] / features['avg_consumption']
                if features['avg_consumption'] > 0 else 0
            )
        else:
            # Если нет данных
            for key in ['avg_consumption', 'max_consumption', 'min_consumption',
                        'std_consumption', 'median_consumption', 'total_consumption',
                        'q1_consumption', 'q3_consumption', 'iqr_consumption',
                        'consumption_range', 'min_to_max_ratio', 'min_to_avg_ratio']:
                features[key] = 0
            features['months_with_data'] = 0
            features['zero_consumption_months'] = 0
            features['missing_data_months'] = 12
            features['has_zero_months'] = 0

        return features

    def _extract_seasonal_features(self, record: Dict) -> Dict:
        """Сезонные признаки"""
        features = {}
        consumption = record.get('consumption', {})

        # Сезонные средние
        seasons = {
            'winter': ['12', '1', '2'],
            'spring': ['3', '4', '5'],
            'summer': ['6', '7', '8'],
            'autumn': ['9', '10', '11']
        }

        for season, months in seasons.items():
            values = []
            for month in months:
                value = consumption.get(month)
                if value is not None and value >= 0:
                    values.append(value)

            features[f'{season}_avg'] = np.mean(values) if values else 0
            features[f'{season}_total'] = sum(values) if values else 0

        # КЛЮЧЕВОЙ ПРИЗНАК: Отопительный сезон (октябрь-апрель)
        heating_months = ['10', '11', '12', '1', '2', '3', '4']
        heating_values = []
        for month in heating_months:
            value = consumption.get(month)
            if value is not None and value >= 0:
                heating_values.append(value)

        features['heating_season'] = np.mean(
            heating_values) if heating_values else 0
        features['heating_season_total'] = sum(
            heating_values) if heating_values else 0

        # Отношение отопительного сезона к летнему периоду
        if features['summer_avg'] > 0:
            features['heating_summer_ratio'] = features['heating_season'] / \
                features['summer_avg']
        else:
            features['heating_summer_ratio'] = 1

        # Индикатор высокого потребления в отопительный сезон (>3000 кВт·ч - критерий из кейса)
        features['high_heating_consumption'] = int(
            features['heating_season'] > 3000)

        # Сезонные отношения
        if features['winter_avg'] > 0:
            features['summer_winter_ratio'] = features['summer_avg'] / \
                features['winter_avg']
            features['spring_winter_ratio'] = features['spring_avg'] / \
                features['winter_avg']
            features['autumn_winter_ratio'] = features['autumn_avg'] / \
                features['winter_avg']
        else:
            features['summer_winter_ratio'] = 1
            features['spring_winter_ratio'] = 1
            features['autumn_winter_ratio'] = 1

        # Индикаторы сезонности
        features['has_seasonality'] = int(
            abs(features['summer_winter_ratio'] - 1) > 0.2 or
            abs(features['spring_winter_ratio'] - 1) > 0.2
        )
        features['no_seasonality'] = int(
            features['summer_winter_ratio'] >= 0.8 and
            features['summer_winter_ratio'] <= 1.2
        )

        # Максимальная сезонная разница
        seasonal_avgs = [features[f'{s}_avg'] for s in seasons.keys()]
        features['max_seasonal_diff'] = max(
            seasonal_avgs) - min(seasonal_avgs) if seasonal_avgs else 0

        return features

    def _extract_stability_features(self, consumption_features: Dict) -> Dict:
        """Признаки стабильности потребления"""
        features = {}

        avg_consumption = consumption_features.get('avg_consumption', 0)
        std_consumption = consumption_features.get('std_consumption', 0)

        # Коэффициент вариации
        features['cv'] = std_consumption / \
            avg_consumption if avg_consumption > 0 else 0

        # Категории стабильности
        features['very_stable'] = int(features['cv'] < 0.2)
        features['stable'] = int(features['cv'] < 0.3)
        features['unstable'] = int(features['cv'] > 0.5)

        # Дополнительные метрики стабильности
        features['consumption_stability_score'] = 1 / (1 + features['cv'])

        return features

    def _extract_anomaly_features(self, record: Dict, consumption_features: Dict) -> Dict:
        """Признаки аномалий и подозрительных паттернов"""
        features = {}

        avg_consumption = consumption_features.get('avg_consumption', 0)
        min_consumption = consumption_features.get('min_consumption', 0)
        cv = consumption_features.get('cv', 0)
        residents = record.get('residentsCount', 0)

        # Индикаторы аномалий
        features['high_min_consumption'] = int(min_consumption > 50)
        features['very_high_consumption'] = int(avg_consumption > 500)
        features['stable_high_consumption'] = int(
            avg_consumption > 400 and cv < 0.5)

        # Потребление на жителя
        features['consumption_per_resident'] = (
            avg_consumption / residents if residents > 0 else avg_consumption
        )
        features['high_consumption_per_resident'] = int(
            features['consumption_per_resident'] > 200
        )

        # Комплексные индикаторы
        features['commercial_pattern_score'] = 0
        if features['high_min_consumption']:
            features['commercial_pattern_score'] += 1
        if features['stable_high_consumption']:
            features['commercial_pattern_score'] += 1
        if consumption_features.get('no_seasonality', 0):
            features['commercial_pattern_score'] += 1
        if features['high_consumption_per_resident']:
            features['commercial_pattern_score'] += 1

        # Подозрительные комбинации
        features['suspicious_pattern'] = int(
            (residents <= 2 and avg_consumption > 500) or
            (min_consumption > 100 and cv < 0.3)
        )

        return features

    def _extract_geo_features(self, record: Dict) -> Dict:
        """Географические признаки"""
        features = {}
        address = record.get('address', '').lower()

        # Тип населенного пункта
        features['is_city'] = int(
            'г ' in address or 'город' in address or
            'г.' in address or 'city' in address
        )
        features['is_village'] = int(
            'с ' in address or 'село' in address or
            'пос' in address or 'дер' in address
        )

        # Крупные города
        major_cities = ['москв', 'петербург', 'краснодар', 'ростов', 'сочи']
        features['is_major_city'] = int(
            any(city in address for city in major_cities)
        )

        return features

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Группировка признаков по важности"""
        return {
            'consumption': [
                'avg_consumption', 'min_consumption', 'max_consumption',
                'total_consumption', 'consumption_range', 'heating_season'
            ],
            'seasonality': [
                'summer_winter_ratio', 'has_seasonality', 'no_seasonality',
                'max_seasonal_diff', 'heating_summer_ratio', 'high_heating_consumption'
            ],
            'stability': [
                'cv', 'very_stable', 'stable', 'unstable', 'consumption_stability_score'
            ],
            'anomalies': [
                'high_min_consumption', 'stable_high_consumption',
                'commercial_pattern_score', 'suspicious_pattern'
            ],
            'basic': [
                'roomsCount', 'residentsCount', 'totalArea',
                'consumption_per_resident', 'area_per_resident'
            ]
        }
