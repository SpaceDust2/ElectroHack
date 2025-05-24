# 🔍 ML Fraud Detection System - Минималистичная версия

Система машинного обучения для обнаружения мошенников в потреблении электроэнергии.

## 🎯 Цель проекта

Выявление объектов, которые:

-   Ведут **коммерческую деятельность**
-   Используют **бытовые тарифы** на электроэнергию
-   Нарушают правила тарификации

### Интерпретация:

-   **isCommercial = True** → 🚨 **МОШЕННИК**
-   **isCommercial = False** → ✅ **Честный житель**

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск приложения

```bash
streamlit run app.py
```

### 3. Первичный анализ данных

```python
from data_analysis import run_full_analysis
analyzer = run_full_analysis()
```

### 4. Обучение модели

```python
from model_training import train_pipeline
report = train_pipeline(
    train_path='data/dataset_train.json',
    test_path='data/dataset_test.json',
    use_gpu=True
)
```

### 5. Предсказание

```python
from predictor import predict_fraud
results = predict_fraud('new_data.json', 'fraud_model.joblib')
```

## 📂 Структура проекта

```
ml_fraud_detection_minimal/
├── config.py              # Конфигурация системы
├── data_analysis.py       # Разведочный анализ данных
├── feature_engineering.py # Создание признаков
├── model_training.py      # Обучение моделей
├── predictor.py          # Предсказания
├── app.py                # Streamlit интерфейс
└── requirements.txt      # Зависимости
```

## 🔑 Ключевые возможности

### 📊 Разведочный анализ данных

-   Базовая статистика и распределения
-   Сравнение групп (мошенники vs честные)
-   Поиск оптимальных порогов
-   Применение правил детекции
-   Интерактивные визуализации

### 🤖 Машинное обучение

-   3 модели: CatBoost, XGBoost, Random Forest
-   GPU ускорение (если доступно)
-   50+ признаков для анализа
-   Кросс-валидация и оценка важности признаков

### 🔮 Предсказания

-   Вероятность мошенничества (0-100%)
-   Уровни риска (HIGH/MEDIUM/LOW/MINIMAL)
-   Анализ паттернов поведения
-   Интерпретация результатов

### 💻 Интерфейс

-   Минималистичный Streamlit UI
-   4 вкладки: Анализ, Обучение, Предсказание, Мониторинг
-   Экспорт результатов (JSON/CSV)
-   Интерактивные графики

## 📈 Признаки мошенничества

1. **Высокое минимальное потребление** (>50 кВт·ч)
2. **Отсутствие сезонности** (лето ≈ зима)
3. **Стабильное потребление** (CV < 0.3)
4. **Нет месяцев с нулевым потреблением**
5. **Высокое потребление на жителя** (>200 кВт·ч)

## 🛠️ API использование

### Анализ данных

```python
from data_analysis import FraudDataAnalyzer

analyzer = FraudDataAnalyzer()
analyzer.load_data('data/dataset_train.json')
analyzer.basic_statistics()
analyzer.create_visualizations()
```

### Обучение модели

```python
from model_training import ModelTrainer

trainer = ModelTrainer(use_gpu=True)
train_df, test_df = trainer.load_data('data/dataset_train.json', 'data/dataset_test.json')
trainer.train_models(train_df)
trainer.save_model(filename='my_model.joblib')
```

### Предсказания

```python
from predictor import FraudPredictor

predictor = FraudPredictor('fraud_model.joblib')
results = predictor.predict([{
    "accountId": "TEST001",
    "roomsCount": 2,
    "residentsCount": 1,
    "consumption": {"1": 600, "2": 600, ...}
}])
```

## 📊 Результаты

-   **Baseline AUC**: 0.7235
-   **Target AUC**: 0.75+
-   **Количество признаков**: 50+
-   **Время обучения**: ~5 мин с GPU

## 🔧 Настройка

Редактируйте `config.py` для изменения:

-   Параметров моделей
-   Правил детекции
-   Уровней риска
-   Путей к данным

## 📝 Примечания

-   Система автоматически обрабатывает отсутствующие значения `totalArea`
-   GPU ускорение работает с NVIDIA RTX 4080 и выше
-   Для больших датасетов рекомендуется batch обработка

## 📞 Поддержка

При возникновении проблем проверьте:

1. Установлены ли все зависимости
2. Корректность формата данных (JSON)
3. Наличие обученной модели (.joblib файл)

---

_ML Fraud Detection System v1.0 - Минималистичная версия_
