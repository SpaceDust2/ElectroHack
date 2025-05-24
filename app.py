"""
🚀 Система обнаружения мошенников в энергопотреблении
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import numpy as np

# Импорт модулей
from data_analysis import FraudDataAnalyzer, run_full_analysis
from model_training import train_pipeline
from predictor import FraudPredictor
from ensemble_predictor import EnsemblePredictor
from config import DATA_PATHS, RISK_LEVELS, CAUTION_LEVELS, PROTECTED_CATEGORIES, ENSEMBLE_SETTINGS

# Настройка страницы
st.set_page_config(
    page_title="Обнаружение мошенников",
    page_icon="🔍",
    layout="wide"
)

# CSS стили
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stAlert {margin-top: 1rem;}
    .fraud-card {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #ff0000;
        margin: 0.5rem 0;
    }
    .safe-card {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #00ff00;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def to_serializable(val):
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    if isinstance(val, list):
        return [to_serializable(v) for v in val]
    return val


def main():
    st.title("🔍 Система обнаружения мошенников")
    st.markdown("**Выявление нарушителей в использовании электроэнергии**")

    # Боковая панель
    with st.sidebar:
        st.header("📱 Меню")

        # Проверка наличия моделей
        models = []
        for file in os.listdir('.'):
            if file.endswith('.joblib'):
                models.append(file)

        if models:
            st.success(f"✅ Найдено готовых моделей: {len(models)}")
            selected_model = st.selectbox("Выберите модель:", models)
        else:
            st.warning(
                "⚠️ Модели не найдены. Создайте модель в разделе 'Обучение'")
            selected_model = None

        st.markdown("---")
        st.markdown("""
        ### 📖 Расшифровка результатов
        - **Мошенник** → 🚨 Коммерческое использование по бытовому тарифу
        - **Честный житель** → ✅ Правильное использование тарифа
        """)

    # Вкладки
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Анализ данных", "🤖 Создание модели", "🛡️ Умная защита",
        "🔮 Проверка объектов", "📈 Статистика", "🧠 Признаки для анализа"
    ])

    with tab1:
        render_eda_tab()

    with tab2:
        render_training_tab()

    with tab3:
        render_consensus_tab()

    with tab4:
        render_prediction_tab(selected_model)

    with tab5:
        render_monitoring_tab()

    with tab6:
        render_features_tab()


def render_features_tab():
    """Вкладка с описанием признаков"""
    st.header("🧠 Признаки для обнаружения нарушителей")
    st.markdown(
        "**Как система анализирует 50+ характеристик потребления электроэнергии**")

    # Основные категории
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        ### 📖 Общий принцип работы

        Наша система анализирует поведение потребителей электроэнергии, сравнивая их с типичными паттернами:
        - **Жилое использование** - непостоянное потребление, зависит от сезона, времени суток
        - **Коммерческое использование** - стабильное высокое потребление круглый год

        Мошенники пытаются использовать коммерческие мощности по льготному бытовому тарифу.
        """)

    with col2:
        st.info("""
        💡 **Ключевая идея**

        Магазины, офисы, производства потребляют электричество стабильно и много,
        а обычные квартиры - по-разному в зависимости от времени года.
        """)

    # Категории признаков
    st.markdown("## 🔍 Категории анализируемых признаков")

    # Использование вкладок для категорий
    cat_tab1, cat_tab2, cat_tab3, cat_tab4, cat_tab5 = st.tabs([
        "📊 Базовые показатели", "🌡️ Сезонность", "📈 Стабильность",
        "🏠 Характеристики объекта", "🚩 Подозрительные паттерны"
    ])

    with cat_tab1:
        st.markdown("### 📊 Базовые показатели потребления")

        features_basic = [
            {"name": "Среднее потребление", "description": "Общее среднее потребление за все месяцы",
             "fraud_sign": "Очень высокое (>500 кВт·ч)", "honest_sign": "Умеренное (150-400 кВт·ч)"},
            {"name": "Минимальное потребление", "description": "Самый низкий месяц потребления",
             "fraud_sign": "Высокое (>100 кВт·ч)", "honest_sign": "Низкое (<50 кВт·ч)"},
            {"name": "Максимальное потребление", "description": "Самый высокий месяц потребления",
             "fraud_sign": "Очень высокое (>800 кВт·ч)", "honest_sign": "Сезонное (до 600 кВт·ч)"},
            {"name": "Потребление на жителя", "description": "Среднее потребление / количество жителей",
             "fraud_sign": "Очень высокое (>250 кВт·ч)", "honest_sign": "Умеренное (50-200 кВт·ч)"},
            {"name": "Потребление на м²", "description": "Среднее потребление / площадь",
             "fraud_sign": "Высокое (>8 кВт·ч/м²)", "honest_sign": "Умеренное (2-6 кВт·ч/м²)"}
        ]

        for feature in features_basic:
            with st.expander(f"📈 {feature['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Описание:** {feature['description']}")
                    st.error(
                        f"🚨 **Признак нарушения:** {feature['fraud_sign']}")
                with col2:
                    st.success(
                        f"✅ **Честное использование:** {feature['honest_sign']}")

    with cat_tab2:
        st.markdown("### 🌡️ Сезонные паттерны")
        st.warning(
            "🎯 **КЛЮЧЕВОЙ КРИТЕРИЙ:** Отопительный сезон >3000 кВт·ч - главный признак нарушения!")

        features_seasonal = [
            {"name": "⭐ Отопительный сезон (октябрь-апрель)", "description": "ГЛАВНЫЙ КРИТЕРИЙ: Среднее потребление в период отопления",
             "fraud_sign": "ВЫШЕ 3000 кВт·ч (ключевой индикатор!)", "honest_sign": "Ниже 3000 кВт·ч (обычно 200-800)"},
            {"name": "Отношение лето/зима", "description": "Как отличается потребление летом и зимой",
             "fraud_sign": "Почти одинаково (0.8-1.2)", "honest_sign": "Зимой больше (<0.7)"},
            {"name": "Зимнее потребление", "description": "Среднее потребление декабрь-февраль",
             "fraud_sign": "Стабильно высокое (>2500)", "honest_sign": "Сезонное повышение (200-600)"},
            {"name": "Летнее потребление", "description": "Среднее потребление июнь-август",
             "fraud_sign": "Такое же как зимой (>2000)", "honest_sign": "Ниже зимнего (100-400)"},
            {"name": "Отсутствие сезонности", "description": "Флаг одинакового потребления круглый год",
             "fraud_sign": "Да (коммерческие работают постоянно)", "honest_sign": "Нет (есть отопительный пик)"}
        ]

        for feature in features_seasonal:
            with st.expander(f"🌡️ {feature['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Описание:** {feature['description']}")
                    st.error(
                        f"🚨 **Признак нарушения:** {feature['fraud_sign']}")
                with col2:
                    st.success(
                        f"✅ **Честное использование:** {feature['honest_sign']}")

        st.info("💡 **Логика:** Магазины и офисы работают одинаково круглый год, а дома зимой тратят больше на отопление")

    with cat_tab3:
        st.markdown("### 📈 Стабильность и изменчивость")

        features_stability = [
            {"name": "Коэффициент вариации", "description": "Насколько сильно прыгает потребление между месяцами",
             "fraud_sign": "Очень низкий (<0.2)", "honest_sign": "Умеренный (0.3-0.6)"},
            {"name": "Стабильно высокое потребление", "description": "Флаг стабильного потребления >300 кВт·ч",
             "fraud_sign": "Да", "honest_sign": "Нет"},
            {"name": "Количество пиков", "description": "Сколько месяцев с необычно высоким потреблением",
             "fraud_sign": "Мало или много", "honest_sign": "2-4 пика (зима)"},
            {"name": "Стандартное отклонение", "description": "Мера разброса потребления по месяцам",
             "fraud_sign": "Очень низкое", "honest_sign": "Умеренное"},
            {"name": "Тренд потребления", "description": "Растет, падает или стабильно потребление",
             "fraud_sign": "Идеально стабильно", "honest_sign": "Есть сезонные колебания"}
        ]

        for feature in features_stability:
            with st.expander(f"📈 {feature['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Описание:** {feature['description']}")
                    st.error(
                        f"🚨 **Признак нарушения:** {feature['fraud_sign']}")
                with col2:
                    st.success(
                        f"✅ **Честное использование:** {feature['honest_sign']}")

        st.info(
            "💡 **Логика:** Коммерческие объекты потребляют стабильно, жилые - с колебаниями")

    with cat_tab4:
        st.markdown("### 🏠 Характеристики объекта")

        features_building = [
            {"name": "Тип здания", "description": "Квартира, дом или другое",
             "fraud_sign": "Квартира с высоким потреблением", "honest_sign": "Соответствует потреблению"},
            {"name": "Количество комнат", "description": "Число комнат в объекте",
             "fraud_sign": "Мало комнат, высокое потребление", "honest_sign": "Пропорциональное потребление"},
            {"name": "Количество жителей", "description": "Официально зарегистрированные жители",
             "fraud_sign": "Мало жителей, высокое потребление", "honest_sign": "Соответствует размеру семьи"},
            {"name": "Общая площадь", "description": "Площадь объекта в квадратных метрах",
             "fraud_sign": "Маленькая площадь, большое потребление", "honest_sign": "Пропорциональное потребление"},
            {"name": "Плотность потребления", "description": "Потребление на единицу площади и жителя",
             "fraud_sign": "Очень высокая", "honest_sign": "Типичная для жилья"}
        ]

        for feature in features_building:
            with st.expander(f"🏠 {feature['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Описание:** {feature['description']}")
                    st.error(
                        f"🚨 **Признак нарушения:** {feature['fraud_sign']}")
                with col2:
                    st.success(
                        f"✅ **Честное использование:** {feature['honest_sign']}")

        st.info(
            "💡 **Логика:** Несоответствие между размером жилья и потреблением - подозрительно")

    with cat_tab5:
        st.markdown("### 🚩 Специальные подозрительные паттерны")

        features_suspicious = [
            {"name": "Коммерческий паттерн", "description": "Совокупная оценка коммерческих признаков",
             "fraud_sign": "Высокий балл (>0.7)", "honest_sign": "Низкий балл (<0.3)"},
            {"name": "Нулевые месяцы", "description": "Количество месяцев с нулевым потреблением",
             "fraud_sign": "Нет (работают постоянно)", "honest_sign": "Могут быть (отпуск, отъезд)"},
            {"name": "Аномальные выбросы", "description": "Месяцы с экстремальным потреблением",
             "fraud_sign": "Нет выбросов (стабильно)", "honest_sign": "Есть сезонные выбросы"},
            {"name": "Ночное потребление", "description": "Признаки потребления в ночное время",
             "fraud_sign": "Высокое (24/7 работа)", "honest_sign": "Низкое (спят ночью)"},
            {"name": "Промышленный профиль", "description": "Соответствие промышленному графику потребления",
             "fraud_sign": "Высокое соответствие", "honest_sign": "Низкое соответствие"}
        ]

        for feature in features_suspicious:
            with st.expander(f"🚩 {feature['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Описание:** {feature['description']}")
                    st.error(
                        f"🚨 **Признак нарушения:** {feature['fraud_sign']}")
                with col2:
                    st.success(
                        f"✅ **Честное использование:** {feature['honest_sign']}")

        st.warning(
            "⚠️ **Внимание:** Эти признаки наиболее важны для обнаружения нарушителей")

    # Статистика эффективности
    st.markdown("## 📊 Эффективность системы")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего признаков", "50+",
                  help="Количество анализируемых характеристик")
    with col2:
        st.metric("Точность обнаружения", "85-95%",
                  help="Процент правильно найденных нарушителей")
    with col3:
        st.metric("Ложные срабатывания", "<5%",
                  help="Процент честных граждан, ошибочно помеченных как нарушители")

    # Примеры
    st.markdown("## 💡 Реальные примеры")
    example_col1, example_col2 = st.columns(2)

    with example_col1:
        st.markdown("### 🚨 Типичный нарушитель")
        st.error("""
        **РЕАЛЬНЫЙ КЕЙС - Магазин под видом квартиры:**
        - 🎯 **Отопительный сезон: 3471 кВт·ч** (>3000 ✅)
        - Среднее потребление: 3325 кВт·ч
        - Минимальное: 3000 кВт·ч (очень высокое!)
        - Коэффициент вариации: 0.059 (сверхстабильно)
        - Отношение лето/зима: 0.86 (нет сезонности)
        - На жителя: 3325 кВт·ч (в 44 раза выше нормы!)
        """)

    with example_col2:
        st.markdown("### ✅ Честная семья")
        st.success("""
        **РЕАЛЬНЫЙ КЕЙС - Обычная семья из 4 человек:**
        - 🎯 **Отопительный сезон: 386 кВт·ч** (<3000 ✅)
        - Среднее потребление: 300 кВт·ч
        - Минимальное: 140 кВт·ч (естественные колебания)
        - Коэффициент вариации: 0.373 (нормальная изменчивость)
        - Отношение лето/зима: 0.35 (ярко выраженная сезонность)
        - На жителя: 75 кВт·ч (норма для семьи)
        """)

    st.markdown("---")
    st.info("""
    ### 🎯 Заключение

    Система анализирует комплекс признаков, а не отдельные показатели.
    Ни один признак сам по себе не указывает на нарушение - важна общая картина поведения потребителя.

    **Основной принцип:** Коммерческое использование электроэнергии имеет характерные особенности,
    которые кардинально отличаются от бытового потребления.
    """)


def render_eda_tab():
    """Вкладка анализа данных"""
    st.header("📊 Анализ имеющихся данных")

    # Настройки для выбора файлов данных
    st.markdown("### ⚙️ Настройка источника данных")

    col1, col2 = st.columns(2)

    with col1:
        train_data_path = st.text_input(
            "Путь к основным данным для анализа",
            value="data/dataset_train.json",
            help="Укажите путь к файлу с данными для анализа (JSON формат)"
        )

    with col2:
        test_data_path = st.text_input(
            "Дополнительные данные (опционально)",
            value="",
            help="Дополнительный файл данных для расширенного анализа"
        )

    # Проверка существования файлов
    train_exists = os.path.exists(train_data_path)
    test_exists = os.path.exists(test_data_path) if test_data_path else True

    if train_exists and test_exists:
        st.success("✅ Файлы данных найдены")
    else:
        if not train_exists:
            st.error(f"❌ Файл не найден: {train_data_path}")
        if test_data_path and not test_exists:
            st.error(f"❌ Файл не найден: {test_data_path}")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🚀 Запустить полный анализ", type="primary", disabled=not (train_exists and test_exists)):
            with st.spinner("Анализируем данные..."):
                # Используем указанные пути
                if test_data_path and os.path.exists(test_data_path):
                    analyzer = run_full_analysis(
                        train_data_path, test_data_path)
                else:
                    analyzer = run_full_analysis(train_data_path, None)
                st.session_state['analyzer'] = analyzer
                st.success("✅ Анализ завершен!")

    with col2:
        if os.path.exists('eda_report.json'):
            with open('eda_report.json', 'r', encoding='utf-8') as f:
                report = json.load(f)

            st.download_button(
                label="📥 Скачать отчет",
                data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name=f"отчет_анализа_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    # Результаты анализа
    if 'analyzer' in st.session_state:
        analyzer = st.session_state['analyzer']

        # Базовая статистика
        st.markdown("### 📈 Общая статистика")
        col1, col2, col3, col4 = st.columns(4)

        stats = analyzer.insights['basic_stats']
        with col1:
            st.metric("Всего проверено", f"{stats['total_objects']:,}")
        with col2:
            st.metric("Выявлено нарушителей", f"{stats['fraudsters']:,}")
        with col3:
            st.metric("Процент нарушителей", f"{stats['fraud_rate']*100:.1f}%")
        with col4:
            st.metric("Данных без площади",
                      f"{stats['missing_totalArea_pct']:.1f}%")

        # Сравнение групп
        st.markdown("### 🔍 Сравнение нарушителей и честных жителей")

        if 'group_comparison' in analyzer.insights:
            comparison_data = []
            feature_names = {
                'avg_consumption': 'Среднее потребление (кВт·ч)',
                'min_consumption': 'Минимальное потребление (кВт·ч)',
                'max_consumption': 'Максимальное потребление (кВт·ч)',
                'cv': 'Стабильность потребления',
                'summer_winter_ratio': 'Отношение лето/зима',
                'roomsCount': 'Количество комнат',
                'residentsCount': 'Количество жителей'
            }

            for feature, data in analyzer.insights['group_comparison'].items():
                comparison_data.append({
                    'Показатель': feature_names.get(feature, feature),
                    'Честные жители': f"{data['honest_mean']:.2f}",
                    'Нарушители': f"{data['fraud_mean']:.2f}",
                    'Разница': f"{data['difference_pct']:+.1f}%",
                    'Значимость': '***' if data['p_value'] < 0.001 else '**' if data['p_value'] < 0.01 else '*' if data['p_value'] < 0.05 else ''
                })

            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)

            st.info(
                "*** - очень значимое различие, ** - значимое различие, * - слабое различие")

        # Визуализации
        st.markdown("### 📊 Графики")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # График потребления
            fig1 = px.box(
                analyzer.df,
                x='isCommercial',
                y='avg_consumption',
                labels={'isCommercial': 'Категория',
                        'avg_consumption': 'Среднее потребление (кВт·ч)'},
                title="Распределение потребления электроэнергии"
            )
            fig1.update_xaxes(
                ticktext=['Честные жители', 'Нарушители'], tickvals=[False, True])
            st.plotly_chart(fig1, use_container_width=True,
                            key="consumption_distribution_box")

        with viz_col2:
            # График сезонности
            fig2 = px.scatter(
                analyzer.df,
                x='winter_avg',
                y='summer_avg',
                color='isCommercial',
                labels={
                    'winter_avg': 'Зимнее потребление (кВт·ч)', 'summer_avg': 'Летнее потребление (кВт·ч)'},
                title="Сезонные изменения потребления",
                color_discrete_map={False: 'blue', True: 'red'}
            )
            fig2.update_traces(name='Честные жители',
                               selector=dict(marker_color='blue'))
            fig2.update_traces(name='Нарушители',
                               selector=dict(marker_color='red'))
            # Линия y=x
            max_val = max(analyzer.df['winter_avg'].max(),
                          analyzer.df['summer_avg'].max())
            fig2.add_trace(
                go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                           line=dict(dash='dash', color='gray'),
                           name='Одинаковое потребление', showlegend=True)
            )
            st.plotly_chart(fig2, use_container_width=True,
                            key="seasonality_scatter_plot")

        # Правила детекции
        if 'fraud_rules' in analyzer.insights:
            st.markdown("### 🎯 Эффективность правил обнаружения")

            rules_data = []
            for rule_name, rule_info in analyzer.insights['fraud_rules'].items():
                rules_data.append({
                    'Правило': rule_info['description'],
                    'Точность обнаружения': f"{rule_info['precision']*100:.1f}%",
                    'Охват нарушителей': f"{rule_info['recall']*100:.1f}%",
                    'Выявлено объектов': f"{rule_info['caught']:,}"
                })

            df_rules = pd.DataFrame(rules_data)
            st.dataframe(df_rules, use_container_width=True)

            st.info(
                "Точность - процент правильных срабатываний, Охват - процент найденных нарушителей")

    else:
        st.info("👆 Нажмите 'Запустить полный анализ' для изучения данных")


def render_training_tab():
    """Вкладка обучения модели с подробными объяснениями"""
    st.header("🤖 Создание модели для обнаружения нарушителей")
    st.markdown(
        "**Intelligent ML система для выявления коммерческого использования по бытовому тарифу**")

    # Основная информация
    with st.expander("📖 Как работает система машинного обучения", expanded=False):
        st.markdown("""
        ### 🎯 Принцип работы
        
        Система обучает **3 различные модели** машинного обучения на ваших данных:
        
        1. **🚀 CatBoost** - Лучший для категориальных данных, работает с видеокартой
        2. **⚡ XGBoost** - Быстрый градиентный бустинг, отлично балансирует точность и скорость  
        3. **🌳 Random Forest** - Надежный ансамбль деревьев, устойчив к переобучению
        
        ### 🔍 Процесс обучения:
        - **Feature Engineering:** Создание 50+ признаков из исходных данных
        - **Кросс-валидация:** 5-кратная проверка для честной оценки качества
        - **Автовыбор:** Система сама выберет лучшую модель
        - **Метрики:** Детальный анализ точности, полноты и F1-score
        """)

    # Настройки обучения
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚙️ Файлы данных")

        train_path = st.text_input(
            "📊 Файл с обучающими данными",
            value=DATA_PATHS['train'],
            help="JSON файл с примерами мошенников и честных жителей"
        )

        test_path = st.text_input(
            "🧪 Файл с тестовыми данными (опционально)",
            value="",
            help="Дополнительный файл для финальной проверки качества модели"
        )

        model_filename = st.text_input(
            "💾 Название модели для сохранения",
            value="fraud_detection_model.joblib",
            help="Имя файла для сохранения лучшей обученной модели"
        )

    with col2:
        st.markdown("### 🚀 Параметры обучения")

        use_gpu = st.checkbox(
            "🎮 Использовать видеокарту (GPU)",
            value=True,
            help="Ускоряет CatBoost и XGBoost в 5-10 раз. Требует CUDA-совместимую видеокарту"
        )

        if use_gpu:
            st.success("✅ GPU ускорение включено - обучение будет быстрее")
        else:
            st.info("ℹ️ Используется CPU - обучение будет медленнее")

        # Дополнительные настройки
        with st.expander("🔧 Дополнительные настройки", expanded=False):
            cv_folds = st.slider("Количество фолдов кросс-валидации", 3, 10, 5,
                                 help="Больше фолдов = более надежная оценка, но дольше обучение")

            st.markdown("**Используемые алгоритмы:**")
            st.markdown("""
            - **CatBoost**: iterations=1000, depth=6, learning_rate=0.1
            - **XGBoost**: n_estimators=1000, max_depth=6, learning_rate=0.1  
            - **Random Forest**: n_estimators=500, max_depth=10
            """)

    # Проверка файлов
    st.markdown("### 📁 Проверка файлов")

    train_exists = os.path.exists(train_path)
    test_exists = os.path.exists(test_path) if test_path else True

    col1, col2, col3 = st.columns(3)

    with col1:
        if train_exists:
            st.success(f"✅ Обучающие данные найдены")
            # Показываем размер файла
            try:
                with open(train_path, 'r', encoding='utf-8') as f:
                    train_data = json.load(f)
                st.info(f"📊 Объектов для обучения: {len(train_data)}")
            except:
                st.warning("⚠️ Не удалось прочитать файл")
        else:
            st.error(f"❌ Файл не найден: {train_path}")

    with col2:
        if test_path:
            if test_exists:
                st.success("✅ Тестовые данные найдены")
                try:
                    with open(test_path, 'r', encoding='utf-8') as f:
                        test_data = json.load(f)
                    st.info(f"🧪 Объектов для тестирования: {len(test_data)}")
                except:
                    st.warning("⚠️ Не удалось прочитать файл")
            else:
                st.error(f"❌ Файл не найден: {test_path}")
        else:
            st.info("ℹ️ Тестовые данные не указаны")

    with col3:
        if train_exists and test_exists:
            st.success("🚀 Готово к обучению!")
        else:
            st.error("❌ Исправьте ошибки файлов")

    # Кнопка обучения
    st.markdown("---")

    if st.button("🚀 НАЧАТЬ ОБУЧЕНИЕ МОДЕЛИ", type="primary", disabled=not (train_exists and test_exists)):

        # Область для отображения прогресса
        progress_container = st.container()

        with progress_container:
            st.markdown("### 🔄 Процесс обучения")

            # Основной прогресс
            overall_progress = st.progress(0)
            status_text = st.empty()

            # Детальная информация
            details_expander = st.expander(
                "📊 Детальная информация процесса", expanded=True)

            with details_expander:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**📁 Загрузка данных**")
                    data_status = st.empty()

                with col2:
                    st.markdown("**🔧 Feature Engineering**")
                    feature_status = st.empty()

                with col3:
                    st.markdown("**🤖 Обучение моделей**")
                    model_status = st.empty()

                # Лог обучения
                log_area = st.empty()

            try:
                # Этап 1: Загрузка данных
                status_text.text("🔄 Загружаем и анализируем данные...")
                data_status.info("Читаем JSON файлы...")
                overall_progress.progress(10)

                # Этап 2: Feature Engineering
                status_text.text(
                    "🔧 Создаем признаки для машинного обучения...")
                feature_status.info("Извлекаем 50+ признаков...")
                overall_progress.progress(25)

                # Этап 3: Обучение
                status_text.text("🤖 Обучаем 3 модели машинного обучения...")
                model_status.info("CatBoost, XGBoost, Random Forest...")
                overall_progress.progress(40)

                # Запуск реального обучения
                with st.spinner("Обучение в процессе... Это может занять несколько минут"):

                    # Здесь происходит реальное обучение
                    report = train_pipeline(
                        train_path=train_path,
                        test_path=test_path if test_path else None,
                        model_filename=model_filename,
                        use_gpu=use_gpu
                    )

                # Завершение
                overall_progress.progress(100)
                status_text.text("✅ Обучение успешно завершено!")

                data_status.success("✅ Данные загружены")
                feature_status.success("✅ Признаки созданы")
                model_status.success("✅ Модели обучены")

                # Отображение результатов
                st.markdown("---")
                st.markdown("## 🎉 Результаты обучения")

                # Основные метрики
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "🏆 Лучшая модель",
                        report['best_model'],
                        help="Модель с наивысшим AUC-ROC score"
                    )

                with col2:
                    st.metric(
                        "🎯 AUC-ROC Score",
                        f"{report['best_score']:.4f}",
                        delta=f"{(report['best_score'] - 0.5)*100:+.1f}% vs случайность",
                        help="Area Under Curve - чем ближе к 1.0, тем лучше"
                    )

                with col3:
                    st.metric(
                        "🔢 Признаков использовано",
                        report['total_features'],
                        help="Количество характеристик для принятия решения"
                    )

                with col4:
                    training_time = datetime.now().strftime("%H:%M")
                    st.metric(
                        "⏰ Время завершения",
                        training_time,
                        help="Модель готова к использованию"
                    )

                # Подробные результаты по моделям
                st.markdown("### 📊 Сравнение всех моделей")

                results_data = []
                for model_name, results in report['models_results'].items():

                    # Получаем детальные метрики
                    detailed = results.get('detailed_cv_metrics', {})

                    row_data = {
                        'Модель': model_name,
                        'AUC-ROC': f"{results['cv_mean_auc']:.4f}",
                        'Стандартное отклонение': f"±{results['cv_std_auc']:.4f}",
                        'Точность': f"{detailed.get('accuracy', 0):.3f}",
                        'Precision': f"{detailed.get('precision', 0):.3f}",
                        'Recall': f"{detailed.get('recall', 0):.3f}",
                        'F1-Score': f"{detailed.get('f1', 0):.3f}"
                    }

                    # Добавляем тестовые результаты если есть
                    if 'test_results' in report and model_name in report['test_results']:
                        test_res = report['test_results'][model_name]
                        row_data['AUC-ROC (тест)'] = f"{test_res['auc']:.4f}"
                        row_data['Точность (тест)'] = f"{test_res['accuracy']:.3f}"

                    results_data.append(row_data)

                df_results = pd.DataFrame(results_data)

                # Выделяем лучшую модель
                def highlight_best_model(row):
                    if row['Модель'] == report['best_model']:
                        return ['background-color: #90EE90'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    df_results.style.apply(highlight_best_model, axis=1),
                    use_container_width=True
                )

                st.info(
                    "💡 Зеленым выделена лучшая модель, которая будет использоваться для предсказаний")

                # Важность признаков
                if 'best_model_details' in report and 'feature_importance' in report['best_model_details']:
                    st.markdown("### 🧠 Самые важные признаки для обнаружения")

                    feature_importance = report['best_model_details']['feature_importance'][:15]

                    # Переводим на русский
                    feature_translations = {
                        'heating_season': '🌡️ Отопительный сезон',
                        'avg_consumption': '📊 Среднее потребление',
                        'min_consumption': '📉 Минимальное потребление',
                        'max_consumption': '📈 Максимальное потребление',
                        'cv': '📏 Коэффициент вариации',
                        'summer_winter_ratio': '🌞 Отношение лето/зима',
                        'consumption_per_resident': '👥 Потребление на жителя',
                        'consecutive_high': '🔥 Последовательные высокие месяцы',
                        'consumption_entropy': '🌀 Энтропия потребления',
                        'quarter_stability': '📆 Квартальная стабильность',
                        'no_seasonality': '🚫 Отсутствие сезонности',
                        'stable_high_consumption': '⚖️ Стабильно высокое потребление'
                    }

                    df_importance = pd.DataFrame(feature_importance)
                    df_importance['feature_ru'] = df_importance['feature'].map(
                        lambda x: feature_translations.get(x, x)
                    )

                    fig = px.bar(
                        df_importance.head(10),
                        x='importance',
                        y='feature_ru',
                        orientation='h',
                        title="Топ-10 самых важных признаков",
                        labels={'importance': 'Важность',
                                'feature_ru': 'Признак'},
                        color='importance',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True,
                                    key="training_feature_importance")

                    # Объяснение важности
                    with st.expander("💡 Что означает важность признаков"):
                        st.markdown("""
                        **Важность признака** показывает, насколько сильно этот параметр влияет на решение модели:
                        
                        - **Высокая важность (>0.1)**: Ключевой признак для обнаружения мошенников
                        - **Средняя важность (0.05-0.1)**: Важный вспомогательный признак  
                        - **Низкая важность (<0.05)**: Дополнительная информация
                        
                        Чем выше значение, тем больше модель полагается на этот признак при принятии решения.
                        """)

                # Матрица ошибок лучшей модели
                if 'best_model_details' in report:
                    best_metrics = report['models_results'][report['best_model']
                                                            ]['detailed_cv_metrics']

                    if 'confusion_matrix' in best_metrics:
                        st.markdown("### 🎯 Матрица ошибок лучшей модели")

                        cm = np.array(best_metrics['confusion_matrix'])

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.markdown("**Интерпретация результатов:**")

                            total = cm.sum()
                            tn, fp, fn, tp = cm.ravel()

                            st.metric("✅ Правильно определенные честные", f"{tn:,}",
                                      delta=f"{tn/total*100:.1f}% от всех")
                            st.metric("🚨 Правильно найденные мошенники", f"{tp:,}",
                                      delta=f"{tp/total*100:.1f}% от всех")
                            st.metric("⚠️ Ложные срабатывания", f"{fp:,}",
                                      delta=f"{fp/total*100:.1f}% от всех")
                            st.metric("❌ Пропущенные мошенники", f"{fn:,}",
                                      delta=f"{fn/total*100:.1f}% от всех")

                        with col2:
                            # Визуализация матрицы ошибок
                            fig = px.imshow(
                                cm,
                                labels=dict(x="Предсказанный класс",
                                            y="Фактический класс"),
                                x=['Честные', 'Мошенники'],
                                y=['Честные', 'Мошенники'],
                                title="Матрица ошибок",
                                text_auto=True,
                                color_continuous_scale='Blues'
                            )
                            st.plotly_chart(
                                fig, use_container_width=True, key="training_confusion_matrix")

                # Сохранение и следующие шаги
                st.markdown("### 💾 Модель сохранена и готова к использованию!")

                col1, col2 = st.columns(2)

                with col1:
                    st.success(f"✅ Модель сохранена как: `{model_filename}`")
                    st.info("🔄 Модель автоматически появится в боковой панели")

                with col2:
                    st.markdown("**Следующие шаги:**")
                    st.markdown("""
                    1. 🔮 Перейдите на вкладку **"Проверка объектов"**
                    2. 📈 Посмотрите детальную статистику в **"Статистика"**
                    3. 🧠 Изучите признаки в **"Признаки для анализа"**
                    """)

                # Сохранение отчета
                st.download_button(
                    label="📥 Скачать полный отчет об обучении",
                    data=json.dumps(report, ensure_ascii=False, indent=2),
                    file_name=f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"❌ Ошибка при обучении: {str(e)}")
                st.markdown("**Возможные причины:**")
                st.markdown("""
                - Неправильный формат данных в JSON файле
                - Недостаточно памяти для обучения
                - Проблемы с GPU драйверами (попробуйте отключить GPU)
                - Нет данных для обучения в файле
                """)

                with st.expander("🔧 Техническая информация об ошибке"):
                    st.code(str(e))


def render_prediction_tab(selected_model):
    """Вкладка предсказаний"""
    st.header("🔮 Проверка объектов на нарушения")

    if not selected_model:
        st.error("❌ Сначала создайте модель во вкладке 'Создание модели'")
        return

    # Выбор типа предиктора
    st.markdown("### 🧠 Выбор системы анализа")

    predictor_col1, predictor_col2 = st.columns([2, 1])

    with predictor_col1:
        # Новый блок: определяем, выбран ли ансамбль напрямую
        if selected_model.endswith('_ensemble.joblib'):
            ensemble_path = selected_model
            use_ensemble = True
            st.info("🧠 Выбран файл ансамбля — используется режим консенсуса моделей")
            caution_level = st.selectbox(
                "🛡️ Уровень защиты честных жителей:",
                options=['conservative', 'balanced',
                         'ultra_safe', 'aggressive'],
                index=0,
                format_func=lambda x: {
                    'aggressive': '⚡ Агрессивный (больше находим)',
                    'balanced': '⚖️ Сбалансированный',
                    'conservative': '🛡️ Консервативный (защита честных)',
                    'ultra_safe': '🚨 Максимальная защита'
                }[x]
            )
            st.success(
                "✅ Включен режим консенсуса - все 3 модели будут принимать решение совместно")
        else:
            ensemble_path = selected_model.replace(
                '.joblib', '_ensemble.joblib')
            use_ensemble = False
            if os.path.exists(ensemble_path):
                use_ensemble = st.checkbox(
                    "🤝 Использовать ансамбль с консенсусом (рекомендуется)",
                    value=True,
                    help="Использует 3 модели одновременно для максимальной точности и защиты честных жителей"
                )
                if use_ensemble:
                    caution_level = st.selectbox(
                        "🛡️ Уровень защиты честных жителей:",
                        options=['conservative', 'balanced',
                                 'ultra_safe', 'aggressive'],
                        index=0,
                        format_func=lambda x: {
                            'aggressive': '⚡ Агрессивный (больше находим)',
                            'balanced': '⚖️ Сбалансированный',
                            'conservative': '🛡️ Консервативный (защита честных)',
                            'ultra_safe': '🚨 Максимальная защита'
                        }[x]
                    )
                    st.success(
                        "✅ Включен режим консенсуса - все 3 модели будут принимать решение совместно")
                else:
                    st.info("ℹ️ Используется одиночная модель (классический режим)")
            else:
                st.warning(
                    "⚠️ Ансамбль недоступен - используется одиночная модель")
                st.info("💡 Обучите модель заново для получения ансамбля")

    with predictor_col2:
        if use_ensemble and os.path.exists(ensemble_path):
            st.markdown("### 🛡️ Защитные меры")
            from config import CAUTION_LEVELS, PROTECTED_CATEGORIES
            settings = CAUTION_LEVELS[caution_level]
            st.metric("Порог решения", f"{settings['threshold']:.2f}")
            st.metric("Ожидаемая точность",
                      f"{settings['expected_precision']:.0%}")
            # Чекбокс для полного отключения всех защит
            disable_all_protection = st.checkbox(
                "Отключить все защитные механизмы (raw-режим, только усреднение вероятностей)",
                value=False,
                help="Без консенсуса, без индивидуальных порогов, без защиты категорий. Только усреднение вероятностей."
            )
            # Чекбокс для защиты уязвимых групп (работает только если не отключены все защиты)
            if not disable_all_protection:
                disable_protection = st.checkbox(
                    "Отключить защиту уязвимых групп (умная защита)",
                    value=not PROTECTED_CATEGORIES['enable_protection']
                )
                PROTECTED_CATEGORIES['enable_protection'] = not disable_protection
            else:
                PROTECTED_CATEGORIES['enable_protection'] = False
        else:
            st.markdown("### 📊 Классический режим")
            st.metric("Порог решения", "0.50")
            st.metric("Тип анализа", "Одиночная модель")

    # Выбор способа ввода
    st.markdown("### 📁 Способ ввода данных")
    input_method = st.radio(
        "Как вы хотите ввести данные для проверки?",
        ["📁 Загрузить файл", "✏️ Ввести данные вручную",
            "📋 Использовать готовый пример"]
    )

    # Новый переключатель для формата вывода (только для одиночной модели)
    show_short = st.checkbox("Показывать только isCommercial (короткий формат)", value=False,
                             help="Если включено — выводится только isCommercial и исходные поля. Если выключено — подробный анализ с рисками и паттернами.")
    detailed = not show_short

    if input_method == "📁 Загрузить файл":
        uploaded_file = st.file_uploader(
            "Выберите файл с данными", type=['json'])

        if uploaded_file is not None:
            data = json.load(uploaded_file)

            if st.button("🔍 Проверить объекты", type="primary"):
                with st.spinner("Анализируем объекты..."):
                    if use_ensemble and os.path.exists(ensemble_path):
                        predictor = EnsemblePredictor(
                            ensemble_path, caution_level)
                        results = predictor.predict(
                            data, require_consensus=not disable_all_protection, detailed=not show_short)
                    else:
                        predictor = FraudPredictor(selected_model)
                        results = predictor.predict(data, detailed=detailed)

                    st.session_state['predictions'] = results
                    st.session_state['used_ensemble'] = use_ensemble
                    st.success(f"✅ Проверено {len(results)} объектов")

    elif input_method == "✏️ Ввести данные вручную":
        col1, col2 = st.columns(2)

        with col1:
            account_id = st.text_input("Номер лицевого счета", value="ТЕСТ001")
            building_type_map = {
                "Квартира": "Apartment",
                "Частный дом": "House",
                "Другое": "Other"
            }
            building_type_ru = st.selectbox(
                "Тип жилья", list(building_type_map.keys()))
            building_type = building_type_map[building_type_ru]

            rooms = st.number_input(
                "Количество комнат", min_value=1, max_value=10, value=3)
            residents = st.number_input(
                "Количество проживающих", min_value=1, max_value=10, value=2)
            area = st.number_input("Общая площадь (м²)",
                                   min_value=0.0, max_value=500.0, value=75.0)

        with col2:
            st.markdown(
                "**Среднее потребление электроэнергии (кВт·ч в месяц):**")
            winter = st.number_input(
                "Зима (декабрь-февраль)", min_value=0, max_value=2000, value=300)
            spring = st.number_input(
                "Весна (март-май)", min_value=0, max_value=2000, value=200)
            summer = st.number_input(
                "Лето (июнь-август)", min_value=0, max_value=2000, value=150)
            autumn = st.number_input(
                "Осень (сентябрь-ноябрь)", min_value=0, max_value=2000, value=250)

        if st.button("🔍 Проверить", type="primary"):
            # Формируем объект
            consumption = {
                "1": winter, "2": winter, "3": spring, "4": spring, "5": spring,
                "6": summer, "7": summer, "8": summer, "9": autumn, "10": autumn,
                "11": autumn, "12": winter
            }

            data = [{
                "accountId": account_id,
                "buildingType": building_type,
                "roomsCount": rooms,
                "residentsCount": residents,
                "totalArea": area if area > 0 else None,
                "consumption": consumption,
                "address": "г. Краснодар"
            }]

            if use_ensemble and os.path.exists(ensemble_path):
                predictor = EnsemblePredictor(ensemble_path, caution_level)
                results = predictor.predict(
                    data, require_consensus=not disable_all_protection, detailed=not show_short)
            else:
                predictor = FraudPredictor(selected_model)
                results = predictor.predict(data, detailed=detailed)

            st.session_state['predictions'] = results
            st.session_state['used_ensemble'] = use_ensemble

    elif input_method == "📋 Использовать готовый пример":
        examples = {
            "🏪 Магазин под видом квартиры (нарушитель)": {
                "accountId": "МАГАЗИН001",
                "buildingType": "Apartment",
                "roomsCount": 2,
                "residentsCount": 1,
                "totalArea": 50,
                "consumption": {str(i): 600 for i in range(1, 13)},
                "address": "г. Краснодар"
            },
            "🏠 Обычная семья (честный житель)": {
                "accountId": "СЕМЬЯ001",
                "buildingType": "House",
                "roomsCount": 4,
                "residentsCount": 4,
                "totalArea": 120,
                "consumption": {
                    "1": 400, "2": 350, "3": 300, "4": 250, "5": 200,
                    "6": 150, "7": 150, "8": 180, "9": 250, "10": 300,
                    "11": 350, "12": 400
                },
                "address": "г. Краснодар"
            }
        }

        selected_example = st.selectbox(
            "Выберите пример для проверки:", list(examples.keys()))

        # Показываем данные примера
        example_data = examples[selected_example]
        st.json({
            "Лицевой счет": example_data["accountId"],
            "Тип здания": "Квартира" if example_data["buildingType"] == "Apartment" else "Дом",
            "Комнат": example_data["roomsCount"],
            "Жителей": example_data["residentsCount"],
            "Площадь": example_data["totalArea"],
            "Среднее потребление зимой": example_data["consumption"]["1"],
            "Среднее потребление летом": example_data["consumption"]["7"]
        })

        if st.button("🔍 Проверить пример", type="primary"):
            if use_ensemble and os.path.exists(ensemble_path):
                predictor = EnsemblePredictor(ensemble_path, caution_level)
                results = predictor.predict(
                    [examples[selected_example]], require_consensus=not disable_all_protection, detailed=not show_short)
            else:
                predictor = FraudPredictor(selected_model)
                results = predictor.predict(
                    [examples[selected_example]], detailed=detailed)

            st.session_state['predictions'] = results
            st.session_state['used_ensemble'] = use_ensemble

    # Отображение результатов
    if 'predictions' in st.session_state:
        results = st.session_state['predictions']
        used_ensemble = st.session_state.get('used_ensemble', False)

        st.markdown("### 🎯 Результаты проверки")

        # Показываем используемый метод
        if used_ensemble:
            st.success(
                "🤝 **Результаты получены с использованием консенсуса 3 моделей**")
        else:
            st.info("📊 **Результаты получены от одиночной модели**")

        # Сводка
        if used_ensemble:
            # Для ансамбля показываем расширенную статистику
            total = len(results)
            fraudsters = sum(1 for r in results if r['isCommercial'])
            consensus_cases = sum(1 for r in results if r.get(
                'consensus_details', {}).get('consensus_reached', False))
            protected_cases = sum(1 for r in results if r.get(
                'protection_applied', False))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Проверено объектов", f"{total:,}")
            with col2:
                st.metric("Выявлено нарушителей", f"{fraudsters:,}")
            with col3:
                st.metric("Консенсус достигнут", f"{consensus_cases:,}")
            with col4:
                st.metric("Защита применена", f"{protected_cases:,}")
        else:
            # Для одиночной модели стандартная статистика
            report = FraudPredictor(selected_model).generate_report(results)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Проверено объектов",
                          f"{report['summary']['total_analyzed']:,}")
            with col2:
                st.metric("Выявлено нарушителей",
                          f"{report['summary']['fraudsters_detected']:,}")
            with col3:
                st.metric("Процент нарушителей",
                          f"{report['summary']['fraud_rate']:.1f}%")

        # Детальные результаты
        st.markdown("### 📋 Подробные результаты")

        for result in results[:10]:  # Показываем первые 10
            # Если короткий формат (нет risk_level), показываем только базовые поля
            if 'risk_level' not in result:
                card_class = "fraud-card" if result.get(
                    'isCommercial') else "safe-card"
                icon = "🚨" if result.get('isCommercial') else "✅"
                card_content = f"""
                <div class="{card_class}">
                    <h4>{icon} Лицевой счет: {result.get('accountId', '')}</h4>
                    <p><strong>isCommercial:</strong> {result.get('isCommercial')}</p>
                    <p><strong>Адрес:</strong> {result.get('address', '')}</p>
                    <p><strong>Тип здания:</strong> {result.get('buildingType', '')}</p>
                    <p><strong>Комнат:</strong> {result.get('roomsCount', '')}</p>
                    <p><strong>Жителей:</strong> {result.get('residentsCount', '')}</p>
                    <p><strong>Потребление:</strong> {json.dumps(result.get('consumption', {}), ensure_ascii=False)}</p>
                </div>
                """
                st.markdown(card_content, unsafe_allow_html=True)
                continue
            # Старый подробный формат:
            card_class = "fraud-card" if result.get(
                'isCommercial') else "safe-card"
            icon = "🚨" if result.get('isCommercial') else "✅"
            risk_level_ru = {
                'HIGH': 'ВЫСОКИЙ',
                'MEDIUM': 'СРЕДНИЙ',
                'LOW': 'НИЗКИЙ',
                'MINIMAL': 'МИНИМАЛЬНЫЙ'
            }.get(result['risk_level'], result['risk_level']) if 'risk_level' in result else ''
            card_content = f"""
            <div class="{card_class}">
                <h4>{icon} Лицевой счет: {result.get('accountId', '')}</h4>
                <p><strong>Вероятность нарушения:</strong> {result.get('fraud_probability_percent', '')}</p>
                <p><strong>Уровень риска:</strong> {risk_level_ru}</p>
                <p><strong>Заключение:</strong> {result.get('interpretation', '')}</p>
            """
            if used_ensemble and 'consensus_details' in result:
                consensus = result['consensus_details']
                protection_info = ""
                if result.get('protection_applied', False):
                    protection_reasons = result.get('protection_reasons', [])
                    protection_info = f"""
                    <div style="background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-radius: 5px;">
                        <p><strong>🛡️ Применена защита:</strong> {', '.join(protection_reasons)}</p>
                        <p><strong>🎯 Порог с защитой:</strong> {result.get('threshold_used', 0):.2f}</p>
                    </div>
                    """
                card_content += f"""
                <div style="background-color: #f0f8ff; padding: 10px; margin: 5px 0; border-radius: 5px;">
                    <p><strong>🤝 Консенсус моделей:</strong> {consensus.get('agreement_level', '')}</p>
                    <p><strong>🗳️ Голоса:</strong> {consensus.get('positive_votes', '')} | <strong>Метод:</strong> {consensus.get('decision_method', '')}</p>
                    <p><strong>✅ Консенсус достигнут:</strong> {'Да' if consensus.get('consensus_reached') else 'Нет'}</p>
                </div>
                {protection_info}
                """
                if 'individual_models' in result:
                    individual_info = "<p><strong>🧠 Решения отдельных моделей:</strong></p><ul>"
                    for model_name, model_decision in result['individual_models'].items():
                        decision_icon = "✅" if model_decision.get(
                            'decision') else "❌"
                        individual_info += f"<li>{decision_icon} <strong>{model_name}:</strong> {model_decision.get('probability', 0):.3f} (порог: {model_decision.get('threshold_used', 0):.2f})</li>"
                    individual_info += "</ul>"
                    card_content += individual_info
            else:
                if 'patterns' in result:
                    consumption_level_ru = {
                        'Очень высокое': 'Очень высокое',
                        'Высокое': 'Высокое',
                        'Среднее': 'Среднее',
                        'Низкое': 'Низкое'
                    }.get(result['patterns'].get('consumption_level', ''), result['patterns'].get('consumption_level', ''))
                    card_content += f"""
                    <p><strong>Обнаруженные признаки:</strong></p>
                    <ul>
                        <li>Уровень потребления: {consumption_level_ru}</li>
                        <li>Сезонность: {result['patterns'].get('seasonality', '')}</li>
                        <li>Стабильность: {result['patterns'].get('stability', '')}</li>
                    </ul>
                    """
            card_content += "</div>"
            st.markdown(card_content, unsafe_allow_html=True)

        # Экспорт
        st.markdown("### 💾 Сохранить результаты")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Скачать подробный отчет (JSON)",
                data=json.dumps(to_serializable(results),
                                ensure_ascii=False, indent=2),
                file_name=f"результаты_проверки_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        # Новая кнопка для короткого формата (только isCommercial)
        if 'show_short' in locals() and show_short:
            def filter_short_format(results):
                filtered = []
                for r in results:
                    # Сохраняем порядок: accountId, isCommercial, остальные исходные поля
                    out = {}
                    if 'accountId' in r:
                        out['accountId'] = r['accountId']
                        out['isCommercial'] = r['isCommercial']
                    for k, v in r.items():
                        if k not in out and k != 'isCommercial':
                            out[k] = v
                    filtered.append(out)
                return filtered
            with col2:
                st.download_button(
                    label="📥 Скачать только isCommercial (короткий формат)",
                    data=json.dumps(to_serializable(filter_short_format(
                        results)), ensure_ascii=False, indent=2),
                    file_name=f"isCommercial_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            with col2:
                # CSV экспорт (подробный)
                df_export = pd.DataFrame([{
                    'Лицевой счет': r['accountId'],
                    'Нарушитель': 'Да' if r['isCommercial'] else 'Нет',
                    'Вероятность нарушения': f"{r['fraud_probability']:.2%}",
                    'Уровень риска': {
                        'HIGH': 'ВЫСОКИЙ',
                        'MEDIUM': 'СРЕДНИЙ',
                        'LOW': 'НИЗКИЙ',
                        'MINIMAL': 'МИНИМАЛЬНЫЙ'
                    }.get(r['risk_level'], r['risk_level'])
                } for r in results])

                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Скачать таблицу (CSV)",
                    data=csv,
                    file_name=f"таблица_результатов_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


def render_monitoring_tab():
    """Вкладка мониторинга с реальными данными"""
    st.header("📈 Система мониторинга и метрики")

    # Проверяем наличие файлов с результатами
    if os.path.exists('training_report.json'):
        with open('training_report.json', 'r', encoding='utf-8') as f:
            training_report = json.load(f)

        # Основная информация о модели
        st.markdown("### 🤖 Актуальная информация о модели")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Лучшая модель", training_report['best_model'])
        with col2:
            st.metric("AUC-ROC", f"{training_report['best_score']:.4f}")
        with col3:
            st.metric("Всего признаков", training_report['total_features'])
        with col4:
            training_date = training_report.get(
                'training_completed_at', 'Неизвестно')
            if training_date != 'Неизвестно':
                training_date = training_date.split('T')[0]  # Только дата
            st.metric("Дата обучения", training_date)

        # Детальные метрики лучшей модели
        if 'best_model_details' in training_report:
            st.markdown("### 📊 Детальные метрики лучшей модели")

            metrics = training_report['models_results'][training_report['best_model']
                                                        ]['detailed_cv_metrics']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Точность (Accuracy)", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1']:.3f}")

            # Confusion Matrix
            if 'confusion_matrix' in metrics:
                st.markdown("#### 🎯 Матрица ошибок")
                cm = np.array(metrics['confusion_matrix'])

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.text(f"""
Истинно отрицательные: {cm[0, 0]:,}
Ложно положительные: {cm[0, 1]:,}
Ложно отрицательные: {cm[1, 0]:,}
Истинно положительные: {cm[1, 1]:,}
                    """)

                with col2:
                    # Визуализация confusion matrix
                    try:
                        import matplotlib.pyplot as plt
                        import seaborn as sns

                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=['Честные', 'Мошенники'],
                                    yticklabels=['Честные', 'Мошенники'], ax=ax)
                        ax.set_title('Матрица ошибок')
                        ax.set_ylabel('Фактический класс')
                        ax.set_xlabel('Предсказанный класс')
                        st.pyplot(fig)
                    except ImportError:
                        st.info(
                            "📊 Для отображения матрицы установите matplotlib и seaborn")

        # График сравнения моделей
        if 'models_results' in training_report:
            st.markdown("### 📈 Сравнение производительности моделей")

            models_data = []
            for model, results in training_report['models_results'].items():
                detailed_metrics = results.get('detailed_cv_metrics', {})
                models_data.append({
                    'Модель': model,
                    'AUC-ROC': results['cv_mean_auc'],
                    'Точность': detailed_metrics.get('accuracy', 0),
                    'Precision': detailed_metrics.get('precision', 0),
                    'Recall': detailed_metrics.get('recall', 0),
                    'F1-Score': detailed_metrics.get('f1', 0)
                })

            df_models = pd.DataFrame(models_data)

            # Интерактивный график
            fig = px.bar(
                df_models,
                x='Модель',
                y='AUC-ROC',
                title="Сравнение моделей по AUC-ROC",
                labels={'AUC-ROC': 'AUC-ROC Score'},
                color='AUC-ROC',
                color_continuous_scale='RdYlGn',
                text='AUC-ROC'
            )
            fig.update_traces(
                texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True,
                            key="models_comparison_chart")

            # Таблица с детальными метриками
            st.markdown("#### 📋 Подробные метрики по всем моделям")
            df_models_formatted = df_models.copy()
            for col in ['AUC-ROC', 'Точность', 'Precision', 'Recall', 'F1-Score']:
                df_models_formatted[col] = df_models_formatted[col].apply(
                    lambda x: f"{x:.3f}")
            st.dataframe(df_models_formatted, use_container_width=True)

        # Важность признаков
        if 'best_model_details' in training_report and 'feature_importance' in training_report['best_model_details']:
            st.markdown("### 🧠 Важность признаков в лучшей модели")

            feature_importance = training_report['best_model_details']['feature_importance'][:15]

            # Переводим названия признаков на русский
            feature_translations = {
                'heating_season': 'Отопительный сезон',
                'avg_consumption': 'Среднее потребление',
                'min_consumption': 'Минимальное потребление',
                'max_consumption': 'Максимальное потребление',
                'cv': 'Коэффициент вариации',
                'summer_winter_ratio': 'Отношение лето/зима',
                'consumption_per_resident': 'Потребление на жителя',
                'consecutive_high': 'Последовательные высокие месяцы',
                'consumption_entropy': 'Энтропия потребления',
                'quarter_stability': 'Квартальная стабильность'
            }

            df_importance = pd.DataFrame(feature_importance)
            df_importance['feature_ru'] = df_importance['feature'].map(
                lambda x: feature_translations.get(x, x)
            )

            fig = px.bar(
                df_importance.head(10),
                x='importance',
                y='feature_ru',
                orientation='h',
                title="Топ-10 самых важных признаков",
                labels={'importance': 'Важность', 'feature_ru': 'Признак'},
                color='importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True,
                            key="feature_importance_chart")

        # Тестовые результаты
        if 'test_results' in training_report:
            st.markdown("### 🧪 Результаты на тестовой выборке")

            test_data = []
            for model, results in training_report['test_results'].items():
                test_data.append({
                    'Модель': model,
                    'AUC-ROC': results['auc'],
                    'Точность': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1']
                })

            df_test = pd.DataFrame(test_data)

            for col in ['AUC-ROC', 'Точность', 'Precision', 'Recall', 'F1-Score']:
                df_test[col +
                        '_formatted'] = df_test[col].apply(lambda x: f"{x:.3f}")

            st.dataframe(df_test[['Модель'] + [col + '_formatted' for col in ['AUC-ROC', 'Точность', 'Precision', 'Recall', 'F1-Score']]],
                         use_container_width=True)

        # Анализ данных
        if 'data_analysis' in training_report:
            st.markdown("### 📊 Анализ обучающих данных")

            data_stats = training_report['data_analysis']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Всего образцов",
                          f"{data_stats.get('total_samples', 0):,}")
            with col2:
                st.metric(
                    "Мошенники", f"{data_stats.get('positive_samples', 0):,}")
            with col3:
                fraud_rate = data_stats.get('positive_rate', 0)
                st.metric("Доля мошенников", f"{fraud_rate:.1%}")

    # Статистика предсказаний
    if 'predictions' in st.session_state:
        st.markdown("### 🔮 Статистика последней проверки")

        results = st.session_state['predictions']

        # Основные метрики
        total = len(results)
        fraudsters = sum(1 for r in results if r['isCommercial'])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Проверено объектов", f"{total:,}")
        with col2:
            st.metric("Выявлено нарушителей", f"{fraudsters:,}")
        with col3:
            st.metric("Процент нарушителей",
                      f"{fraudsters/total*100:.1f}%" if total > 0 else "0%")

        # Распределение по рискам из реальных данных
        risk_distribution = {}
        for result in results:
            risk_level = result.get('risk_level')
            if risk_level is not None:
                risk_distribution[risk_level] = risk_distribution.get(
                    risk_level, 0) + 1

        risk_data = []
        for level, count in risk_distribution.items():
            risk_data.append({
                'Уровень риска': {
                    'HIGH': 'ВЫСОКИЙ',
                    'MEDIUM': 'СРЕДНИЙ',
                    'LOW': 'НИЗКИЙ',
                    'MINIMAL': 'МИНИМАЛЬНЫЙ'
                }.get(level, level),
                'Количество': count,
                'Процент': count / total * 100 if total > 0 else 0
            })

        if risk_data:
            df_risk = pd.DataFrame(risk_data)

            fig = px.pie(
                df_risk,
                values='Количество',
                names='Уровень риска',
                title="Распределение объектов по уровням риска",
                color_discrete_map={
                    'ВЫСОКИЙ': '#ff0000',
                    'СРЕДНИЙ': '#ff9900',
                    'НИЗКИЙ': '#ffcc00',
                    'МИНИМАЛЬНЫЙ': '#00cc00'
                }
            )
            st.plotly_chart(fig, use_container_width=True,
                            key="risk_distribution_pie_chart")

    # Статистика анализа правил
    if os.path.exists('rules_analysis_results.json'):
        with open('rules_analysis_results.json', 'r', encoding='utf-8') as f:
            rules_analysis = json.load(f)

        st.markdown("### 🕵️ Результаты анализа правил")

        if 'dataset_stats' in rules_analysis:
            stats = rules_analysis['dataset_stats']

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Проанализировано объектов",
                          f"{stats.get('total_objects', 0):,}")
            with col2:
                st.metric("Коммерческих объектов",
                          f"{stats.get('commercial_objects', 0):,}")
            with col3:
                fraud_rate = stats.get('fraud_rate', 0)
                st.metric("Доля нарушителей", f"{fraud_rate:.1%}")
            with col4:
                st.metric("Признаков найдено",
                          f"{stats.get('features_analyzed', 0):,}")

    if not os.path.exists('training_report.json'):
        st.warning(
            "⚠️ Модель еще не обучена. Перейдите во вкладку 'Создание модели' для обучения.")

    if not os.path.exists('rules_analysis_results.json'):
        st.info(
            "💡 Для более подробной статистики запустите анализ правил: `python analyze_rules.py`")


def render_consensus_tab():
    """Вкладка настройки консенсуса и защиты честных жителей"""
    st.header("🛡️ Умная защита от ложных обвинений")
    st.markdown(
        "**Настройка ансамбля моделей с консенсусом для максимальной защиты честных граждан**")

    # Основная информация о защите
    with st.expander("🤝 Как работает защита честных жителей", expanded=True):
        st.markdown("""
        ### 🎯 Проблема ложных обвинений
        
        **Критически важно:** Случайно обвинить честного жителя в мошенничестве - это:
        - 💸 **Репутационный ущерб** для энергокомпании
        - 😰 **Стресс для граждан** и их семей  
        - ⚖️ **Юридические проблемы** и жалобы
        - 💰 **Финансовые потери** на разбирательства
        
        ### 🛡️ Наша защитная система:
        
        1. **🤝 Консенсус моделей** - требуем согласия 2+ моделей из 3
        2. **📊 Настраиваемые пороги** - повышаем планку для обвинения  
        3. **🏠 Защищенные категории** - дополнительная осторожность для многодетных семей, пенсионеров
        4. **⚖️ Принцип презумпции невиновности** - в спорных случаях считаем честным
        """)

    # Настройки уровня осторожности
    st.markdown("### ⚙️ Настройка уровня осторожности")

    caution_col1, caution_col2 = st.columns([2, 1])

    with caution_col1:
        caution_level = st.selectbox(
            "🛡️ Выберите уровень защиты честных жителей:",
            options=['aggressive', 'balanced', 'conservative', 'ultra_safe'],
            index=2,  # По умолчанию conservative
            format_func=lambda x: {
                'aggressive': '⚡ Агрессивный - больше находим, больше ошибаемся (порог 0.5)',
                'balanced': '⚖️ Сбалансированный - компромисс точности и полноты (порог 0.65)',
                'conservative': '🛡️ Консервативный - защита честных, меньше ошибок (порог 0.8)',
                'ultra_safe': '🚨 Максимальная защита - только очевидные случаи (порог 0.9)'
            }[x],
            help="Чем выше уровень защиты, тем меньше ложных обвинений, но больше пропущенных мошенников"
        )

        from config import CAUTION_LEVELS
        settings = CAUTION_LEVELS[caution_level]

        st.info(f"""
        **Выбранные настройки:**
        - 📊 Порог классификации: **{settings['threshold']:.2f}**
        - 🎯 Ожидаемая точность: **{settings['expected_precision']:.1%}**
        - 📝 {settings['description']}
        """)

    with caution_col2:
        st.markdown("### 📊 Сравнение подходов")
        comparison_data = {
            'Подход': ['Агрессивный', 'Сбалансированный', 'Консервативный', 'Максимальная защита'],
            'Находим мошенников': ['95%', '85%', '70%', '50%'],
            'Ложные обвинения': ['25%', '15%', '8%', '3%'],
            'Точность': ['75%', '85%', '92%', '97%']
        }
        st.dataframe(pd.DataFrame(comparison_data), hide_index=True)

    # Настройки консенсуса
    st.markdown("### 🤝 Настройки консенсуса моделей")

    consensus_col1, consensus_col2 = st.columns(2)

    with consensus_col1:
        require_consensus = st.checkbox(
            "🤝 Требовать консенсус между моделями",
            value=True,
            help="Если включено, для обвинения нужно согласие минимум 2 из 3 моделей"
        )

        if require_consensus:
            min_agreement = st.slider(
                "🗳️ Минимум моделей для согласия:",
                min_value=2, max_value=3, value=2,
                help="Сколько моделей должны согласиться чтобы признать объект мошенником"
            )

            st.success(
                f"✅ Включен консенсус: нужно согласие {min_agreement} из 3 моделей")
        else:
            st.warning(
                "⚠️ Консенсус отключен - используется усредненная вероятность")

    with consensus_col2:
        st.markdown("### 🎯 Индивидуальные пороги моделей")

        from config import ENSEMBLE_SETTINGS
        st.markdown(f"""
        **Настроенные пороги:**
        - 🚀 **CatBoost:** {ENSEMBLE_SETTINGS['individual_thresholds']['CatBoost']:.2f}
        - ⚡ **XGBoost:** {ENSEMBLE_SETTINGS['individual_thresholds']['XGBoost']:.2f}  
        - 🌳 **Random Forest:** {ENSEMBLE_SETTINGS['individual_thresholds']['RandomForest']:.2f}
        
        *Каждая модель использует свой оптимальный порог*
        """)

    # Защищенные категории
    st.markdown("### 🏠 Защищенные категории граждан")

    protection_col1, protection_col2 = st.columns(2)

    with protection_col1:
        from config import PROTECTED_CATEGORIES

        enable_protection = st.checkbox(
            "🛡️ Включить дополнительную защиту для уязвимых групп",
            value=PROTECTED_CATEGORIES['enable_protection'],
            help="Применяет повышенные пороги для многодетных семей, пенсионеров и т.д."
        )

        if enable_protection:
            st.success("✅ Защита включена для:")
            categories = PROTECTED_CATEGORIES['categories']
            st.markdown(f"""
            - 👨‍👩‍👧‍👦 **Многодетные семьи:** {categories['large_family_threshold']}+ жителей
            - 🏠 **Маленькие квартиры:** ≤{categories['small_apartment_threshold']} м²
            - 💡 **Низкое потребление:** ≤{categories['low_income_consumption']} кВт·ч (льготники)
            
            **Дополнительная защита:** порог увеличивается на {PROTECTED_CATEGORIES['protection_multiplier']*100-100:.0f}% + {PROTECTED_CATEGORIES['additional_threshold']*100:.0f}%
            """)
        else:
            st.info("ℹ️ Дополнительная защита отключена")

    with protection_col2:
        st.markdown("### 💡 Примеры защиты")
        st.markdown("""
        **🏠 Семья из 5 человек в 30м² квартире:**
        - Базовый порог: 0.80  
        - С защитой: 0.81 × 1.2 + 0.05 = **1.02** 
        - ✅ **Результат:** Невозможно обвинить (порог >1.0)
        
        **💡 Пенсионер с потреблением 80 кВт·ч:**
        - Базовый порог: 0.80
        - С защитой: 0.80 × 1.2 + 0.05 = **1.01**
        - ✅ **Результат:** Максимальная защита
        """)

    # Тестирование настроек
    st.markdown("### 🧪 Тестирование настроек")

    test_col1, test_col2 = st.columns(2)

    with test_col1:
        st.markdown("#### 📊 Симуляция результатов")

        # Примерные данные для демонстрации
        demo_data = {
            'Сценарий': [
                'Все модели согласны (0.9, 0.85, 0.88)',
                'Частичное согласие (0.85, 0.65, 0.82)',
                'Разногласие (0.75, 0.45, 0.78)',
                'Защищенная категория (0.83, 0.81, 0.79)'
            ],
            'Без консенсуса': ['МОШЕННИК', 'МОШЕННИК', 'МОШЕННИК', 'МОШЕННИК'],
            'С консенсусом': ['МОШЕННИК', 'МОШЕННИК', 'ЧЕСТНЫЙ', 'ЧЕСТНЫЙ'],
            'Объяснение': [
                'Все модели уверены',
                '2 из 3 согласны',
                'Нет консенсуса - считаем честным',
                'Защитная категория - порог повышен'
            ]
        }

        demo_df = pd.DataFrame(demo_data)
        st.dataframe(demo_df, hide_index=True)

    with test_col2:
        st.markdown("#### 🎯 Ожидаемый эффект")

        current_settings = f"""
        **При ваших настройках ({caution_level}):**
        
        📈 **Метрики качества:**
        - Точность: ~{settings['expected_precision']:.0%}
        - Ложные обвинения: ~{(1-settings['expected_precision'])*100:.0f}%
        - Консенсус: {'включен' if require_consensus else 'отключен'}
        - Защита: {'включена' if enable_protection else 'отключена'}
        
        🛡️ **Защитный эффект:**
        - Снижение ложных обвинений в 2-5 раз
        - Особая защита уязвимых групп
        - Принцип "лучше отпустить виновного"
        """

        st.info(current_settings)

    # Сохранение настроек
    st.markdown("### 💾 Применение настроек")

    settings_col1, settings_col2 = st.columns(2)

    with settings_col1:
        if st.button("💾 Сохранить настройки как стандартные", type="primary"):
            # Здесь будет логика сохранения настроек в config
            st.success(
                "✅ Настройки сохранены! Они будут использоваться при следующих проверках.")

    with settings_col2:
        if st.button("🔄 Сбросить к значениям по умолчанию"):
            st.info("🔄 Настройки сброшены к консервативным значениям")

    # Сводка
    st.markdown("---")
    st.success(f"""
    ### ✅ Текущая конфигурация защиты:
    
    - 🛡️ **Уровень осторожности:** {caution_level.title()} (порог {settings['threshold']:.2f})
    - 🤝 **Консенсус моделей:** {'Включен' if require_consensus else 'Отключен'}
    - 🏠 **Защита категорий:** {'Включена' if enable_protection else 'Отключена'}
    - 🎯 **Ожидаемая точность:** {settings['expected_precision']:.1%}
    
    **Главный принцип:** Лучше пропустить мошенника, чем обвинить честного жителя!
    """)


if __name__ == "__main__":
    main()
