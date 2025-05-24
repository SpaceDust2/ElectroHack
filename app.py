"""
🚀 ML Fraud Detection System - Минималистичный интерфейс
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# Импорт модулей
from data_analysis import FraudDataAnalyzer, run_full_analysis
from model_training import train_pipeline
from predictor import FraudPredictor
from config import DATA_PATHS, RISK_LEVELS

# Настройка страницы
st.set_page_config(
    page_title="ML Fraud Detection",
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


def main():
    st.title("🔍 ML Fraud Detection System")
    st.markdown("**Обнаружение мошенников в потреблении электроэнергии**")

    # Боковая панель
    with st.sidebar:
        st.header("📱 Меню")

        # Проверка наличия моделей
        models = []
        for file in os.listdir('.'):
            if file.endswith('.joblib'):
                models.append(file)

        if models:
            st.success(f"✅ Найдено моделей: {len(models)}")
            selected_model = st.selectbox("Выберите модель:", models)
        else:
            st.warning(
                "⚠️ Модели не найдены. Обучите модель в разделе 'Обучение'")
            selected_model = None

        st.markdown("---")
        st.markdown("""
        ### 📖 Интерпретация
        - **isCommercial = True** → 🚨 МОШЕННИК
        - **isCommercial = False** → ✅ Честный житель
        """)

    # Вкладки
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Анализ данных", "🤖 Обучение модели", "🔮 Предсказание", "📈 Мониторинг"])

    with tab1:
        render_eda_tab()

    with tab2:
        render_training_tab()

    with tab3:
        render_prediction_tab(selected_model)

    with tab4:
        render_monitoring_tab()


def render_eda_tab():
    """Вкладка разведочного анализа данных"""
    st.header("📊 Разведочный анализ данных")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🚀 Запустить полный анализ", type="primary"):
            with st.spinner("Анализируем данные..."):
                analyzer = run_full_analysis()
                st.session_state['analyzer'] = analyzer
                st.success("✅ Анализ завершен!")

    with col2:
        if os.path.exists('eda_report.json'):
            with open('eda_report.json', 'r', encoding='utf-8') as f:
                report = json.load(f)

            st.download_button(
                label="📥 Скачать отчет",
                data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name=f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    # Результаты анализа
    if 'analyzer' in st.session_state:
        analyzer = st.session_state['analyzer']

        # Базовая статистика
        st.markdown("### 📈 Базовая статистика")
        col1, col2, col3, col4 = st.columns(4)

        stats = analyzer.insights['basic_stats']
        with col1:
            st.metric("Всего объектов", stats['total_objects'])
        with col2:
            st.metric("Мошенников", stats['fraudsters'])
        with col3:
            st.metric("% мошенников", f"{stats['fraud_rate']*100:.1f}%")
        with col4:
            st.metric("Без площади", f"{stats['missing_totalArea_pct']:.1f}%")

        # Сравнение групп
        st.markdown("### 🔍 Сравнение мошенников vs честных")

        if 'group_comparison' in analyzer.insights:
            comparison_data = []
            for feature, data in analyzer.insights['group_comparison'].items():
                comparison_data.append({
                    'Признак': feature,
                    'Честные': f"{data['honest_mean']:.2f}",
                    'Мошенники': f"{data['fraud_mean']:.2f}",
                    'Разница': f"{data['difference_pct']:+.1f}%",
                    'Значимость': '***' if data['p_value'] < 0.001 else '**' if data['p_value'] < 0.01 else '*' if data['p_value'] < 0.05 else ''
                })

            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)

        # Визуализации
        st.markdown("### 📊 Визуализации")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # График потребления
            fig1 = px.box(
                analyzer.df,
                x='isCommercial',
                y='avg_consumption',
                labels={'isCommercial': 'Тип',
                        'avg_consumption': 'Среднее потребление'},
                title="Распределение потребления"
            )
            fig1.update_xaxis(
                ticktext=['Честные', 'Мошенники'], tickvals=[False, True])
            st.plotly_chart(fig1, use_container_width=True)

        with viz_col2:
            # График сезонности
            fig2 = px.scatter(
                analyzer.df,
                x='winter_avg',
                y='summer_avg',
                color='isCommercial',
                labels={'winter_avg': 'Зима', 'summer_avg': 'Лето'},
                title="Сезонные паттерны",
                color_discrete_map={False: 'blue', True: 'red'}
            )
            # Линия y=x
            max_val = max(analyzer.df['winter_avg'].max(),
                          analyzer.df['summer_avg'].max())
            fig2.add_trace(
                go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                           line=dict(dash='dash', color='gray'), showlegend=False)
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Правила детекции
        if 'fraud_rules' in analyzer.insights:
            st.markdown("### 🎯 Эффективность правил детекции")

            rules_data = []
            for rule_name, rule_info in analyzer.insights['fraud_rules'].items():
                rules_data.append({
                    'Правило': rule_info['description'],
                    'Точность': f"{rule_info['precision']*100:.1f}%",
                    'Покрытие': f"{rule_info['recall']*100:.1f}%",
                    'Поймано': rule_info['caught']
                })

            df_rules = pd.DataFrame(rules_data)
            st.dataframe(df_rules, use_container_width=True)

    else:
        st.info("👆 Нажмите 'Запустить полный анализ' для начала")


def render_training_tab():
    """Вкладка обучения модели"""
    st.header("🤖 Обучение модели")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚙️ Настройки")

        train_path = st.text_input(
            "Путь к тренировочным данным",
            value=DATA_PATHS['train']
        )
        test_path = st.text_input(
            "Путь к тестовым данным",
            value=DATA_PATHS['test']
        )
        model_filename = st.text_input(
            "Имя файла модели",
            value="fraud_model.joblib"
        )

    with col2:
        st.markdown("### 🚀 Параметры")

        use_gpu = st.checkbox("Использовать GPU", value=True)
        st.info("GPU ускоряет обучение в 5-10 раз")

        # Проверка файлов
        train_exists = os.path.exists(train_path)
        test_exists = os.path.exists(test_path)

        if train_exists and test_exists:
            st.success("✅ Файлы данных найдены")
        else:
            if not train_exists:
                st.error(f"❌ Не найден: {train_path}")
            if not test_exists:
                st.error(f"❌ Не найден: {test_path}")

    if st.button("🚀 Начать обучение", type="primary", disabled=not (train_exists and test_exists)):
        with st.spinner("Обучение моделей..."):
            progress = st.progress(0)
            status = st.empty()

            # Симуляция прогресса
            status.text("📁 Загрузка данных...")
            progress.progress(20)

            status.text("🔧 Feature engineering...")
            progress.progress(40)

            status.text("🤖 Обучение моделей...")
            progress.progress(60)

            # Запуск обучения
            report = train_pipeline(
                train_path=train_path,
                test_path=test_path,
                model_filename=model_filename,
                use_gpu=use_gpu
            )

            progress.progress(100)
            status.text("✅ Обучение завершено!")

            # Отображение результатов
            st.success(
                f"✅ Лучшая модель: {report['best_model']} (AUC: {report['best_score']:.4f})")

            # Результаты по моделям
            st.markdown("### 📊 Результаты обучения")

            results_data = []
            for model_name, results in report['models_results'].items():
                results_data.append({
                    'Модель': model_name,
                    'CV AUC': f"{results['cv_mean_auc']:.4f} ± {results['cv_std_auc']:.4f}"
                })

            if 'test_results' in report:
                for model_name, test_res in report['test_results'].items():
                    for i, row in enumerate(results_data):
                        if row['Модель'] == model_name:
                            results_data[i]['Test AUC'] = f"{test_res['auc']:.4f}"
                            results_data[i]['Accuracy'] = f"{test_res['accuracy']:.4f}"

            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True)

            # Топ признаков
            if 'top_features' in report:
                st.markdown("### 🏆 Топ-10 важных признаков")

                features_df = pd.DataFrame(report['top_features'])
                fig = px.bar(
                    features_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Важность признаков"
                )
                st.plotly_chart(fig, use_container_width=True)


def render_prediction_tab(selected_model):
    """Вкладка предсказаний"""
    st.header("🔮 Предсказание мошенников")

    if not selected_model:
        st.error("❌ Сначала обучите модель во вкладке 'Обучение'")
        return

    # Выбор способа ввода
    input_method = st.radio(
        "Выберите способ ввода данных:",
        ["📁 Загрузить JSON файл", "✏️ Ввести вручную", "📋 Использовать пример"]
    )

    if input_method == "📁 Загрузить JSON файл":
        uploaded_file = st.file_uploader("Выберите JSON файл", type=['json'])

        if uploaded_file is not None:
            data = json.load(uploaded_file)

            if st.button("🔍 Анализировать", type="primary"):
                with st.spinner("Анализируем объекты..."):
                    predictor = FraudPredictor(selected_model)
                    results = predictor.predict(data)

                    st.session_state['predictions'] = results
                    st.success(f"✅ Проанализировано {len(results)} объектов")

    elif input_method == "✏️ Ввести вручную":
        col1, col2 = st.columns(2)

        with col1:
            account_id = st.text_input("Account ID", value="TEST001")
            building_type = st.selectbox(
                "Тип здания", ["Apartment", "House", "Other"])
            rooms = st.number_input(
                "Комнат", min_value=1, max_value=10, value=3)
            residents = st.number_input(
                "Жителей", min_value=1, max_value=10, value=2)
            area = st.number_input(
                "Площадь", min_value=0.0, max_value=500.0, value=75.0)

        with col2:
            st.markdown("**Потребление (кВт·ч):**")
            winter = st.number_input(
                "Зима (среднее)", min_value=0, max_value=2000, value=300)
            spring = st.number_input(
                "Весна (среднее)", min_value=0, max_value=2000, value=200)
            summer = st.number_input(
                "Лето (среднее)", min_value=0, max_value=2000, value=150)
            autumn = st.number_input(
                "Осень (среднее)", min_value=0, max_value=2000, value=250)

        if st.button("🔍 Анализировать", type="primary"):
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

            predictor = FraudPredictor(selected_model)
            results = predictor.predict(data)
            st.session_state['predictions'] = results

    elif input_method == "📋 Использовать пример":
        examples = {
            "🏪 Магазин (мошенник)": {
                "accountId": "SHOP001",
                "buildingType": "Apartment",
                "roomsCount": 2,
                "residentsCount": 1,
                "totalArea": 50,
                "consumption": {str(i): 600 for i in range(1, 13)},
                "address": "г. Краснодар"
            },
            "🏠 Честная семья": {
                "accountId": "FAM001",
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
            "Выберите пример:", list(examples.keys()))

        # Показываем данные примера
        st.json(examples[selected_example])

        if st.button("🔍 Анализировать пример", type="primary"):
            predictor = FraudPredictor(selected_model)
            results = predictor.predict([examples[selected_example]])
            st.session_state['predictions'] = results

    # Отображение результатов
    if 'predictions' in st.session_state:
        results = st.session_state['predictions']

        st.markdown("### 🎯 Результаты анализа")

        # Сводка
        report = FraudPredictor(selected_model).generate_report(results)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Проанализировано", report['summary']['total_analyzed'])
        with col2:
            st.metric("Мошенников", report['summary']['fraudsters_detected'])
        with col3:
            st.metric("% мошенников",
                      f"{report['summary']['fraud_rate']:.1f}%")

        # Детальные результаты
        st.markdown("### 📋 Детальные результаты")

        for result in results[:10]:  # Показываем первые 10
            if result['isCommercial']:
                card_class = "fraud-card"
                icon = "🚨"
            else:
                card_class = "safe-card"
                icon = "✅"

            st.markdown(f"""
            <div class="{card_class}">
                <h4>{icon} {result['accountId']}</h4>
                <p><strong>Вероятность мошенничества:</strong> {result['fraud_probability_percent']}</p>
                <p><strong>Уровень риска:</strong> {result['risk_level']}</p>
                <p><strong>Интерпретация:</strong> {result['interpretation']}</p>
                <p><strong>Паттерны:</strong></p>
                <ul>
                    <li>Потребление: {result['patterns']['consumption_level']}</li>
                    <li>Сезонность: {result['patterns']['seasonality']}</li>
                    <li>Стабильность: {result['patterns']['stability']}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Экспорт
        st.markdown("### 💾 Экспорт результатов")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Скачать JSON",
                data=json.dumps(results, ensure_ascii=False, indent=2),
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with col2:
            # CSV экспорт
            df_export = pd.DataFrame([{
                'accountId': r['accountId'],
                'isCommercial': r['isCommercial'],
                'fraud_probability': r['fraud_probability'],
                'risk_level': r['risk_level']
            } for r in results])

            csv = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Скачать CSV",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def render_monitoring_tab():
    """Вкладка мониторинга"""
    st.header("📈 Мониторинг системы")

    # Проверяем наличие файлов с результатами
    if os.path.exists('training_report.json'):
        with open('training_report.json', 'r', encoding='utf-8') as f:
            training_report = json.load(f)

        st.markdown("### 🤖 Статус моделей")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Лучшая модель", training_report['best_model'])
        with col2:
            st.metric("Best AUC", f"{training_report['best_score']:.4f}")
        with col3:
            st.metric("Признаков", training_report['total_features'])

        # График сравнения моделей
        if 'models_results' in training_report:
            models_data = []
            for model, results in training_report['models_results'].items():
                models_data.append({
                    'Модель': model,
                    'AUC': results['cv_mean_auc']
                })

            df_models = pd.DataFrame(models_data)

            fig = px.bar(
                df_models,
                x='Модель',
                y='AUC',
                title="Сравнение моделей по AUC",
                color='AUC',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Статистика предсказаний
    if 'predictions' in st.session_state:
        st.markdown("### 📊 Статистика последних предсказаний")

        predictor = FraudPredictor(st.sidebar.selectbox(
            "", os.listdir('.'), label_visibility="hidden"))
        report = predictor.generate_report(st.session_state['predictions'])

        # Распределение по рискам
        risk_data = []
        for level, info in report['risk_distribution'].items():
            risk_data.append({
                'Уровень': level,
                'Количество': info['count'],
                'Процент': info['percentage']
            })

        df_risk = pd.DataFrame(risk_data)

        fig = px.pie(
            df_risk,
            values='Количество',
            names='Уровень',
            title="Распределение по уровням риска",
            color_discrete_map={
                'HIGH': '#ff0000',
                'MEDIUM': '#ff9900',
                'LOW': '#ffcc00',
                'MINIMAL': '#00cc00'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📊 Данные мониторинга появятся после анализа объектов")


if __name__ == "__main__":
    main()
