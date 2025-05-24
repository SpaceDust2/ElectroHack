@echo off
chcp 65001 >nul
echo ====================================
echo 🔍 ML Fraud Detection System
echo    Минималистичная версия
echo ====================================
echo.

:: Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден! Установите Python 3.8+
    pause
    exit /b 1
)

:: Активация виртуального окружения если есть
if exist venv\Scripts\activate.bat (
    echo 🔧 Активация виртуального окружения...
    call venv\Scripts\activate.bat
)

:: Проверка зависимостей
echo 📦 Проверка зависимостей...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📥 Установка зависимостей...
    pip install -r requirements.txt
)

:: Копирование данных если нужно
if not exist data (
    echo 📁 Создание папки data...
    mkdir data
    echo ⚠️ Поместите файлы dataset_train.json и dataset_test.json в папку data/
)

echo.
echo 🚀 Запуск приложения...
echo.
echo Откройте браузер: http://localhost:8501
echo.

:: Запуск Streamlit
streamlit run app.py --server.port 8501 --server.headless true

pause 