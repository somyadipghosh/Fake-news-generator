@echo off
echo ========================================
echo  Fake News Detection System
echo  Installation Script
echo ========================================
echo.
echo This script will install all required dependencies
echo.
pause

echo.
echo [1/4] Installing Python packages...
pip install -r requirements.txt

echo.
echo [2/4] Downloading spaCy language model...
python -m spacy download en_core_web_sm

echo.
echo [3/4] Setting up NLTK data...
python setup_nltk.py

echo.
echo [4/4] Creating directories...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "models\saved_models" mkdir "models\saved_models"
if not exist "results" mkdir "results"

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Train a model: python train_example.py
echo 2. Run Streamlit app: streamlit run streamlit_app.py
echo    or just double-click: run_streamlit.bat
echo.
echo For more information, see STREAMLIT_README.md
echo.
pause
