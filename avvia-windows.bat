@echo off
title Portfolio Optimizer
echo.
echo  Avvio Portfolio Optimizer...
echo.

REM Controlla se Python e' installato
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERRORE: Python non trovato.
    echo  Scaricalo da https://www.python.org/downloads/
    echo  Assicurati di spuntare "Add Python to PATH" durante l'installazione.
    pause
    exit /b 1
)

REM Avvia il server in background e apri il browser
start "" python "%~dp0server.py"
timeout /t 2 /nobreak >nul
start "" "%~dp0portfolio-markowitz.html"

echo  Server avviato! Il browser si apre automaticamente.
echo  Lascia aperta questa finestra mentre usi l'app.
echo.
pause
