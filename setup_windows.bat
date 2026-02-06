@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================================
echo  GraphRAG Masterarbeit - Windows Setup Script
echo ============================================================
echo.

:: Farben für Ausgabe
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "NC=[0m"

:: Aktuelles Verzeichnis
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo [1/8] Pruefe Python Installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%FEHLER: Python nicht gefunden!%NC%
    echo Bitte installiere Python 3.11+ von https://python.org
    echo Oder: winget install Python.Python.3.11
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo %GREEN%OK: Python %PYTHON_VERSION% gefunden%NC%
echo.

echo [2/8] Pruefe Ollama Installation...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%WARNUNG: Ollama nicht erreichbar!%NC%
    echo.
    echo Bitte installiere Ollama:
    echo   1. Download: https://ollama.com/download/windows
    echo   2. Installieren und starten
    echo   3. Terminal oeffnen: ollama pull llama3.1:8b
    echo.
    set /p CONTINUE="Trotzdem fortfahren? (j/n): "
    if /i "!CONTINUE!" neq "j" exit /b 1
) else (
    echo %GREEN%OK: Ollama laeuft%NC%
)
echo.

echo [3/8] Pruefe ob llama3.1:8b vorhanden ist...
curl -s http://localhost:11434/api/tags | findstr "llama3.1:8b" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%Modell llama3.1:8b nicht gefunden. Lade herunter...%NC%
    ollama pull llama3.1:8b
    if %errorlevel% neq 0 (
        echo %RED%FEHLER beim Laden des Modells%NC%
        pause
        exit /b 1
    )
)
echo %GREEN%OK: llama3.1:8b verfuegbar%NC%
echo.

echo [4/8] Erstelle Virtual Environment...
if not exist "venv" (
    python -m venv venv
    echo %GREEN%OK: venv erstellt%NC%
) else (
    echo %GREEN%OK: venv existiert bereits%NC%
)
echo.

echo [5/8] Aktiviere venv und installiere Dependencies...
call venv\Scripts\activate.bat

python -m pip install --upgrade pip --quiet
echo Installiere Requirements (kann einige Minuten dauern)...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo %RED%FEHLER bei pip install%NC%
    pause
    exit /b 1
)
echo %GREEN%OK: Dependencies installiert%NC%
echo.

echo [6/8] Pruefe Neo4j Verbindung...
python -c "from neo4j import GraphDatabase; d=GraphDatabase.driver('bolt://localhost:7687',auth=('neo4j','masterarbeit2024')); d.verify_connectivity(); print('OK'); d.close()" 2>nul
if %errorlevel% neq 0 (
    echo %YELLOW%WARNUNG: Neo4j nicht erreichbar!%NC%
    echo.
    echo Option A - Neo4j Desktop:
    echo   1. Download: https://neo4j.com/download/
    echo   2. Neue Datenbank erstellen
    echo   3. Passwort: masterarbeit2024
    echo.
    echo Option B - Docker:
    echo   docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/masterarbeit2024 neo4j:5.15.0
    echo.
    echo Du kannst auch ohne Neo4j testen mit --no-neo4j
    echo.
) else (
    echo %GREEN%OK: Neo4j verbunden%NC%
)
echo.

echo [7/8] Teste Ollama Client...
python -c "from src.llm.ollama_client import OllamaClient; c=OllamaClient(model='llama3.1:8b'); print('Modelle:', c.list_models())"
if %errorlevel% neq 0 (
    echo %RED%FEHLER: Ollama Client Test fehlgeschlagen%NC%
    pause
    exit /b 1
)
echo %GREEN%OK: Ollama Client funktioniert%NC%
echo.

echo [8/8] Erstelle results Ordner...
if not exist "results" mkdir results
echo %GREEN%OK: results Ordner bereit%NC%
echo.

echo ============================================================
echo %GREEN% SETUP ABGESCHLOSSEN!%NC%
echo ============================================================
echo.
echo Naechste Schritte:
echo.
echo   1. Test mit 3 Beispielen:
echo      run_test.bat
echo.
echo   2. Volle Evaluation (500 Beispiele, ~8-16h):
echo      run_evaluation.bat
echo.
echo   3. Fortschritt pruefen:
echo      type results\evaluation.log
echo.
echo ============================================================
pause
