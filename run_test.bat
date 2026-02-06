@echo off
chcp 65001 >nul
setlocal

echo ============================================================
echo  GraphRAG - Schnelltest (5 Beispiele)
echo ============================================================
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat

echo Starte Test-Evaluation...
echo.

python scripts/run_comprehensive_evaluation.py ^
    --strategies 1,3,4 ^
    --sample-size 5 ^
    --benchmark hotpotqa ^
    --output-dir results/comprehensive

echo.
echo ============================================================
echo  Test abgeschlossen!
echo ============================================================
pause
