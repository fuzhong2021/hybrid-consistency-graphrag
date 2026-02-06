@echo off
chcp 65001 >nul
setlocal

echo ============================================================
echo  GraphRAG - Comprehensive Evaluation (4 Strategien)
echo ============================================================
echo.
echo Strategien:
echo   1. Intrinsische Evaluation (synthetische Fehler)
echo   2. Manuelle Annotation (Export fuer Human Review)
echo   3. Graph-Qualitaet (Vorher/Nachher)
echo   4. Ablation Study (Information Quality)
echo.
set /p CONFIRM="Fortfahren? (j/n): "
if /i "%CONFIRM%" neq "j" exit /b 0

cd /d "%~dp0"
call venv\Scripts\activate.bat

echo.
echo Starte Evaluation um %date% %time%
echo.

python scripts/run_comprehensive_evaluation.py ^
    --strategies 1,2,3,4 ^
    --sample-size 200 ^
    --benchmark hotpotqa ^
    --output-dir results/comprehensive

echo.
echo ============================================================
echo  Evaluation abgeschlossen um %date% %time%
echo ============================================================
echo.
echo Ergebnisse in: results/comprehensive/
echo   - intrinsic_results.json
echo   - annotation_sample.json
echo   - graph_quality_results.json
echo   - ablation_results.json
echo   - latex_tables.tex
echo.
pause
