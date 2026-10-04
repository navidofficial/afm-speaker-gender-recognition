@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  NEW Reduced-model variants - RESUMABLE
REM  Uses the trained Cascade model from results_github\cascade_timit\best.pt
REM  (produced by run_all_paper.bat). Nothing before pole reduction is retrained.
REM
REM    A   task-aware ERB-warped weighted mean-shift
REM    B   learnable L0 gates on the Cascade filters
REM    C   clustering of effective Cascade responses
REM    D   paper Mean-shift filters + distillation + trainable f1/f2
REM    AD  A filters + distillation + trainable f1/f2
REM
REM  Each variant: TIMIT (50 epochs) -> fine-tune LibriSpeech (20) -> RAVDESS / VoxCeleb1.
REM  Close anytime; re-run this bat to continue from last.pt.
REM ============================================================

cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set TQDM_MININTERVAL=0.3
set "RES=results_github"
set "AFM_RESULTS_DIR=%CD%\%RES%"
set "PYARGS=--device cuda --batch-size 128"
set VARIANTS=A B C D AD

echo.
echo ============================================================
echo  NEW REDUCED PIPELINE (RESUME)
echo  Project: %CD%
echo  Results: %AFM_RESULTS_DIR%\reduced_^<variant^>_timit
echo  Start:   %DATE% %TIME%
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] python not found on PATH.
  pause
  exit /b 1
)

if not exist "%RES%\cascade_timit\best.pt" (
  echo [ERROR] Missing %RES%\cascade_timit\best.pt
  echo         Run run_all_paper.bat first ^(at least the Cascade TIMIT stage^).
  pause
  exit /b 1
)

for %%V in (%VARIANTS%) do (
  echo.
  echo ============================================================
  echo  [TIMIT] reduced variant %%V
  echo ============================================================
  set "NEED=1"
  if exist "%RES%\reduced_%%V_timit\STAGE_DONE.json" (
    if exist "%RES%\reduced_%%V_timit\metrics_timit_test.json" set "NEED=0"
  )
  if "!NEED!"=="0" (
    echo [skip] %RES%\reduced_%%V_timit already complete
  ) else (
    if exist "%RES%\reduced_%%V_timit\last.pt" echo [resume] found %RES%\reduced_%%V_timit\last.pt - continuing
    python -u run_new_reduced.py --variant %%V --stage timit %PYARGS%
    if errorlevel 1 (
      echo [ERROR] TIMIT failed for variant %%V - re-run this bat to continue
      pause
      exit /b 1
    )
  )

  echo.
  echo ============================================================
  echo  [FINETUNE] reduced variant %%V + eval RAVDESS / VoxCeleb1-test
  echo ============================================================
  set "NEED=1"
  if exist "%RES%\reduced_%%V_timit_finetune\metrics_ravdess.json" (
    if exist "%RES%\reduced_%%V_timit_finetune\metrics_voxceleb1_test.json" set "NEED=0"
  )
  if "!NEED!"=="0" (
    echo [skip] %RES%\reduced_%%V_timit_finetune already complete
  ) else (
    python -u run_new_reduced.py --variant %%V --stage finetune %PYARGS%
    if errorlevel 1 (
      echo [ERROR] Fine-tune failed for variant %%V - re-run this bat to continue
      pause
      exit /b 1
    )
  )
)

echo.
echo ============================================================
echo  [TABLES] new_reduced_comparison + comparison_table
echo ============================================================
python -u run_new_reduced.py --stage table
python -u scripts\make_comparison_table.py --results-dir "%RES%"

echo.
echo ============================================================
echo  DONE / UP TO DATE
echo  End: %DATE% %TIME%
echo  Comparison: %AFM_RESULTS_DIR%\new_reduced_comparison
echo ============================================================
echo.
pause
endlocal
exit /b 0
