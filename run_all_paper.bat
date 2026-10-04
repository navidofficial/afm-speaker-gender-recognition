@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  Full paper pipeline (SGR_AFM-aligned code) - RESUMABLE
REM  Close anytime (lid / Ctrl+C / power). Re-run this bat:
REM    - finished stages  -> skip (STAGE_DONE.json + metrics)
REM    - mid-training     -> continue from last.pt (next epoch)
REM    - mid-eval         -> skip done metrics_*.json, finish rest
REM
REM  Results go to results_github\ (old results\ is left untouched).
REM
REM  NOTE: Do NOT chain "if exist A if exist B (...) else (...)"
REM  in cmd.exe - else binds only to the last IF and is skipped
REM  when A is missing.
REM ============================================================

cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set TQDM_MININTERVAL=0.3
set "RES=results_github"
set "AFM_RESULTS_DIR=%CD%\%RES%"
set "PYARGS=--device cuda --batch-size 128 --num-workers 0 --resume"

echo.
echo ============================================================
echo  AFM Speaker Gender Recognition - FULL PIPELINE (RESUME)
echo  Project: %CD%
echo  Results: %AFM_RESULTS_DIR%
echo  Start:   %DATE% %TIME%
echo.
echo  Tip: stop anytime. Re-run this bat to continue from last.pt
echo  Force one stage from scratch:
echo    python run_pipeline.py --stage timit --frontend free --force --no-resume
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] python not found on PATH.
  pause
  exit /b 1
)

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if errorlevel 1 (
  echo [ERROR] PyTorch import failed. Install: pip install -r requirements.txt
  pause
  exit /b 1
)

REM ---- TIMIT for each frontend (50 epochs) ----
set FRONTENDS=free mel sinc sinc2 gamma gauss cascade

for %%F in (%FRONTENDS%) do (
  echo.
  echo ============================================================
  echo  [TIMIT] frontend=%%F  ^(skip if done / resume last.pt^)
  echo ============================================================
  set "NEED=1"
  if exist "%RES%\%%F_timit\STAGE_DONE.json" (
    if exist "%RES%\%%F_timit\metrics_timit_test.json" set "NEED=0"
  )
  if "!NEED!"=="0" (
    echo [skip] %RES%\%%F_timit already complete
  ) else (
    if exist "%RES%\%%F_timit\last.pt" echo [resume] found %RES%\%%F_timit\last.pt - continuing
    python -u run_pipeline.py --stage timit --frontend %%F %PYARGS%
    if errorlevel 1 (
      echo [ERROR] TIMIT failed for %%F - re-run this bat to continue from last.pt
      pause
      exit /b 1
    )
  )
)

REM ---- Reduced (mean-shift on cascade poles; needs cascade TIMIT checkpoint) ----
echo.
echo ============================================================
echo  [REDUCED] Mean-shift pruned sinc2 filter bank on TIMIT
echo ============================================================
set "NEED=1"
if exist "%RES%\reduced_timit\STAGE_DONE.json" (
  if exist "%RES%\reduced_timit\metrics_timit_test.json" set "NEED=0"
)
if "!NEED!"=="0" (
  echo [skip] reduced already complete
) else (
  if exist "%RES%\reduced_timit\last.pt" echo [resume] found %RES%\reduced_timit\last.pt - continuing
  python -u run_pipeline.py --stage reduced --frontend cascade %PYARGS%
  if errorlevel 1 (
    echo [ERROR] Reduced failed - re-run this bat to continue from last.pt
    pause
    exit /b 1
  )
)

REM ---- Fine-tune (20 epochs) + eval RAVDESS / VoxCeleb1 (paper Tables 7-8; Mel skipped) ----
set FT_FRONTENDS=free sinc sinc2 gamma gauss cascade reduced

for %%F in (%FT_FRONTENDS%) do (
  echo.
  echo ============================================================
  echo  [FINETUNE] %%F + eval RAVDESS / VoxCeleb1-test
  echo ============================================================
  set "NEED=1"
  if exist "%RES%\%%F_timit_finetune\STAGE_DONE.json" (
    if exist "%RES%\%%F_timit_finetune\metrics_ravdess.json" (
      if exist "%RES%\%%F_timit_finetune\metrics_voxceleb1_test.json" set "NEED=0"
    )
  )
  if "!NEED!"=="0" (
    echo [skip] %RES%\%%F_timit_finetune already complete
  ) else (
    if not exist "%RES%\%%F_timit\best.pt" (
      echo [ERROR] Missing TIMIT checkpoint: %RES%\%%F_timit\best.pt
      pause
      exit /b 1
    )
    if exist "%RES%\%%F_timit_finetune\last.pt" echo [resume] found %RES%\%%F_timit_finetune\last.pt - continuing
    echo [run] starting fine-tune for %%F ...
    python -u run_pipeline.py --stage finetune --frontend %%F %PYARGS%
    if errorlevel 1 (
      echo [ERROR] Fine-tune failed for %%F - re-run this bat to continue from last.pt
      pause
      exit /b 1
    )
  )
)

REM ---- Pole reduction figure + comparison table ----
echo.
echo ============================================================
echo  [FIGURES / TABLE] pole reduction + comparison_table.csv / .md / .png
echo ============================================================
python -u scripts\plot_pole_reduction.py
python -u scripts\make_comparison_table.py --results-dir "%RES%"

echo.
echo ============================================================
echo  DONE / UP TO DATE
echo  End: %DATE% %TIME%
echo  Results: %AFM_RESULTS_DIR%
echo ============================================================
echo.
pause
endlocal
exit /b 0
