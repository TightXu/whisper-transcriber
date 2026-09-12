@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  Single launcher. All logic lives in whisper_transcribe.py.
rem    Double-click            -> interactive prompt
rem    bat "file.mp3"          -> transcribe directly
rem ============================================================
set "PROG_DIR=%~dp0"
set "PYEXE=%PROG_DIR%.venv\Scripts\python.exe"
set "SCRIPT=%PROG_DIR%whisper_transcribe.py"
rem  Keep this file pure ASCII: chcp 65001 + non-ASCII comments make
rem  cmd re-read the file with misaligned byte offsets (garbled parsing).
rem  All bytecode caches go into .venv\pycache via PYTHONPYCACHEPREFIX.
set "PYTHONPYCACHEPREFIX=%PROG_DIR%.venv\pycache"

rem ---- Files extracted from a browser download carry a "came from the
rem      internet" mark, and Windows then warns on every launch. The
rem      helper clears it silently and only prints when it removed
rem      something. (dir /r | findstr proved unreliable for detecting
rem      that stream, so the check lives inside the PowerShell call.) ----
call :unblockmotw

rem ---- venv present AND able to start? (a venv copied from
rem      another machine may be broken until claimed) ----
if exist "%PYEXE%" (
    "%PYEXE%" -c "import sys" >nul 2>nul && goto :run
    echo [setup] .venv found but not working, trying to claim it ...
)

set "BOOT="
call :findpython
if defined BOOT goto :havepy

rem ---- No python at all: offer automatic install (up to 3 attempts).
rem      Default 3.12.10: same minor version as the bundled .venv, so a
rem      copied .venv can be claimed instead of reinstalling ~470MB deps. ----
set /p "ANS=No Python found. Install Python 3.12.10 automatically? [Y/n]: "
if /i "%ANS%"=="n" goto :nopy
set "PYEXE_DL=%TEMP%\python-3.12.10-amd64.exe"
set /a PYTRIES=0

:dlpy
set /a PYTRIES+=1
echo [setup] Downloading Python 3.12.10 installer (attempt %PYTRIES%/3) ...
call :dlfile "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" "%PYEXE_DL%"
if exist "%PYEXE_DL%" goto :installpy
call :dlfile "https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe" "%PYEXE_DL%"
if exist "%PYEXE_DL%" goto :installpy
if %PYTRIES% LSS 3 (
    echo [setup] Download failed, retrying in 3s ...
    ping -n 4 127.0.0.1 >nul
    goto :dlpy
)
goto :nopy

:installpy
echo [setup] Installing Python silently (user scope, add to PATH) ...
"%PYEXE_DL%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
del "%PYEXE_DL%" >nul 2>nul
rem  PATH of this session is stale after silent install: findpython
rem  also probes fixed install locations, so it still finds it.
call :findpython
if not defined BOOT goto :nopy

:havepy
if exist "%PYEXE%" (
    rem  Broken .venv present: run the script with the bootstrap
    rem  python and let it claim/rebuild the venv itself.
    "%BOOT%" "%SCRIPT%" %*
    goto :end
)
echo [setup] First run: creating .venv ...
"%BOOT%" -m venv "%PROG_DIR%.venv"
if exist "%PYEXE%" goto :deps
rem  venv module missing (rare): use virtualenv
"%BOOT%" -m pip install --quiet virtualenv
"%BOOT%" -m virtualenv "%PROG_DIR%.venv"
if exist "%PYEXE%" goto :deps
echo [ERROR] Could not create .venv automatically.
echo   Python found: %BOOT%
echo   Try: delete the .venv folder and rerun, or create it manually:
echo     "%BOOT%" -m venv "%PROG_DIR%.venv"
pause
exit /b 1

:deps
echo [setup] Installing dependencies (first run only) ...
"%PYEXE%" -m pip install --progress-bar on --upgrade pip
rem  Probe the official index first (12s cap, abort if slower than 5KB/s
rem  for 5s): a reachable-but-stalled CDN never returns an error, so
rem  "retry on failure" alone can hang forever. If it fails, start from a
rem  mirror instead; all three sources serve as the retry ladder, each
rem  tried at most once per run.
set "PIP1="
set "PIP2=-i https://mirrors.aliyun.com/pypi/simple/"
set "PIP3=-i https://pypi.tuna.tsinghua.edu.cn/simple/"
curl -s -o NUL --max-time 12 --speed-time 5 --speed-limit 5000 "https://pypi.org/simple/faster-whisper/" >nul 2>nul
if errorlevel 1 (
    echo [setup] Official PyPI unreachable or stalled, starting from a mirror ...
    set "PIP1=-i https://mirrors.aliyun.com/pypi/simple/"
    set "PIP2=-i https://pypi.tuna.tsinghua.edu.cn/simple/"
    set "PIP3="
)
set "PIPTRY=0"

:pipinst
set /a PIPTRY+=1
if %PIPTRY%==1 set "PIPIDX=%PIP1%"
if %PIPTRY%==2 set "PIPIDX=%PIP2%"
if %PIPTRY%==3 set "PIPIDX=%PIP3%"
echo [setup] Installing dependencies (attempt %PIPTRY%/3) ...
"%PYEXE%" -m pip install --progress-bar on %PIPIDX% faster-whisper ctranslate2 openvino openvino-genai
if not errorlevel 1 goto :run
if %PIPTRY% LSS 3 goto :pipnext
echo [setup] Dependency install failed after 3 attempts.
echo          The app can still repair itself - type /fix inside it.
goto :run

:pipnext
echo [setup] Install failed, trying the next source ...
ping -n 4 127.0.0.1 >nul
goto :pipinst

:run
"%PYEXE%" "%SCRIPT%" %*
goto :end

:nopy
echo.
echo [setup] Python is required but could not be installed automatically.
echo   1. Manual install: https://www.python.org/downloads/windows/
echo      After installing, close this window and double-click the .bat again.
echo   2. Press R to retry the automatic download (nothing else is re-downloaded).
echo   3. Close this window to exit.
set "ANS="
set /p "ANS=Choose [R=retry / Enter=exit]: "
if /i "%ANS%"=="r" goto :retrypy
pause
exit /b 1

:retrypy
set /a PYTRIES=0
goto :dlpy

:end
echo.
pause
endlocal
exit /b 0

rem ---- find a python 3.11+ and set BOOT (full path preferred) ----
:findpython
set "BOOT="
for %%V in (3.14 3.13 3.12 3.11) do (
    py -%%V -c "import sys" >nul 2>nul && (
        for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)" 2^>nul') do (
            set "BOOT=%%P"
            exit /b 0
        )
    )
)
python -c "import sys" >nul 2>nul && (
    for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)" 2^>nul') do (
        set "BOOT=%%P"
        exit /b 0
    )
)
rem ---- fixed locations (PATH-independent; e.g. just installed).
rem      cmd for-set globs only match the last path component, so
rem      these must be literal paths, one per version. ----
for %%P in ("%LocalAppData%\Programs\Python\Python311\python.exe" "C:\Python311\python.exe" "%LocalAppData%\Programs\Python\Python312\python.exe" "C:\Python312\python.exe" "%LocalAppData%\Programs\Python\Python313\python.exe" "C:\Python313\python.exe" "%LocalAppData%\Programs\Python\Python314\python.exe" "C:\Python314\python.exe") do (
    if exist "%%~P" (
        set "BOOT=%%~P"
        exit /b 0
    )
)
exit /b 1

rem ---- strip the "downloaded from the internet" mark (Mark-of-the-Web).
rem      Runs on every start; prints a line only when it cleared something. ----
:unblockmotw
powershell -NoProfile -ExecutionPolicy Bypass -Command "$c=0; Get-ChildItem -LiteralPath '%~dp0.' -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object { if ((Get-Item -LiteralPath $_.FullName -Stream * -ErrorAction SilentlyContinue).Stream -contains 'Zone.Identifier') { Unblock-File -LiteralPath $_.FullName -ErrorAction SilentlyContinue; $c++ } }; if ($c -gt 0) { Write-Output ('[setup] Cleared network-origin marks on ' + $c + ' file(s).') }" 2>nul
exit /b 0


rem ---- download helper: curl, stall-guarded, with a progress bar ----
:dlfile
if exist "%~2" del "%~2" >nul 2>nul
curl -L --fail --max-time 900 --speed-time 30 --speed-limit 10000 --progress-bar -o "%~2" "%~1"
if not exist "%~2" exit /b 1
exit /b 0
