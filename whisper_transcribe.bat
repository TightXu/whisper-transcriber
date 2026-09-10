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

rem ---- venv present AND able to start? (a venv copied from
rem      another machine may be broken until claimed) ----
if exist "%PYEXE%" (
    "%PYEXE%" -c "import sys" >nul 2>nul && goto :run
    echo [setup] .venv found but not working, trying to claim it ...
)

set "BOOT="
call :findpython
if defined BOOT goto :havepy

rem ---- No python at all: offer automatic install ----
rem  Default 3.12.10: same minor version as the bundled .venv, so a
rem  copied .venv can be claimed instead of reinstalling ~470MB deps.
set /p "ANS=No Python found. Install Python 3.12.10 automatically? [Y/n]: "
if /i "%ANS%"=="n" goto :nopy
echo [setup] Downloading Python 3.12.10 installer ...
set "PYEXE_DL=%TEMP%\python-3.12.10-amd64.exe"
call :dlfile "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" "%PYEXE_DL%"
if not exist "%PYEXE_DL%" call :dlfile "https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe" "%PYEXE_DL%"
if not exist "%PYEXE_DL%" goto :nopy
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
"%PYEXE%" -m pip install --quiet --upgrade pip
rem  Preflight the official index first: curl with a 12s cap, aborting if
rem  it runs slower than 5KB/s for 5s. A reachable-but-stalled CDN never
rem  returns an error, so "retry on failure" alone can hang forever.
set "PIPIDX="
curl -s -o NUL --max-time 12 --speed-time 5 --speed-limit 5000 "https://pypi.org/simple/faster-whisper/" >nul 2>nul
if errorlevel 1 (
    echo [setup] Official PyPI unreachable/slow, using mirror ...
    set "PIPIDX=-i https://mirrors.aliyun.com/pypi/simple/"
)
"%PYEXE%" -m pip install --quiet %PIPIDX% faster-whisper ctranslate2 openvino openvino-genai
if errorlevel 1 (
    echo [setup] Install failed, retrying via mirror ...
    "%PYEXE%" -m pip install --quiet -i https://mirrors.aliyun.com/pypi/simple/ faster-whisper ctranslate2 openvino openvino-genai
)
goto :run

:run
"%PYEXE%" "%SCRIPT%" %*
goto :end

:nopy
echo [ERROR] Python is required. Install Python 3.11+ then rerun.
echo Download: https://www.python.org/downloads/
pause
exit /b 1

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

rem ---- download helper: curl with fallbacks ----
:dlfile
curl -L -m 900 -o "%~2" "%~1" >nul 2>nul
exit /b 0
