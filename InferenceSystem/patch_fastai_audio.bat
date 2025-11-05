@echo off
REM Script to patch fastai_audio for Python 3.11+ compatibility on Windows

echo Applying Python 3.11 compatibility patch to fastai_audio...

REM Find the site-packages directory
for /f "delims=" %%i in ('python -c "import sys; import site; print([p for p in sys.path if 'site-packages' in p][0])"') do set AUDIO_PACKAGE_DIR=%%i

if "%AUDIO_PACKAGE_DIR%"=="" (
    echo Error: Could not find site-packages directory
    exit /b 1
)

REM Build path to audio/data.py
set AUDIO_DATA_FILE=%AUDIO_PACKAGE_DIR%\audio\data.py

if not exist "%AUDIO_DATA_FILE%" (
    echo Error: Could not find audio/data.py at %AUDIO_DATA_FILE%
    exit /b 1
)

echo Found audio/data.py at: %AUDIO_DATA_FILE%

REM Create a backup
copy "%AUDIO_DATA_FILE%" "%AUDIO_DATA_FILE%.bak" > nul

REM Apply the fix using PowerShell
powershell -Command "(Get-Content '%AUDIO_DATA_FILE%') -replace 'from dataclasses import dataclass, asdict', 'from dataclasses import dataclass, asdict, field' | Set-Content '%AUDIO_DATA_FILE%'"
powershell -Command "(Get-Content '%AUDIO_DATA_FILE%') -replace 'sg_cfg: SpectrogramConfig = SpectrogramConfig\(\)', 'sg_cfg: SpectrogramConfig = field(default_factory=SpectrogramConfig)' | Set-Content '%AUDIO_DATA_FILE%'"

echo Patch applied successfully!
echo Backup saved to: %AUDIO_DATA_FILE%.bak
