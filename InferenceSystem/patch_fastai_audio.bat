@echo off
REM Script to patch fastai_audio for Python 3.11+ compatibility on Windows

echo Applying Python 3.11 compatibility patch to fastai_audio...

REM Find the fastai_audio installation directory
for /f "delims=" %%i in ('python -c "import audio.data; import os; print(os.path.abspath(audio.data.__file__))"') do set AUDIO_DATA_FILE=%%i

if "%AUDIO_DATA_FILE%"=="" (
    echo Error: Could not find audio.data module
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
