@echo off
REM Build the Lirik Studio executables. Requires: pip install pyinstaller
REM Set FFMPEG_DIR to a folder holding ffmpeg.exe + ffprobe.exe to embed them.
setlocal
cd /d "%~dp0"
python build_exe.py %*
if errorlevel 1 (
  echo.
  echo Build gagal. Lihat pesan di atas.
  pause
  exit /b 1
)
echo.
echo Selesai. Jalankan: dist\LirikStudio\LirikStudio.exe
pause
