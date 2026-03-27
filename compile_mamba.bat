@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Compiling mamba-ssm on Windows
echo ========================================
echo.

REM 设置编译环境
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM 设置 CUDA
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%PATH%"

REM 激活环境
call D:\code\anaconda\Scripts\activate.bat antimamba02

echo.
echo Step 1: Installing causal-conv1d==1.1.3.post1
echo This will take 5-10 minutes...
echo.

pip install causal-conv1d==1.1.3.post1 --no-cache-dir > causal_install.log 2>&1

if %ERRORLEVEL% EQU 0 (
    echo [OK] causal-conv1d installed successfully
) else (
    echo [FAIL] causal-conv1d installation failed
    echo Check causal_install.log for details
    pause
    exit /b 1
)

echo.
echo Step 2: Installing mamba-ssm==1.1.4
echo This will take 5-10 minutes...
echo.

pip install mamba-ssm==1.1.4 --no-cache-dir > mamba_install.log 2>&1

if %ERRORLEVEL% EQU 0 (
    echo [OK] mamba-ssm installed successfully
) else (
    echo [FAIL] mamba-ssm installation failed
    echo Check mamba_install.log for details
    pause
    exit /b 1
)

echo.
echo ========================================
echo Testing installation...
echo ========================================

python -c "import causal_conv1d; import mamba_ssm; print('SUCCESS! mamba-ssm version:', mamba_ssm.__version__)"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Installation completed successfully!
    echo ========================================
) else (
    echo.
    echo Installation completed but import failed
    echo Check the log files for details
)

echo.
pause




