@echo off
REM XLSR-Mamba 仅评估脚本（跳过训练）

echo ========================================
echo   XLSR-Mamba - Evaluation Only
echo ========================================

REM 激活conda环境
call D:\code\anaconda\Scripts\activate.bat antimamba02

REM 仅运行评估（跳过训练）
python main.py --algo 5 --batch_size 2 --num_epochs 1 --database_path . --protocols_path . --train False

echo.
echo ========================================
echo   Evaluation Complete!
echo ========================================
pause


