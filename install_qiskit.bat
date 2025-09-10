@echo off


REM 
where conda >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Miniconda not found. Downloading...
    powershell -Command "Invoke-WebRequest -Uri https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile miniconda-installer.exe"
    echo Installing Miniconda...
    start /wait miniconda-installer.exe /S /AddToPath=1 /RegisterPython=0
    del miniconda-installer.exe
    call %USERPROFILE%\Miniconda3\Scripts\activate.bat
) ELSE (
    echo Miniconda is already installed.
)

REM 
echo Creating conda environment 'qiskit'...
conda env list | findstr "qiskit" >nul
IF %ERRORLEVEL% NEQ 0 (
    conda create -n qiskit python=3.11 -y
) ELSE (
    echo Environment 'qiskit' already exists.
)

REM 
echo Installing Qiskit packages into 'qiskit'...
call conda activate qiskit
pip install --upgrade pip
pip install qiskit-aer qiskit-ibm-runtime "qiskit[visualization]" jupyter matplotlib numpy

REM 
where code >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo VS Code not found. Downloading...
    powershell -Command "Invoke-WebRequest -Uri https://update.code.visualstudio.com/latest/win32-x64-user/stable -OutFile vscode-installer.exe"
    echo Installing VS Code...
    start /wait vscode-installer.exe /silent
    del vscode-installer.exe
) ELSE (
    echo VS Code is already installed.
)

REM
echo Writing Hello Quantum World notebook...
mkdir "%USERPROFILE%\QiskitNotebooks"
echo { "cells": [ { "cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": [ "from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)\nqc.measure_all()\nqc.draw('mpl')" ] } ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5 } > "%USERPROFILE%\QiskitNotebooks\hello_quantum.ipynb"

echo ==================================================
echo Installation Complete!
echo - To start coding:
echo   1. Open VS Code
echo   2. Open folder: QiskitNotebooks
echo   3. Select Python interpreter: 'qiskit'
echo   4. Run hello_quantum.ipynb
echo ==================================================
pause
